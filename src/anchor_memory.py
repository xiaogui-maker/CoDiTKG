import torch
import torch.nn as nn
import torch.nn.functional as F


class ATCEncoder(nn.Module):

    def __init__(
        self,
        h_dim: int,
        num_rels: int,
        K: int = 6,
        alpha_s: float = 0.7,
        alpha_l: float = 0.3,
        decay_s: float = 1.0,
        decay_l: float = 0.1,
        anchor_loss_lambda: float = 0.5,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert abs(alpha_s + alpha_l - 1.0) < 1e-6, 

        self.h_dim = h_dim
        self.num_rels = num_rels
        self.K = K
        self.anchor_loss_lambda = anchor_loss_lambda

    
        self._hawkes_logits = nn.Parameter(
            torch.tensor([alpha_s, alpha_l]).log()
        )
        self.log_decay_s = nn.Parameter(torch.tensor(decay_s).log())
        self.log_decay_l = nn.Parameter(torch.tensor(decay_l).log())
        self.mu = nn.Parameter(torch.zeros(1))   

        self.W_Q = nn.Linear(h_dim, h_dim, bias=False)
        self.W_K = nn.Linear(h_dim, h_dim, bias=False)
        self.W_V = nn.Linear(h_dim, h_dim, bias=False)

        self.phi = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim * 2, h_dim),
        )

        self.w_r = nn.Linear(h_dim * 2, 1, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_ent = nn.LayerNorm(h_dim)
        self.layer_norm_rel = nn.LayerNorm(h_dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_Q, self.W_K, self.W_V]:
            nn.init.xavier_uniform_(m.weight)
        for layer in self.phi:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.w_r.weight)
        nn.init.zeros_(self.w_r.bias)

    def _hawkes_intensity(
        self,
        event_times: torch.Tensor,  
        query_time: float,          
    ) -> torch.Tensor:
        
        alpha = F.softmax(self._hawkes_logits, dim=0)   # [α_s, α_l]
        decay_s = self.log_decay_s.exp().clamp(min=1e-3)
        decay_l = self.log_decay_l.exp().clamp(min=1e-3)

        delta = (query_time - event_times).clamp(min=0.0)   # [L]，Δt ≥ 0

        g_s = torch.exp(-decay_s * delta)
        g_l = torch.exp(-decay_l * delta)

        intensity = self.mu + alpha[0] * g_s + alpha[1] * g_l   # [L]

        intensity = F.softmax(intensity, dim=0)
        return intensity

    def _select_anchors(
        self,
        intensity: torch.Tensor,   # [L]
    ) -> torch.Tensor:

        K = min(self.K, intensity.shape[0])
        _, indices = torch.topk(intensity, K)    # [K]
        return indices

    def forward(
        self,
        entity_history_seq: list,    
        relation_history_seq: list,  
        h_current: torch.Tensor,     # [num_ents, h_dim]  
        r_current: torch.Tensor,     # [num_rels*2, h_dim] 
        timestamps: torch.Tensor | None = None, 
    ):
        L = len(entity_history_seq)
        device = h_current.device

        if L == 0:
            return h_current, r_current, None, None

        if timestamps is None:
            timestamps = torch.arange(L, dtype=torch.float, device=device)
        query_time = float(L) 

        intensity = self._hawkes_intensity(timestamps, query_time)   # [L]

        anchor_idx = self._select_anchors(intensity)                 # [K]
        K_actual = anchor_idx.shape[0]

        anchor_ent_embs = torch.stack(
            [entity_history_seq[i] for i in anchor_idx.tolist()], dim=0
        )   # [K, num_ents, h_dim]

        anchor_rel_embs = torch.stack(
            [relation_history_seq[i] for i in anchor_idx.tolist()], dim=0
        )   # [K, num_rels*2, h_dim]


        hawkes_weights = intensity[anchor_idx]                       # [K]

        hawkes_weights = hawkes_weights / (hawkes_weights.sum() + 1e-8)

        # query: W_Q · h_T → [num_ents, h_dim]
        Q = self.W_Q(h_current)                                      # [N, d]


        # anchor_ent_embs: [K, N, d]
        K_proj = self.W_K(anchor_ent_embs)                           # [K, N, d]
        V_proj = self.W_V(anchor_ent_embs)                           # [K, N, d]


        # Q: [N, d] -> [1, N, d], K_proj: [K, N, d]
        dot = (Q.unsqueeze(0) * K_proj).sum(dim=-1) / (self.h_dim ** 0.5)  # [K, N]


        attn_scores = dot * hawkes_weights.unsqueeze(1)              # [K, N]
        attn_weights = F.softmax(attn_scores, dim=0)                 # 在 K 维 softmax

        h_attn = (attn_weights.unsqueeze(-1) * V_proj).sum(dim=0)   # [N, d]
        h_out = self.phi(torch.cat([h_attn, h_current], dim=-1))     # [N, d]
        h_out = self.layer_norm_ent(h_out + h_current)             

        r_bar = (hawkes_weights[:, None, None] * anchor_rel_embs).sum(dim=0)  # [R, d]

        beta = torch.sigmoid(
            self.w_r(torch.cat([r_current, r_bar], dim=-1))
        )                                                             # [R, 1]

        r_out = beta * r_current + (1.0 - beta) * r_bar
        r_out = self.layer_norm_rel(r_out)

        return h_out, r_out, anchor_ent_embs, anchor_idx

    def anchor_loss(
        self,
        h_current: torch.Tensor,        # [num_ents, h_dim]
        anchor_ent_embs: torch.Tensor,  # [K, num_ents, h_dim] 
    ) -> torch.Tensor:
        if anchor_ent_embs is None:
            return torch.zeros(1, device=h_current.device)
        a_last = anchor_ent_embs[-1]                           # [N, d]
        loss_main = F.mse_loss(h_current, a_last)
        loss_anchor = sum(
            F.mse_loss(anchor_ent_embs[k], h_current)
            for k in range(anchor_ent_embs.shape[0])
        ) / anchor_ent_embs.shape[0]

        return loss_main + self.anchor_loss_lambda * loss_anchor
