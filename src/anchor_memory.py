import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalAnchorMemoryNetwork(nn.Module):

    def __init__(self, h_dim, num_rels, K=6,
                 beta_fast=0.5, beta_slow=0.1, dropout=0.1):
        super(TemporalAnchorMemoryNetwork, self).__init__()

        self.h_dim = h_dim
        self.num_rels = num_rels
        self.K = K
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self.logit_alpha = nn.Parameter(torch.tensor(0.0))   # αs 初始值 0.5

        self.W_Q = nn.Linear(h_dim, h_dim, bias=False)
        self.W_K = nn.Linear(h_dim, h_dim, bias=False)
        self.W_V = nn.Linear(h_dim, h_dim, bias=False)

        self.phi = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim * 2),
            nn.ReLU(),
            nn.Linear(h_dim * 2, h_dim))

        self.w_r = nn.Linear(h_dim * 2, 1)   # 合并Linear+bias

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(h_dim)

    def _select_anchors(self, seq_len):
        K = min(self.K, seq_len)
        if K == seq_len:
            return list(range(seq_len))
        indices = [int(round(i * (seq_len - 1) / (K - 1))) for i in range(K)]
        return sorted(set(indices))

    def _hawkes_weights(self, anchor_indices, current_t):

        device  = self.logit_alpha.device
        alpha_s = torch.sigmoid(self.logit_alpha)
        alpha_l = 1.0 - alpha_s

        weights = []
        for ak in anchor_indices:
            delta_t = float(current_t - ak)
            w = (alpha_s * torch.exp(torch.tensor(-self.beta_fast * delta_t, device=device))
               + alpha_l * torch.exp(torch.tensor(-self.beta_slow * delta_t, device=device)))
            weights.append(w)

        weights = torch.stack(weights)
        weights = weights / (weights.sum() + 1e-10)
        return weights

    def _anchor_attention_entity(self, h_current, anchor_embs, hawkes_weights):
        num_ents = h_current.size(0)
        K = anchor_embs.size(0)
        scale = math.sqrt(self.h_dim)

        Q = self.W_Q(h_current)                           # [num_ents, h_dim]
        anchor_embs_t = anchor_embs.permute(1, 0, 2)      # [num_ents, K, h_dim]
        K_proj = self.W_K(anchor_embs_t)                  # [num_ents, K, h_dim]
        V_proj = self.W_V(anchor_embs_t)                  # [num_ents, K, h_dim]

        attn_scores = torch.bmm(
            Q.unsqueeze(1),                               # [num_ents, 1, h_dim]
            K_proj.transpose(1, 2)                        # [num_ents, h_dim, K]
        ).squeeze(1) / scale                              # [num_ents, K]

        hawkes_weights = hawkes_weights.to(attn_scores.device)
        attn_scores = attn_scores * hawkes_weights.unsqueeze(0)  # [num_ents, K]
        attn_weights = F.softmax(attn_scores, dim=1)              # [num_ents, K]
        attn_weights = self.dropout(attn_weights)

        context = torch.bmm(
            attn_weights.unsqueeze(1),                    # [num_ents, 1, K]
            V_proj                                        # [num_ents, K, h_dim]
        ).squeeze(1)                                      # [num_ents, h_dim]

        h_out = self.phi(torch.cat([context, h_current], dim=-1))  # [num_ents, h_dim]
        h_out = self.layer_norm(h_out + h_current)        # 残差 + LayerNorm
        return h_out

    def _dynamic_relation_update(self, r_current, anchor_rel_embs, hawkes_weights):
        hawkes_weights = hawkes_weights.to(r_current.device)
        r_bar = (anchor_rel_embs * hawkes_weights.view(-1, 1, 1)).sum(dim=0)  # [num_rels*2, h_dim]
        # βj
        beta = torch.sigmoid(
            self.w_r(torch.cat([r_current, r_bar], dim=-1)))   # [num_rels*2, 1]
        # r̃^T_j
        r_updated = beta * r_current + (1.0 - beta) * r_bar   # [num_rels*2, h_dim]
        return r_updated

    def forward(self, history_embs, history_rel_embs):
        seq_len = len(history_embs)
        anchor_indices = self._select_anchors(seq_len)    # List[int], 长度K'≤K
        K_actual = len(anchor_indices)
        hawkes_weights = self._hawkes_weights(anchor_indices, seq_len)  # [K']
        anchor_ent_embs = torch.stack(
            [history_embs[ak] for ak in anchor_indices], dim=0)
        # anchor_rel_embs: [K', num_rels*2, h_dim]
        anchor_rel_embs = torch.stack(
            [history_rel_embs[ak] for ak in anchor_indices], dim=0)
        h_current = history_embs[-1]                      # [num_ents, h_dim]
        r_current = history_rel_embs[-1]                  # [num_rels*2, h_dim]

        h_anchor = self._anchor_attention_entity(
            h_current, anchor_ent_embs, hawkes_weights)   # [num_ents, h_dim]

        r_anchor = self._dynamic_relation_update(
            r_current, anchor_rel_embs, hawkes_weights)   # [num_rels*2, h_dim]

        return h_anchor, r_anchor