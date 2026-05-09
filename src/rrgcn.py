import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from model import BaseRGCN
from decoder import *
from rise import ANGLLayer
from anchor_memory import TemporalAnchorMemoryNetwork
try:
    from inductive_enhanced_diffusion import InductiveEnhancedDiffusion
    INDUCTIVE_DIFFUSION_AVAILABLE = True
    print("✅ DiffuTKG归纳式扩散模块已加载")
except ImportError as e:
    INDUCTIVE_DIFFUSION_AVAILABLE = False

class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "convgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop,
                             skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "convgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')

class TemporalTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [N, seq_len, h_dim]
        x = self.layer_norm(x)
        seq_len = x.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        x = x.transpose(0, 1)                   # [seq_len, N, h_dim]
        x = self.transformer(x, mask=mask)       # [seq_len, N, h_dim]
        x = x.transpose(0, 1)                   # [N, seq_len, h_dim]
        x = self.output_proj(x)
        return x[:, -1, :]                       # [N, h_dim]

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask

class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels,
                 num_static_rels, num_words, num_times, time_interval,
                 h_dim, opn, history_rate, sequence_len,
                 num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False,
                 skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat',
                 weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False,
                 use_cuda=False, gpu=0, analysis=False,
                 transformer_nhead=4,
                 transformer_num_layers=2,
                 use_anel=False,
                 anel_k_neighbors=6,
                 anel_time_window=3,
                 anchor_K=6,
                 anchor_beta_fast=0.5,
                 anchor_beta_slow=0.1,
                 use_inductive_diffusion=False,
                 use_time_decay=False,
                 use_multikernel_hawkes=False,
                 gru_beta=0.3,
                 gru_beta_fast=0.5,
                 gru_beta_slow=0.1,
                 diffusion_timesteps=100,
                 diffusion_sampling_steps=10,
                 diffusion_noise_level_seen=0.3,
                 diffusion_noise_level_unseen=0.5,
                 ood_loss_weight=0.2,
                 diffusion_loss_weight=0.1,
                 distribution_loss_weight=0.05,
                 use_meta_learning=True,
                 meta_lr=0.01,
                 use_distribution_alignment=True,
                 alignment_strength=0.1):
        super(RecurrentRGCN, self).__init__()
        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.history_rate = history_rate
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.num_times = num_times
        self.time_interval = time_interval
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        self.sin = torch.sin
        self.linear_0 = nn.Linear(num_times, 1)
        self.linear_1 = nn.Linear(num_times, self.h_dim - 1)
        self.tanh = nn.Tanh()
        self.use_cuda = None
        self.use_anel = use_anel
        self.anel_time_window = anel_time_window
        self.use_time_decay = use_time_decay
        self.use_multikernel_hawkes = use_multikernel_hawkes
        self.gru_beta = gru_beta
        self.gru_beta_fast = gru_beta_fast
        self.gru_beta_slow = gru_beta_slow
        if self.use_multikernel_hawkes:
            self.hawkes_alpha = nn.Parameter(torch.tensor(0.7))

        self.w1 = nn.Parameter(torch.Tensor(h_dim, h_dim)); nn.init.xavier_normal_(self.w1)
        self.w2 = nn.Parameter(torch.Tensor(h_dim, h_dim)); nn.init.xavier_normal_(self.w2)
        self.emb_rel = nn.Parameter(torch.Tensor(num_rels * 2, h_dim)); nn.init.xavier_normal_(self.emb_rel)
        self.dynamic_emb = nn.Parameter(torch.Tensor(num_ents, h_dim));   nn.init.normal_(self.dynamic_emb)
        self.weight_t1 = nn.Parameter(torch.randn(1, h_dim))
        self.bias_t1 = nn.Parameter(torch.randn(1, h_dim))
        self.weight_t2 = nn.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.Parameter(torch.randn(1, h_dim))

        if self.use_static:
            self.words_emb = nn.Parameter(torch.Tensor(self.num_words, h_dim))
            nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(
                h_dim, h_dim, num_static_rels * 2, num_bases,
                activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = nn.MSELoss()

        self.loss_r = nn.CrossEntropyLoss()
        self.loss_e = nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents, h_dim, h_dim, num_rels * 2,
                             num_bases, num_basis, num_hidden_layers,
                             dropout, self_loop, skip_connect,
                             encoder_name, self.opn, self.emb_rel,
                             use_cuda, analysis)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.zeros(h_dim))

        self.global_weight = nn.Parameter(torch.Tensor(num_ents, 1))
        nn.init.xavier_uniform_(self.global_weight, gain=nn.init.calculate_gain('relu'))
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.relation_cell_1 = nn.GRUCell(h_dim * 2, h_dim)
        self.entity_cell_1   = nn.GRUCell(h_dim, h_dim)

        self.anel = ANGLLayer(
            h_dim=h_dim, k_neighbors=anel_k_neighbors,
            time_window=anel_time_window, dropout=dropout)

        self.entity_transformer = TemporalTransformer(
            h_dim, transformer_nhead, transformer_num_layers, dropout)
        self.relation_transformer = TemporalTransformer(
            h_dim, transformer_nhead, transformer_num_layers, dropout)
        self.entity_fusion_gate = nn.Sequential(nn.Linear(h_dim * 2, h_dim), nn.Sigmoid())
        self.relation_fusion_gate = nn.Sequential(nn.Linear(h_dim * 2, h_dim), nn.Sigmoid())

        self.anchor_memory = TemporalAnchorMemoryNetwork(
            h_dim=h_dim,
            num_rels=num_rels,
            K=anchor_K,
            beta_fast=anchor_beta_fast,
            beta_slow=anchor_beta_slow,
            dropout=dropout)

        self.anchor_ent_gate = nn.Sequential(nn.Linear(h_dim * 2, h_dim), nn.Sigmoid())
        self.anchor_rel_gate = nn.Sequential(nn.Linear(h_dim * 2, h_dim), nn.Sigmoid())

        if decoder_name == "timeconvtranse":
            self.decoder_ob1  = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob2  = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re1 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re2 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError

        self.use_inductive_diffusion = use_inductive_diffusion and INDUCTIVE_DIFFUSION_AVAILABLE
        self.diffusion_noise_level_seen = diffusion_noise_level_seen
        self.diffusion_noise_level_unseen = diffusion_noise_level_unseen
        self.ood_loss_weight = ood_loss_weight
        self.diffusion_loss_weight = diffusion_loss_weight
        self.distribution_loss_weight = distribution_loss_weight
        self._diffusion_losses = {}
        self._seen_entity_ids = None

        if self.use_inductive_diffusion:
            self.inductive_diffusion = InductiveEnhancedDiffusion(
                h_dim=h_dim, num_rels=num_rels,
                diffusion_timesteps=diffusion_timesteps,
                diffusion_sampling_steps=diffusion_sampling_steps,
                use_meta_learning=use_meta_learning, meta_lr=meta_lr,
                use_distribution_alignment=use_distribution_alignment,
                alignment_strength=alignment_strength)

            self.denoising_net = nn.Sequential(
                nn.Linear(h_dim * 3, h_dim * 2),
                nn.SiLU(),
                nn.Linear(h_dim * 2, h_dim * 2),
                nn.SiLU(),
                nn.Linear(h_dim * 2, h_dim))

            self.diffusion_time_emb = nn.Embedding(diffusion_timesteps + 1, h_dim)

            alpha_start = 0.9999
            alpha_end = 0.0001
            betas = torch.linspace(alpha_end, alpha_start, diffusion_timesteps)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)            # [M]
            self.register_buffer('alphas_cumprod', alphas_cumprod)   # 不参与梯度更新
            self.diffusion_timesteps = diffusion_timesteps

        self._cached_rel_cond = None
        self._cached_entity_rel_ids = None
        self._cached_neighbor_embs = None
        self._current_query_ids = None
    def forward(self, g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []

        entity_history_seq = []
        relation_history_seq = []

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []

        if self.use_time_decay_gru:
            device = self.h.device
            last_time = torch.full((self.num_ents,), -1, dtype=torch.long, device=device)
        else:
            last_time = None

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = (torch.zeros(self.num_rels*2, self.h_dim).float().cuda()
                       if use_cuda
                       else torch.zeros(self.num_rels*2, self.h_dim).float())
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x_input[r_idx] = torch.mean(temp_e[span[0]:span[1], :], dim=0, keepdim=True)

            x_input  = torch.cat((self.emb_rel, x_input), dim=1)
            self.h_0 = self.relation_cell_1(x_input,
                           self.emb_rel if i == 0 else self.h_0)
            self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0

            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h

            if self.use_anel:
                node_ids_cur = g.ndata['id'].squeeze().long().to(current_h.device)

                hist_graphs_win = g_list[max(0, i - self.anel_time_window): i]

                current_h_full = self.h.clone()
                current_h_full[node_ids_cur] = current_h
                current_h_full = self.anel(
                    g, current_h_full, self.h_0,
                    history_graphs = hist_graphs_win,
                    history_node_embs = None,
                    query_ids = self._current_query_ids)
                current_h = (F.normalize(current_h_full, dim=-1)
                             if self.layer_norm
                             else current_h_full)[node_ids_cur]

            delta_t = torch.clamp(float(i) - last_time.float(), min=0.0)

            if self.use_multikernel_hawkes:
                alpha = torch.sigmoid(self.hawkes_alpha)
                decay = (alpha * torch.exp(-self.gru_beta_fast * delta_t)
                            + (1-alpha) * torch.exp(-self.gru_beta_slow * delta_t)).unsqueeze(1)
            else:
                decay = torch.exp(-self.gru_beta * delta_t).unsqueeze(1)
            h_for = self.h * decay

            self.h = self.entity_cell_1(current_h, h_for)
            self.h = F.normalize(self.h) if self.layer_norm else self.h
            history_embs.append(self.h)

            last_time[g.ndata['id'].squeeze().long().to(self.h.device)] = i

            entity_history_seq.append(self.h.clone())
            relation_history_seq.append(self.h_0.clone())

        if self.use_anchor_memory and len(entity_history_seq) > 0:
            h_anchor, r_anchor = self.anchor_memory(
                entity_history_seq, relation_history_seq)

            gate_ent = self.anchor_ent_gate(
                torch.cat([self.h, h_anchor], dim=-1))       # [num_ents, h_dim]
            self.h = gate_ent * self.h + (1 - gate_ent) * h_anchor
            self.h = F.normalize(self.h) if self.layer_norm else self.h

            gate_rel = self.anchor_rel_gate(
                torch.cat([self.h_0, r_anchor], dim=-1))     # [num_rels*2, h_dim]
            self.h_0 = gate_rel * self.h_0 + (1 - gate_rel) * r_anchor
            self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0

            if history_embs:
                history_embs[-1] = self.h.clone()

            entity_history_seq[-1]   = self.h.clone()
            relation_history_seq[-1] = self.h_0.clone()

        if self.use_transformer and len(entity_history_seq) > 0:
            ent_seq = torch.stack(entity_history_seq, dim=1)   # [num_ents, seq_len, h_dim]
            transformer_ent_out = self.entity_transformer(ent_seq)

            gate_ent = self.entity_fusion_gate(
                torch.cat([self.h, transformer_ent_out], dim=-1))
            self.h = gate_ent * self.h + (1 - gate_ent) * transformer_ent_out
            self.h = F.normalize(self.h) if self.layer_norm else self.h
            if history_embs:
                history_embs[-1] = self.h.clone()

            rel_seq = torch.stack(relation_history_seq, dim=1) # [num_rels*2, seq_len, h_dim]
            transformer_rel_out = self.relation_transformer(rel_seq)
            gate_rel   = self.relation_fusion_gate(
                torch.cat([self.h_0, transformer_rel_out], dim=-1))
            self.h_0   = gate_rel * self.h_0 + (1 - gate_rel) * transformer_rel_out
            self.h_0   = F.normalize(self.h_0) if self.layer_norm else self.h_0

        if self.use_inductive_diffusion and hasattr(self, 'inductive_diffusion'):
            rel_cond       = self._build_entity_rel_cond_from_anchor()
            entity_rel_ids = self._build_entity_rel_ids(g_list)
            neighbor_embs  = self._build_neighbor_embs(g_list)

            if self.training:
                if self._seen_entity_ids is not None:
                    seen_ids = torch.tensor(list(self._seen_entity_ids),
                                            dtype=torch.long, device=self.h.device)
                    seen_ids = seen_ids[seen_ids < self.h.shape[0]]
                    self.inductive_diffusion.update_seen_statistics(self.h[seen_ids].detach())
                else:
                    self.inductive_diffusion.update_seen_statistics(self.h.detach())

                is_seen_labels = self._make_pseudo_unseen_labels(self.h.shape[0])
                _, losses = self.inductive_diffusion(
                    self.h, rel_cond, mode='train',
                    is_seen_labels=is_seen_labels,
                    neighbor_embs=neighbor_embs,
                    entity_rel_ids=entity_rel_ids)
                self._diffusion_losses = losses
            else:
                h_refined = self.inductive_diffusion(
                    self.h, rel_cond, mode='test',
                    neighbor_embs=neighbor_embs,
                    seen_entity_ids=self._seen_entity_ids,
                    current_entity_ids=list(range(self.num_ents)),
                    entity_rel_ids=entity_rel_ids)
                self.h = h_refined
                if history_embs:
                    history_embs[-1] = self.h.clone()

        return history_embs, static_emb, self.h_0, gate_list, degree_list

    def _make_pseudo_unseen_labels(self, num_entities, unseen_ratio=0.3):
        labels   = torch.ones(num_entities, dtype=torch.long, device=self.h.device)
        n_unseen = max(1, int(num_entities * unseen_ratio))
        labels[torch.randperm(num_entities, device=self.h.device)[:n_unseen]] = 0
        return labels

    def _build_entity_rel_cond_from_anchor(self):
        device = self.h.device
        num_ents = self.h.shape[0]
        rel_cond = self.h_0.mean(dim=0).unsqueeze(0).expand(num_ents, -1).clone()
        return rel_cond

    def _build_entity_rel_ids(self, g_list):
        entity_rel_ids = torch.zeros(self.num_ents, dtype=torch.long, device=self.h.device)
        if not g_list:
            return entity_rel_ids
        g = g_list[-1]
        try:
            node_ids = g.ndata['id'].squeeze()
            src, dst = g.edges()
            rel_data = g.edata.get('type', None)
            if rel_data is not None:
                from collections import Counter
                ent_rel_counter = {}
                for s, d, r in zip(src.tolist(), dst.tolist(), rel_data.tolist()):
                    sg = node_ids[s].item()
                    dg = node_ids[d].item()
                    rc = min(r, self.num_rels - 1)
                    ent_rel_counter.setdefault(sg, Counter())[rc] += 1
                    ent_rel_counter.setdefault(dg, Counter())[rc] += 1
                for eid, counter in ent_rel_counter.items():
                    if eid < self.num_ents:
                        entity_rel_ids[eid] = counter.most_common(1)[0][0]
        except Exception:
            pass
        return entity_rel_ids

    def _build_neighbor_embs(self, g_list, k=3):
        device        = self.h.device
        num_ents      = self.h.shape[0]
        neighbor_embs = torch.zeros(num_ents, k, self.h_dim, device=device)
        if not g_list:
            return neighbor_embs
        g = g_list[-1]
        try:
            node_ids = g.ndata['id'].squeeze()
            src, dst = g.edges()
            from collections import defaultdict
            neighbors = defaultdict(set)
            for s, d in zip(src.tolist(), dst.tolist()):
                sg = node_ids[s].item()
                dg = node_ids[d].item()
                if sg < num_ents and dg < num_ents:
                    neighbors[sg].add(dg)
                    neighbors[dg].add(sg)
            for eid, nbr_set in neighbors.items():
                if eid >= num_ents:
                    continue
                nbr_list   = list(nbr_set)[:k]
                nbr_tensor = torch.tensor(nbr_list, dtype=torch.long, device=device)
                nbr_embs   = self.h[nbr_tensor]
                pad_len    = k - nbr_embs.shape[0]
                if pad_len > 0:
                    nbr_embs = torch.cat([nbr_embs,
                        torch.zeros(pad_len, self.h_dim, device=device)], dim=0)
                neighbor_embs[eid] = nbr_embs
        except Exception as e:
            print(f"[_build_neighbor_embs] Warning: {e}")
        return neighbor_embs

    def set_seen_entity_ids(self, seen_entity_ids: set):
        self._seen_entity_ids = seen_entity_ids

    def predict(self, test_graph, num_rels, static_graph, test_triplets,
                entity_history_vocabulary, rel_history_vocabulary, use_cuda):
        self.use_cuda = use_cuda
        with torch.no_grad():
            inv_triples = test_triplets[:, [2, 1, 0, 3]].clone()
            inv_triples[:, 1] += num_rels
            all_triples = torch.cat((test_triplets, inv_triples))

            self._current_query_ids = torch.cat([
                all_triples[:, 0],
                all_triples[:, 2]
            ]).unique()

            evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)

            self._current_query_ids = None

            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            time_embs = self.get_init_time(all_triples)

            score_rel = torch.log(
                self.history_rate * self.rel_history_mode(embedding, r_emb, time_embs, all_triples, rel_history_vocabulary)
                + (1 - self.history_rate) * self.rel_raw_mode(embedding, r_emb, time_embs, all_triples))
            score = torch.log(
                self.history_rate * self.history_mode(embedding, r_emb, time_embs, all_triples, entity_history_vocabulary)
                + (1 - self.history_rate) * self.raw_mode(embedding, r_emb, time_embs, all_triples))

            return all_triples, score, score_rel

    def get_loss(self, glist, triples, static_graph,
                 entity_history_vocabulary, rel_history_vocabulary, use_cuda):
        self.use_cuda = use_cuda

        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_diff = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inv_triples = triples[:, [2, 1, 0, 3]].clone()
        inv_triples[:, 1] += self.num_rels
        all_triples = torch.cat([triples, inv_triples]).to(self.gpu)

        self._current_query_ids = torch.cat([
            all_triples[:, 0],
            all_triples[:, 2]
        ]).unique()

        evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, use_cuda)
        self._current_query_ids = None
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
        time_embs = self.get_init_time(all_triples)

        score_en = torch.log(
                self.history_rate * self.history_mode(pre_emb, r_emb, time_embs, all_triples, entity_history_vocabulary)
                + (1 - self.history_rate) * self.raw_mode(pre_emb, r_emb, time_embs, all_triples))
        loss_ent += F.nll_loss(score_en, all_triples[:, 2])

        score_re = torch.log(
                self.history_rate * self.rel_history_mode(pre_emb, r_emb, time_embs, all_triples, rel_history_vocabulary)
                + (1 - self.history_rate) * self.rel_raw_mode(pre_emb, r_emb, time_embs, all_triples))
        loss_rel += F.nll_loss(score_re, all_triples[:, 1])

        if self.use_static:
            for time_step, evolve_emb in enumerate(evolve_embs):
                step = (self.angle * math.pi / 180) * (time_step + 1 if self.discount == 1 else 1)
                if self.layer_norm:
                    sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                else:
                    sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                    c = (torch.norm(static_emb, p=2, dim=1)
                                  * torch.norm(evolve_emb, p=2, dim=1))
                    sim_matrix = sim_matrix / c
                mask = (math.cos(step) - sim_matrix) > 0
                loss_static += self.weight * torch.sum(
                    torch.masked_select(math.cos(step) - sim_matrix, mask))

        if self.use_inductive_diffusion and self._diffusion_losses:
            dl = self._diffusion_losses
            if 'ood_cls' in dl: loss_diff += self.ood_loss_weight       * dl['ood_cls']
            if 'ood_contrast' in dl: loss_diff += self.ood_loss_weight * 0.1 * dl['ood_contrast']
            if 'inductive_enhanced_diffusion.py' in dl: loss_diff += self.diffusion_loss_weight * dl['inductive_enhanced_diffusion.py']
            if 'distribution' in dl: loss_diff += self.distribution_loss_weight * dl['distribution']
            if 'meta' in dl: loss_diff += 0.1 * dl['meta']
            self._diffusion_losses = {}

        return loss_ent, loss_rel, loss_static, loss_diff

    def get_init_time(self, quadrupleList):
        T_idx = (quadrupleList[:, 3] // self.time_interval).unsqueeze(1).float()
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = self.sin(self.weight_t2 * T_idx + self.bias_t2)
        return t1, t2

    def raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        return F.softmax(
            self.decoder_ob1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, self.num_ents), dim=1)

    def history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        global_index = (torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float)).to('cuda')
                        if self.use_cuda
                        else torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float)))
        return F.softmax(
            self.decoder_ob2.forward(pre_emb, r_emb, time_embs, all_triples,
                                     partial_embeding=global_index), dim=1)

    def rel_raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        return F.softmax(
            self.rdecoder_re1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, 2*self.num_rels), dim=1)

    def rel_history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        global_index = (torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float)).to('cuda')
                        if self.use_cuda
                        else torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float)))
        return F.softmax(
            self.rdecoder_re2.forward(pre_emb, r_emb, time_embs, all_triples,
                                      partial_embeding=global_index), dim=1)