import torch
import torch.nn as nn
import torch.nn.functional as F

class ANGLLayer(nn.Module):
    def __init__(self, h_dim, k_neighbors=6, time_window=3, dropout=0.1):
        super(ANGLLayer, self).__init__()
        self.h_dim = h_dim
        self.k = k_neighbors
        self.time_window = time_window

        self.degree_gate = nn.Sequential(
            nn.Linear(1, h_dim // 2), nn.ReLU(),
            nn.Linear(h_dim // 2, 1), nn.Sigmoid())

        self.lambda_param = nn.Parameter(torch.tensor(0.0))

        self.W_agg = nn.Linear(h_dim, h_dim)
        self.dropout = nn.Dropout(dropout)

    def _build_candidate_freq_matrix(self, query_global_ids,
                                     history_graphs, device):
        Q = query_global_ids.size(0)
        freq_matrix = torch.zeros(Q, Q, device=device)

        if history_graphs is None or len(history_graphs) == 0:
            return freq_matrix

        max_id = query_global_ids.max().item() + 1
        id_table = torch.full((max_id,), -1, dtype=torch.long, device=device)
        id_table[query_global_ids] = torch.arange(Q, dtype=torch.long, device=device)

        recent_graphs = history_graphs[-self.time_window:]
        N_window = len(recent_graphs)

        for hist_g in recent_graphs:
            try:
                hist_g = hist_g.to(device)
                hist_nids = hist_g.ndata['id'].squeeze().long()  # [num_hist_nodes]
                src, dst = hist_g.edges()

                src_g = hist_nids[src.long()]                    # [E]
                dst_g = hist_nids[dst.long()]                    # [E]

                valid = (src_g < max_id) & (dst_g < max_id)
                src_g = src_g[valid]
                dst_g = dst_g[valid]

                li = id_table[src_g]                             # [E']
                lj = id_table[dst_g]                             # [E']

                in_query = (li >= 0) & (lj >= 0)
                li = li[in_query]
                lj = lj[in_query]

                if li.numel() == 0:
                    continue

                flat_ij = li * Q + lj
                flat_ji = lj * Q + li
                freq_flat = freq_matrix.view(-1)
                ones = torch.ones(flat_ij.size(0), device=device)
                freq_flat.scatter_add_(0, flat_ij, ones)
                freq_flat.scatter_add_(0, flat_ji, ones)

            except Exception:
                continue

        # [0,1]
        freq_matrix = freq_matrix / float(N_window + 1)
        freq_matrix = freq_matrix.clamp(max=1.0)
        return freq_matrix

    def compute_latent_adjacency(self, ent_emb, g, query_global_ids,
                                 history_graphs=None, device=None):
        Q = ent_emb.size(0)
        device = device or ent_emb.device

        if Q <= 1:
            return torch.eye(Q, device=device)
        norm_q = F.normalize(ent_emb, p=2, dim=1)            # [Q', h_dim]
        sem_sim = torch.mm(norm_q, norm_q.t())                 # [Q', Q']
        freq_matrix = self._build_candidate_freq_matrix(
            query_global_ids, history_graphs, device)            # [Q', Q']
        lam = torch.sigmoid(self.lambda_param)
        score_mat = (1.0 - lam) * sem_sim + lam * freq_matrix   # [Q', Q']
        score_mat = score_mat.fill_diagonal_(-1e9)
        src_full, dst_full = g.edges()
        if src_full.numel() > 0:
            snap_nids = g.ndata['id'].squeeze().long().to(device)
            max_snap = snap_nids.max().item() + 1
            max_q_id = query_global_ids.max().item() + 1
            q_table = torch.full(
                (max(max_snap, max_q_id),), -1,
                dtype=torch.long, device=device)
            q_table[query_global_ids] = torch.arange(Q, dtype=torch.long, device=device)
            src_gid = snap_nids[src_full.long()]
            dst_gid = snap_nids[dst_full.long()]

            valid = (src_gid < q_table.size(0)) & (dst_gid < q_table.size(0))
            src_gid = src_gid[valid]
            dst_gid = dst_gid[valid]

            qi = q_table[src_gid]
            qj = q_table[dst_gid]
            both_in = (qi >= 0) & (qj >= 0)
            qi = qi[both_in]
            qj = qj[both_in]

            if qi.numel() > 0:
                adj_mask = torch.zeros(Q, Q, device=device)
                adj_mask[qi, qj] = 1
                adj_mask[qj, qi] = 1
                score_mat = score_mat.masked_fill(adj_mask > 0, -1e9)
        k = min(self.k, max(1, Q - 1))
        topk_values, topk_indices = torch.topk(score_mat, k, dim=1)
        attn_weights = F.softmax(topk_values, dim=1)             # [Q', k]

        latent_adj = torch.zeros(Q, Q, device=device)
        latent_adj.scatter_(1, topk_indices, attn_weights)
        latent_adj = 0.5 * (latent_adj + latent_adj.t())
        latent_adj = latent_adj + torch.eye(Q, device=device) * 0.1
        row_sum = latent_adj.sum(dim=1, keepdim=True).clamp(min=1e-10)
        return latent_adj / row_sum

    def forward(self, g, entity_embs, relation_embs,
                history_graphs=None, history_node_embs=None,
                query_ids=None):
        device = entity_embs.device
        node_ids = g.ndata['id'].squeeze()
        node_ids = (node_ids.long().to(device)
                    if isinstance(node_ids, torch.Tensor)
                    else torch.tensor(node_ids, dtype=torch.long, device=device))
        N_t = node_ids.size(0)
        if N_t <= 1:
            return entity_embs

        curr_nodes_emb = entity_embs[node_ids]   # [N_t, h_dim]
        if query_ids is not None and query_ids.numel() > 0:
            query_ids = query_ids.to(device)

            max_id = node_ids.max().item() + 1
            id_table = torch.full((max_id,), -1, dtype=torch.long, device=device)
            id_table[node_ids] = torch.arange(N_t, dtype=torch.long, device=device)

            valid_mask = query_ids < max_id
            valid_q = query_ids[valid_mask]
            local_q = id_table[valid_q]
            in_cur_mask = local_q >= 0
            local_q = local_q[in_cur_mask]
            global_q = valid_q[in_cur_mask]

            if local_q.numel() < 2:
                local_q = torch.arange(N_t, device=device)
                global_q = node_ids
        else:
            local_q = torch.arange(N_t, device=device)
            global_q = node_ids
        #  [Q', h_dim]
        target_emb = curr_nodes_emb[local_q]

        in_deg = g.in_degrees().float().to(device)
        out_deg = g.out_degrees().float().to(device)
        degree_all = (in_deg + out_deg)
        degree_mean = degree_all.mean().clamp(min=1.0)
        degree_q = degree_all[local_q].unsqueeze(1)          # [Q', 1]
        degree_norm = degree_q / degree_mean                    # d_s / d̄
        phi = self.degree_gate(degree_norm)              # [Q', 1], φ∈(0,1)

        latent_adj = self.compute_latent_adjacency(
            target_emb, g, global_q,
            history_graphs=history_graphs,
            device=device)                                       # [Q', Q']
        latent_embs = torch.mm(latent_adj, target_emb)          # [Q', h_dim]
        latent_embs = self.W_agg(latent_embs)
        latent_embs = self.dropout(latent_embs)
        # e_{s,t} = h^obs_s + φ(s) · h^lat_s
        enhanced_target = target_emb + phi * latent_embs        # [Q', h_dim]

        enhanced_full_emb = entity_embs.clone()
        enhanced_full_emb[global_q] = enhanced_target
        return enhanced_full_emb