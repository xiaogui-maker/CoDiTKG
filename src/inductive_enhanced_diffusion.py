import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EnhancedOODDetector(nn.Module):

    def __init__(self, h_dim, num_rels):
        super().__init__()

        self.h_dim = h_dim
        self.num_rels = num_rels

        self.detector = nn.Sequential(
            nn.Linear(h_dim, h_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h_dim * 2, 2)
        )
        self.relation_prototypes = nn.Parameter(
            torch.randn(num_rels * 2, h_dim)
        )
        nn.init.xavier_normal_(self.relation_prototypes)

        self.meta_generator = nn.Sequential(
            nn.Linear(h_dim * 3, h_dim * 2),  # [rel_proto, neighbor_mean, neighbor_max]
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h_dim * 2, h_dim)
        )

    def detect(self, entity_features):

        ood_scores = self.detector(entity_features)
        ood_prob = F.softmax(ood_scores, dim=-1)

        is_unseen = (ood_prob[:, 1] > 0.5)
        confidence = ood_prob[:, 1]  # unseen的置信度

        return is_unseen, confidence

    def generate_unseen_embedding(self, relation_ids, neighbor_embs=None):
        batch_size = relation_ids.shape[0]
        device = relation_ids.device

        rel_proto = self.relation_prototypes[relation_ids]  # [N, h_dim]

        if neighbor_embs is not None and neighbor_embs.numel() > 0:
            # neighbor_embs: [N, K, h_dim]
            neighbor_mean = neighbor_embs.mean(dim=1)  # [N, h_dim]
            neighbor_max = neighbor_embs.max(dim=1)[0]  # [N, h_dim]
        else:
            neighbor_mean = torch.zeros(batch_size, self.h_dim, device=device)
            neighbor_max = torch.zeros(batch_size, self.h_dim, device=device)
        meta_input = torch.cat([rel_proto, neighbor_mean, neighbor_max], dim=-1)
        unseen_emb_init = self.meta_generator(meta_input)

        return unseen_emb_init

    def compute_loss(self, embeddings, is_seen_labels, seen_embs=None):

        ood_scores = self.detector(embeddings)
        loss_cls = F.cross_entropy(ood_scores, is_seen_labels.long())

        losses = {'ood_cls': loss_cls}

        if seen_embs is not None and seen_embs.numel() > 0:
            seen_center = seen_embs.mean(dim=0)

            unseen_mask = (is_seen_labels == 0)
            if unseen_mask.any():
                unseen_embs = embeddings[unseen_mask]
                dist_to_center = F.cosine_similarity(
                    unseen_embs,
                    seen_center.unsqueeze(0),
                    dim=-1
                )
                loss_contrast = -torch.log(1 - dist_to_center.abs() + 1e-8).mean()
                losses['ood_contrast'] = loss_contrast

        return losses

class MetaLearner(nn.Module):
    def __init__(self, h_dim, num_rels, meta_lr=0.01):
        super().__init__()

        self.h_dim = h_dim
        self.num_rels = num_rels
        self.meta_lr = meta_lr

        self.fast_adapter = nn.Sequential(
            nn.Linear(h_dim * 3, h_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h_dim * 2, h_dim)
        )

        self.rel_adapters = nn.ModuleList([
            nn.Linear(h_dim, h_dim) for _ in range(num_rels * 2)
        ])

    def inner_loop_adapt(self, support_examples, n_steps=5):

        params = [p.clone() for p in self.fast_adapter.parameters()]

        for step in range(n_steps):
            support_loss = 0

            for (relation, neighbors, target) in support_examples:

                pred_emb = self._generate_with_params(relation, neighbors, params)

                support_loss += F.mse_loss(pred_emb, target)

            grads = torch.autograd.grad(
                support_loss,
                params,
                create_graph=True
            )

            params = [p - self.meta_lr * g for p, g in zip(params, grads)]

        return params

    def _generate_with_params(self, relation, neighbor_embs, params=None):

        if params is None:
            params = list(self.fast_adapter.parameters())

        if neighbor_embs is not None and len(neighbor_embs) > 0:
            neighbor_mean = neighbor_embs.mean(dim=0)
            neighbor_max = neighbor_embs.max(dim=0)[0]
        else:
            neighbor_mean = torch.zeros(self.h_dim, device=relation.device)
            neighbor_max = torch.zeros(self.h_dim, device=relation.device)

        x = torch.cat([relation, neighbor_mean, neighbor_max], dim=-1)

        for i, (weight, bias) in enumerate(zip(params[::2], params[1::2])):
            x = F.linear(x, weight, bias)
            if i < len(params) // 2 - 1:
                x = F.gelu(x)

        return x

    def generate_embedding(self, relation_embs, neighbor_embs, few_shot_examples=None):
        batch_size = relation_embs.shape[0]
        if few_shot_examples is not None:
            adapted_params = self.inner_loop_adapt(few_shot_examples)
        else:
            adapted_params = None
        generated_embs = []
        for i in range(batch_size):
            rel_emb = relation_embs[i]
            neigh_embs = neighbor_embs[i] if neighbor_embs is not None else None

            emb = self._generate_with_params(rel_emb, neigh_embs, adapted_params)
            generated_embs.append(emb)

        return torch.stack(generated_embs)


class DistributionAlignment(nn.Module):

    def __init__(self, h_dim):
        super().__init__()

        self.h_dim = h_dim
        self.register_buffer('seen_mean', torch.zeros(h_dim))
        self.register_buffer('seen_std', torch.ones(h_dim))
        self.register_buffer('seen_count', torch.tensor(0.))

        print("✅ DistributionAlignment初始化完成")

    def update_seen_statistics(self, seen_embs):
        if seen_embs.numel() == 0:
            return

        batch_mean = seen_embs.mean(dim=0)
        batch_std = seen_embs.std(dim=0)
        batch_size = seen_embs.shape[0]
        total = self.seen_count + batch_size

        self.seen_mean = (
                                 self.seen_count * self.seen_mean + batch_size * batch_mean
                         ) / total

        self.seen_std = (
                                self.seen_count * self.seen_std + batch_size * batch_std
                        ) / total

        self.seen_count = total

    def align(self, unseen_embs, strength=0.1):
        unseen_mean = unseen_embs.mean(dim=-1, keepdim=True)
        unseen_std = unseen_embs.std(dim=-1, keepdim=True) + 1e-8

        unseen_whitened = (unseen_embs - unseen_mean) / unseen_std
        unseen_colored = unseen_whitened * self.seen_std + self.seen_mean
        aligned = (1 - strength) * unseen_embs + strength * unseen_colored

        return aligned

    def compute_distribution_loss(self, unseen_embs):
        unseen_mean = unseen_embs.mean(dim=0)
        unseen_std = unseen_embs.std(dim=0) + 1e-8
        kl_div = (
                torch.log(self.seen_std / unseen_std) +
                (unseen_std ** 2 + (unseen_mean - self.seen_mean) ** 2) / (2 * self.seen_std ** 2) -
                0.5
        ).sum()

        return kl_div

class InductiveDiffusionRefiner(nn.Module):

    def __init__(
            self,
            h_dim: int = 200,
            num_timesteps: int = 100,
            noise_schedule: str = 'linear',
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            sampling_steps: int = 10,
    ):
        super().__init__()

        self.h_dim = h_dim
        self.num_timesteps = num_timesteps
        self.sampling_steps = sampling_steps
        if noise_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif noise_schedule == 'cosine':
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.denoiser = DenoisingNetwork(h_dim)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, x_t, t, condition=None):
        predicted_noise = self.denoiser(x_t, t, condition)
        alpha_cumprod_t = self.alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        while len(alpha_cumprod_t.shape) < len(x_t.shape):
            alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)
        pred_x0 = (x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -5, 5)
        if t[0] == 0:
            return pred_x0
        alpha_cumprod_prev = self.alphas_cumprod[t - 1]
        while len(alpha_cumprod_prev.shape) < len(x_t.shape):
            alpha_cumprod_prev = alpha_cumprod_prev.unsqueeze(-1)

        pred_x_prev = torch.sqrt(alpha_cumprod_prev) * pred_x0 + \
                      torch.sqrt(1 - alpha_cumprod_prev) * predicted_noise

        return pred_x_prev

    def refine_with_alignment(
            self,
            x_init,
            noise_level=0.5,
            condition=None,
            seen_mean=None,
            seen_std=None,
            alignment_strength=0.1
    ):

        batch_size = x_init.shape[0]
        device = x_init.device
        t_start = int(self.num_timesteps * noise_level)
        t = torch.full((batch_size,), t_start, dtype=torch.long, device=device)

        noise = torch.randn_like(x_init)
        x_t = self.q_sample(x_init, t, noise)
        step_size = max(1, t_start // self.sampling_steps)

        for step_idx in range(t_start, 0, -step_size):
            t_curr = torch.full((batch_size,), step_idx, dtype=torch.long, device=device)
            x_t = self.p_sample(x_t, t_curr, condition)
            if step_idx % 10 == 0 and seen_mean is not None:
                # Whitening
                x_mean = x_t.mean(dim=-1, keepdim=True)
                x_std = x_t.std(dim=-1, keepdim=True) + 1e-8
                x_whitened = (x_t - x_mean) / x_std
                x_colored = x_whitened * seen_std + seen_mean
                x_t = (1 - alignment_strength) * x_t + alignment_strength * x_colored

        return x_t

    def forward(self, x, mode='train', **kwargs):
        if mode == 'train':
            batch_size = x.shape[0]
            device = x.device

            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
            noise = torch.randn_like(x)

            x_noisy = self.q_sample(x, t, noise)
            predicted_noise = self.denoiser(x_noisy, t, kwargs.get('condition'))

            loss = F.mse_loss(predicted_noise, noise)

            return x, loss

        else:
            is_unseen = kwargs.get('is_unseen', False)

            if is_unseen:
                noise_level = kwargs.get('noise_level', 0.5)

                refined = self.refine_with_alignment(
                    x,
                    noise_level=noise_level,
                    condition=kwargs.get('condition'),
                    seen_mean=kwargs.get('seen_mean'),
                    seen_std=kwargs.get('seen_std'),
                    alignment_strength=kwargs.get('alignment_strength', 0.1)
                )
            else:
                noise_level = kwargs.get('noise_level', 0.3)

                refined = self.refine_with_alignment(
                    x,
                    noise_level=noise_level,
                    condition=kwargs.get('condition'),
                    seen_mean=None,
                    seen_std=None
                )

            return refined


class DenoisingNetwork(nn.Module):
    def __init__(self, h_dim):
        super().__init__()

        self.h_dim = h_dim

        self.time_embed = nn.Sequential(
            nn.Linear(h_dim, h_dim * 2),
            nn.GELU(),
            nn.Linear(h_dim * 2, h_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(h_dim, h_dim * 2),
            nn.LayerNorm(h_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(h_dim * 2, h_dim * 2),
            nn.LayerNorm(h_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(h_dim * 2, h_dim)
        )

    def get_time_embedding(self, t):
        half_dim = self.h_dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        if self.h_dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))

        return self.time_embed(embeddings)

    def forward(self, x, t, condition=None):
        t_emb = self.get_time_embedding(t)
        x = x + t_emb
        if condition is not None:
            x = x + condition

        return self.net(x)

class InductiveEnhancedDiffusion(nn.Module):
    def __init__(
            self,
            h_dim: int = 200,
            num_rels: int = 230,
            diffusion_timesteps: int = 100,
            diffusion_sampling_steps: int = 10,
            use_meta_learning: bool = True,
            meta_lr: float = 0.01,
            use_distribution_alignment: bool = True,
            alignment_strength: float = 0.1,
    ):
        super().__init__()

        self.h_dim = h_dim
        self.num_rels = num_rels
        self.use_meta_learning = use_meta_learning
        self.use_distribution_alignment = use_distribution_alignment
        self.alignment_strength = alignment_strength

        self.ood_detector = EnhancedOODDetector(h_dim, num_rels)
        if use_meta_learning:
            self.meta_learner = MetaLearner(h_dim, num_rels, meta_lr)
        if use_distribution_alignment:
            self.dist_aligner = DistributionAlignment(h_dim)
        self.diffusion_refiner = InductiveDiffusionRefiner(
            h_dim=h_dim,
            num_timesteps=diffusion_timesteps,
            sampling_steps=diffusion_sampling_steps
        )

    def update_seen_statistics(self, seen_embs):
        if self.use_distribution_alignment:
            self.dist_aligner.update_seen_statistics(seen_embs)

    def forward(
            self,
            entity_embs,
            relation_embs,
            mode='train',
            is_seen_labels=None,
            neighbor_embs=None,
            seen_entity_ids=None,
            current_entity_ids=None,
            entity_rel_ids=None,
    ):
        batch_size = entity_embs.shape[0]
        device = entity_embs.device

        if mode == 'train':
            losses = {}
            embs_detached = entity_embs.detach()
            rel_detached = relation_embs.detach()

            seen_embs_for_contrast = (embs_detached[is_seen_labels == 1]
                                      if is_seen_labels is not None else None)
            ood_losses = self.ood_detector.compute_loss(
                embs_detached, is_seen_labels, seen_embs_for_contrast
            )
            losses.update(ood_losses)

            _, diffusion_loss = self.diffusion_refiner(
                embs_detached, mode='train', condition=rel_detached
            )
            losses['diffusion'] = diffusion_loss
            if self.use_distribution_alignment and is_seen_labels is not None:
                unseen_mask = (is_seen_labels == 0)
                if unseen_mask.any():
                    dist_loss = self.dist_aligner.compute_distribution_loss(
                        embs_detached[unseen_mask]
                    )
                    losses['distribution'] = dist_loss
            if self.use_meta_learning and is_seen_labels is not None:
                unseen_mask = (is_seen_labels == 0)
                if unseen_mask.any():
                    unseen_idx = torch.where(unseen_mask)[0]
                    nbr = (neighbor_embs[unseen_idx].detach()
                           if neighbor_embs is not None else None)
                    meta_embs = self.meta_learner.generate_embedding(
                        rel_detached[unseen_idx], nbr
                    )
                    meta_loss = F.mse_loss(meta_embs, embs_detached[unseen_idx])
                    losses['meta'] = meta_loss

            return entity_embs, losses

        else:
            refined_embs = entity_embs.clone()

            if current_entity_ids is not None and seen_entity_ids is not None:
                unseen_mask = torch.tensor(
                    [eid not in seen_entity_ids for eid in current_entity_ids],
                    dtype=torch.bool, device=device
                )
            else:
                is_unseen_prob, _ = self.ood_detector.detect(entity_embs)
                unseen_mask = is_unseen_prob

            num_unseen = unseen_mask.sum().item()

            if num_unseen > 0:
                unseen_idx = torch.where(unseen_mask)[0]
                unseen_embs_raw = entity_embs[unseen_idx]
                unseen_rel_embs = relation_embs[unseen_idx]

                if entity_rel_ids is not None:
                    relation_ids = entity_rel_ids[unseen_idx]
                else:
                    relation_ids = torch.zeros(len(unseen_idx), dtype=torch.long, device=device)

                nbr_for_unseen = neighbor_embs[unseen_idx] if neighbor_embs is not None else None

                if self.use_meta_learning:
                    unseen_embs_init = self.meta_learner.generate_embedding(
                        relation_embs[unseen_idx], nbr_for_unseen
                    )
                else:
                    unseen_embs_init = self.ood_detector.generate_unseen_embedding(
                        relation_ids, nbr_for_unseen
                    )
                seen_mean = self.dist_aligner.seen_mean if self.use_distribution_alignment else None
                seen_std = self.dist_aligner.seen_std if self.use_distribution_alignment else None

                unseen_embs_refined = self.diffusion_refiner.refine_with_alignment(
                    unseen_embs_init,
                    noise_level=0.4,
                    condition=unseen_rel_embs,
                    seen_mean=seen_mean,
                    seen_std=seen_std,
                    alignment_strength=self.alignment_strength
                )
                _, confidence = self.ood_detector.detect(unseen_embs_raw)
                alpha = confidence.unsqueeze(1).clamp(0.0, 0.9)

                fused = alpha * unseen_embs_refined + (1.0 - alpha) * unseen_embs_raw
                refined_embs[unseen_idx] = fused

            return refined_embs
