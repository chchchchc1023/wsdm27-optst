import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from dcn import DCNModel


def compute_semantic_densities(profile_vectors: torch.Tensor, k: int = 20,
                               batch_size: int = 1024,
                               reference_vectors: torch.Tensor = None
                               ) -> torch.Tensor:
    device = profile_vectors.device
    N = profile_vectors.shape[0]
    query_norms = F.normalize(profile_vectors, p=2, dim=-1)

    self_ref = reference_vectors is None
    if self_ref:
        ref_norms = query_norms
        actual_k = min(k, N - 1)
    else:
        ref_norms = F.normalize(reference_vectors, p=2, dim=-1)
        actual_k = min(k, ref_norms.shape[0])

    densities = torch.zeros(N, device=device)
    if actual_k == 0:
        return densities

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = query_norms[start:end]
        cos_sim = batch @ ref_norms.T
        if self_ref:
            for j in range(end - start):
                cos_sim[j, start + j] = -2.0
        else:
            cos_sim = cos_sim.masked_fill(cos_sim > 0.999, -2.0)
        topk_cos, _ = cos_sim.topk(actual_k, dim=1)
        densities[start:end] = topk_cos.mean(dim=1)

    mean_d = densities.mean()
    std_d = densities.std() + 1e-8
    densities = ((densities - mean_d) / std_d).clamp(-3.0, 3.0)
    return densities


class RQEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class RQDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class GateNetwork(nn.Module):
    INPUT_DIM = 7

    def __init__(self, hidden_dim: int = 16, init_bias: float = 1.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        with torch.no_grad():
            self.mlp[-1].bias[0] = init_bias        # continue
            self.mlp[-1].bias[1] = -init_bias       # stop

    def forward(self, residual_norm, log_res_ratio, cos_rc, margin,
                log_user_freq, log_item_freq, layer_frac):
        gate_input = torch.cat(
            [residual_norm, log_res_ratio, cos_rc, margin,
             log_user_freq, log_item_freq, layer_frac],
            dim=1)
        return self.mlp(gate_input)


class LoRADelta(nn.Module):
    def __init__(self, dim: int, rank: int, init_scale: float = 0.01,
                 rank_anchor: int = 8):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.W_down = nn.Parameter(torch.empty(dim, rank))
        self.W_up = nn.Parameter(torch.zeros(rank, dim))
        init_std = init_scale * math.sqrt(rank_anchor / rank)
        nn.init.normal_(self.W_down, std=init_std)

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        return C @ self.W_down @ self.W_up


class OptSTModel(nn.Module):
    def __init__(self, profile_dim: int, latent_dim: int,
                 sid_output_dim: int,
                 feature_dims: Dict[str, int], config,
                 max_layers: int = 4,
                 anchor_m: int = 5, anchor_epsilon: float = 0.0):
        super().__init__()
        self.profile_dim = profile_dim
        self.latent_dim = latent_dim
        self.sid_output_dim = sid_output_dim
        self.max_layers = max_layers
        self.anchor_m = anchor_m
        self.anchor_epsilon = anchor_epsilon

        self.tau_depth = float(getattr(config, 'TAU_DEPTH_INIT', 1.0))
        self.tau_ste = float(getattr(config, 'STE_TEMPERATURE', 1.0))

        # LoRA is mandatory in this variant.
        self.lora_rank = int(getattr(config, 'LORA_RANK', 4))
        if self.lora_rank <= 0:
            raise ValueError(
                f"OptSTModel requires --lora_rank > 0 (got {self.lora_rank}).")
        self.lora_init_scale = float(getattr(config, 'LORA_INIT_SCALE', 0.01))

        encoder_hidden = getattr(config, 'ENCODER_HIDDEN_DIM', 128)
        self.encoder = RQEncoder(profile_dim, latent_dim, encoder_hidden)
        self.decoder = RQDecoder(latent_dim, profile_dim, encoder_hidden)

        self.lora_adapters = nn.ModuleList([
            LoRADelta(dim=latent_dim, rank=self.lora_rank,
                      init_scale=self.lora_init_scale)
            for _ in range(max_layers)
        ])
        print(f"[LoRA delta] rank={self.lora_rank}, "
              f"params/layer = 2·{latent_dim}·{self.lora_rank} = "
              f"{2 * latent_dim * self.lora_rank}")

        self.code_embeddings = nn.ModuleList()
        self._codebook_sizes = []

        self.gates = nn.ModuleList([
            GateNetwork(
                hidden_dim=getattr(config, 'GATE_HIDDEN_DIM', 16),
                init_bias=getattr(config, 'GATE_INIT_BIAS', 1.5),
            )
            for _ in range(max_layers - 1)
        ])

        self.ctr_predictor = DCNModel(
            feature_dims=feature_dims,
            embed_dim=config.EMBED_DIM,
            sid_embed_dim=sid_output_dim,
            cross_layers=config.DCN_CROSS_LAYERS,
            deep_layers=config.DCN_DEEP_LAYERS,
            dropout=config.DROPOUT,
        )

        self.register_buffer('user_log_freq', torch.zeros(1))
        self.register_buffer('item_log_freq', torch.zeros(1))

        self._mean_depth = None
        self._config = config

    def load_pretrained(self, pretrained_dir: str, device: torch.device = None):
        if device is None:
            device = next(self.parameters()).device

        meta_path = os.path.join(pretrained_dir, 'meta.pt')
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location='cpu')
            assert meta['num_layers'] == self.max_layers, (
                f"Pretrained has {meta['num_layers']} layers, "
                f"model expects {self.max_layers}")
            assert meta['latent_dim'] == self.latent_dim, (
                f"Pretrained latent_dim={meta['latent_dim']}, "
                f"model latent_dim={self.latent_dim}")
            assert meta['profile_dim'] == self.profile_dim, (
                f"Pretrained profile_dim={meta['profile_dim']}, "
                f"model profile_dim={self.profile_dim}")

        enc_path = os.path.join(pretrained_dir, 'encoder.pt')
        dec_path = os.path.join(pretrained_dir, 'decoder.pt')
        self.encoder.load_state_dict(torch.load(enc_path, map_location=device))
        if os.path.exists(dec_path):
            self.decoder.load_state_dict(torch.load(dec_path, map_location=device))
            print(f"Loaded encoder/decoder from {pretrained_dir}")
        else:
            print(f"Loaded encoder from {pretrained_dir} (no decoder.pt)")

        for i in range(self.max_layers):
            path = os.path.join(pretrained_dir, f'codebook_layer{i+1}.pt')
            centers = torch.load(path, map_location=device)
            K_i = centers.shape[0]
            assert centers.shape[1] == self.latent_dim, (
                f"Layer {i+1} codebook dim {centers.shape[1]} "
                f"!= model latent_dim {self.latent_dim}")
            self._codebook_sizes.append(K_i)
            print(f"Layer {i+1}: loaded {K_i} frozen centers "
                  f"(dim={centers.shape[1]})")

            self.register_buffer(f'codebook_{i}', centers.detach().clone())

            emb = nn.Embedding(K_i, self.sid_output_dim)
            nn.init.xavier_uniform_(emb.weight)
            self.code_embeddings.append(emb.to(device))

        self._build_anchor_data()

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def _build_anchor_data(self):
        for i in range(self.max_layers):
            C = getattr(self, f'codebook_{i}')
            K_i = C.shape[0]
            M = min(self.anchor_m, K_i - 1)
            if M == 0:
                self.register_buffer(f'anchor_nb_{i}',
                                     torch.zeros(K_i, 0, dtype=torch.long,
                                                 device=C.device))
                self.register_buffer(f'anchor_cos_{i}',
                                     torch.zeros(K_i, 0, device=C.device))
                continue

            C_norm = F.normalize(C, p=2, dim=-1)
            cos_sim = C_norm @ C_norm.T
            cos_masked = cos_sim.clone()
            cos_masked.fill_diagonal_(-2.0)
            _, nb_idx = cos_masked.topk(M, dim=1)
            orig_cos = torch.gather(cos_sim, 1, nb_idx)

            self.register_buffer(f'anchor_nb_{i}', nb_idx)
            self.register_buffer(f'anchor_cos_{i}', orig_cos)

    def _layer_delta(self, layer_idx: int) -> torch.Tensor:
        C = getattr(self, f'codebook_{layer_idx}')
        return self.lora_adapters[layer_idx](C)

    def get_adjusted_codebook(self, layer_idx: int) -> torch.Tensor:
        C = getattr(self, f'codebook_{layer_idx}')
        return C + self._layer_delta(layer_idx)

    def set_user_item_freqs(self, user_counts: torch.Tensor,
                            item_counts: torch.Tensor):
        log_uf = torch.log1p(user_counts.float())
        log_uf = log_uf - log_uf.mean()
        self.user_log_freq = log_uf

        log_if = torch.log1p(item_counts.float())
        log_if = log_if - log_if.mean()
        self.item_log_freq = log_if

    def set_temperature(self, tau_depth: float):
        self.tau_depth = tau_depth

    @property
    def num_layers(self):
        return self.max_layers

    def _ste_lookup(self, dists: torch.Tensor, hard_id: torch.Tensor,
                    emb_table: nn.Embedding) -> torch.Tensor:
        soft_p = F.softmax(-dists / self.tau_ste, dim=1)
        out_soft = soft_p @ emb_table.weight
        out_hard = emb_table(hard_id)
        return out_soft + (out_hard - out_soft).detach()

    def forward(self, profile_embeddings: torch.Tensor,
                features: Dict[str, torch.Tensor]):
        profile_embeddings = profile_embeddings.detach()
        batch_size = profile_embeddings.shape[0]
        device = profile_embeddings.device

        z = self.encoder(profile_embeddings)
        z_norm = z.norm(dim=1, keepdim=True).clamp_min(1e-8)

        user_ids = features['user_id']
        item_ids = features['item_id']

        if self.user_log_freq.shape[0] > 1:
            log_user_freq = self.user_log_freq[user_ids].unsqueeze(1)
        else:
            log_user_freq = torch.zeros(batch_size, 1, device=device)
        if self.item_log_freq.shape[0] > 1:
            log_item_freq = self.item_log_freq[item_ids].unsqueeze(1)
        else:
            log_item_freq = torch.zeros(batch_size, 1, device=device)

        sid_features = torch.zeros(batch_size, self.sid_output_dim,
                                   device=device)

        if self.training:
            self._forward_train(z, z_norm, log_user_freq, log_item_freq,
                                sid_features, batch_size, device)
        else:
            self._forward_eval(z, z_norm, log_user_freq, log_item_freq,
                               sid_features, batch_size, device)

        return self.ctr_predictor(features, sid_features)

    def _forward_train(self, z, z_norm, log_user_freq, log_item_freq,
                       sid_features, batch_size, device):
        x = z
        cum_continue = torch.ones(batch_size, 1, device=device)
        depth_contributions = []
        all_codes = []

        for i in range(self.max_layers):
            C_tilde = self.get_adjusted_codebook(i)
            K_i = C_tilde.shape[0]

            dists = torch.cdist(x, C_tilde)
            hard_indices = dists.argmin(dim=1)
            code = C_tilde[hard_indices]            # for residual + gate inputs
            all_codes.append(code)

            # Main path: STE lookup (gradient to Emb + LoRA).
            encoded = self._ste_lookup(dists, hard_indices,
                                       self.code_embeddings[i])
            sid_features.add_(cum_continue * encoded)

            residual = x - code

            if i < self.max_layers - 1:
                sorted_dist, _ = dists.sort(dim=1)
                d_best = sorted_dist[:, 0:1]
                d_second = (sorted_dist[:, 1:2] if K_i > 1
                            else torch.zeros_like(d_best))
                margin = ((d_second - d_best)
                          / (d_best + 1e-8)).clamp(-5.0, 5.0)

                residual_norm = residual.norm(dim=1, keepdim=True)
                log_res_ratio = (residual_norm / z_norm).clamp_min(
                    1e-8).log().clamp(-10.0, 0.0)
                cos_rc = F.cosine_similarity(
                    residual, code, dim=1, eps=1e-8).unsqueeze(1)
                layer_frac = torch.full(
                    (batch_size, 1), (i + 1) / self.max_layers,
                    device=device)

                gate_logits = self.gates[i](
                    residual_norm, log_res_ratio, cos_rc, margin,
                    log_user_freq, log_item_freq, layer_frac)

                soft_gate = F.softmax(gate_logits / self.tau_depth, dim=1)
                hard_gate_idx = gate_logits.argmax(dim=1)

                hard_gate = F.one_hot(hard_gate_idx, 2).float()
                st_gate = hard_gate - soft_gate.detach() + soft_gate
                continue_val = st_gate[:, 0:1]
                cum_continue = cum_continue * continue_val
                depth_contributions.append(cum_continue)

            x = residual

        # Mean depth (soft, for depth_lambda regularizer)
        if depth_contributions:
            depth_stack = torch.cat(depth_contributions, dim=1)
            self._mean_depth = 1.0 + depth_stack.sum(dim=1).mean()
        else:
            self._mean_depth = torch.tensor(1.0, device=device)


    def _forward_eval(self, z, z_norm, log_user_freq, log_item_freq,
                      sid_features, batch_size, device):
        x = z.clone()
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        sample_depths = torch.ones(batch_size, device=device)

        for i in range(self.max_layers):
            if not active.any():
                break

            active_idx = active.nonzero(as_tuple=True)[0]
            n_active = active_idx.numel()
            x_active = x[active_idx]

            C_tilde = self.get_adjusted_codebook(i)
            K_i = C_tilde.shape[0]

            dists = torch.cdist(x_active, C_tilde)
            hard_indices = dists.argmin(dim=1)
            code = C_tilde[hard_indices]

            encoded = self.code_embeddings[i](hard_indices)
            sid_features[active_idx] = sid_features[active_idx] + encoded

            residual = x_active - code
            x[active_idx] = residual

            if i < self.max_layers - 1:
                sorted_dist, _ = dists.sort(dim=1)
                d_best = sorted_dist[:, 0:1]
                d_second = (sorted_dist[:, 1:2] if K_i > 1
                            else torch.zeros_like(d_best))
                margin = ((d_second - d_best)
                          / (d_best + 1e-8)).clamp(-5.0, 5.0)

                residual_norm = residual.norm(dim=1, keepdim=True)
                log_res_ratio = (residual_norm / z_norm[active_idx]).clamp_min(
                    1e-8).log().clamp(-10.0, 0.0)
                cos_rc = F.cosine_similarity(
                    residual, code, dim=1, eps=1e-8).unsqueeze(1)
                layer_frac = torch.full(
                    (n_active, 1), (i + 1) / self.max_layers,
                    device=device)

                gate_logits = self.gates[i](
                    residual_norm, log_res_ratio, cos_rc, margin,
                    log_user_freq[active_idx], log_item_freq[active_idx],
                    layer_frac)

                continue_decision = (gate_logits.argmax(dim=1) == 0)

                continuing_idx = active_idx[continue_decision]
                stopping_idx = active_idx[~continue_decision]

                sample_depths[continuing_idx] += 1
                active[stopping_idx] = False

        self._mean_depth = sample_depths.mean()

    def compute_depth_loss(self):
        return self._mean_depth / self.max_layers

    def _has_offset(self) -> bool:
        return self.lora_adapters is not None and len(self.lora_adapters) > 0

    def _offset_device(self):
        return self.lora_adapters[0].W_down.device

    def compute_anchor_loss(self):
        if not self._has_offset():
            return torch.tensor(0.0)
        device = self._offset_device()

        per_layer_losses = []
        for i in range(self.max_layers):
            delta = self._layer_delta(i)
            K_i, d = delta.shape[0], delta.shape[1]
            offset_penalty = (delta ** 2).sum() / (K_i * d)

            nb = getattr(self, f'anchor_nb_{i}', None)
            cos_orig = getattr(self, f'anchor_cos_{i}', None)

            topo_loss = torch.tensor(0.0, device=device)
            if nb is not None and cos_orig is not None and nb.shape[1] > 0:
                M_eff = nb.shape[1]
                C_tilde = self.get_adjusted_codebook(i)
                C_tilde_norm = F.normalize(C_tilde, p=2, dim=-1)
                nb_embs = C_tilde_norm[nb.reshape(-1)].reshape(K_i, M_eff, -1)
                cos_new = (C_tilde_norm.unsqueeze(1) * nb_embs).sum(-1)
                topo_loss = F.relu(
                    cos_new - cos_orig - self.anchor_epsilon).sum()
                topo_loss = topo_loss / (K_i * M_eff)

            per_layer_losses.append(offset_penalty + topo_loss)

        return torch.stack(per_layer_losses).mean()
