import argparse
import os
import sys

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from config import get_config
from dataset import load_data
from model import RQEncoder
from tools import set_seed
from datasets import DATASET_CHOICES, apply_dataset_config


DEFAULT_SEED = 3407


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


class RQVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_layers: int,
                 codebook_size: int, encoder_hidden: int = 128,
                 beta: float = 0.25,
                 revival_mu: float = 0.99,
                 revival_threshold: float = 1.0,
                 revival_enabled: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.beta = beta
        self.revival_mu = revival_mu
        self.revival_threshold = revival_threshold
        self.revival_enabled = revival_enabled

        self.encoder = RQEncoder(input_dim, latent_dim, encoder_hidden)
        self.decoder = RQDecoder(latent_dim, input_dim, encoder_hidden)

        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, latent_dim))
            for _ in range(num_layers)
        ])
        # Init EMA at revival_threshold so the first batch doesn't
        # prematurely kill low-count-but-alive codes.
        for l in range(num_layers):
            self.register_buffer(
                f'usage_ema_{l}',
                torch.full((codebook_size,), float(revival_threshold)))

    @torch.no_grad()
    def init_kmeans(self, data: torch.Tensor, kmeans_iters: int = 20,
                    seed: int = 3407):
        z = self.encoder(data)
        x = z
        K = self.codebook_size
        for l in range(self.num_layers):
            N = x.shape[0]
            assert N >= K, (
                f"Layer {l}: codebook_size ({K}) > samples ({N})")
            x_np = x.detach().cpu().numpy()
            km = MiniBatchKMeans(
                n_clusters=K, max_iter=kmeans_iters,
                batch_size=min(4096, N), n_init=3,
                random_state=seed + l).fit(x_np)
            centers = torch.tensor(km.cluster_centers_,
                                   dtype=x.dtype, device=x.device)
            self.codebooks[l].copy_(centers)
            dists = torch.cdist(x, self.codebooks[l])
            indices = dists.argmin(dim=1)
            x = x - self.codebooks[l][indices]

    @torch.no_grad()
    def init_random_sample(self, data: torch.Tensor):
        z = self.encoder(data)
        x = z
        for l in range(self.num_layers):
            K = self.codebook_size
            N = x.shape[0]
            assert N >= K
            perm = torch.randperm(N, device=x.device)[:K]
            self.codebooks[l].copy_(x[perm])
            dists = torch.cdist(x, self.codebooks[l])
            indices = dists.argmin(dim=1)
            x = x - self.codebooks[l][indices]

    @torch.no_grad()
    def _revive_dead_codes(self, layer_idx: int, residual: torch.Tensor,
                           indices: torch.Tensor):
        K = self.codebook_size
        batch_counts = torch.bincount(indices, minlength=K).float()

        usage_ema = getattr(self, f'usage_ema_{layer_idx}')
        usage_ema.mul_(self.revival_mu).add_(
            batch_counts, alpha=(1.0 - self.revival_mu))

        if not self.revival_enabled:
            return

        dead = usage_ema < self.revival_threshold
        num_dead = int(dead.sum().item())
        if num_dead == 0:
            return

        pool = residual.detach()
        N = pool.shape[0]
        # Jukebox: when batch N < K, tile pool with tiny noise to boost
        # diversity. Noise std = 0.01/sqrt(D).
        if N < K:
            reps = (K + N - 1) // N
            tiled = pool.repeat(reps, 1)[:K]
            noise_std = 0.01 / (self.latent_dim ** 0.5)
            sample_pool = tiled + torch.randn_like(tiled) * noise_std
        else:
            sample_pool = pool
        pool_perm = torch.randperm(
            sample_pool.shape[0], device=pool.device)[:num_dead]
        replacements = sample_pool[pool_perm]

        self.codebooks[layer_idx].data[dead] = replacements
        usage_ema[dead] = self.revival_threshold

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        residual = z
        codes = []
        vq_loss = torch.tensor(0.0, device=x.device)
        commit_loss = torch.tensor(0.0, device=x.device)

        for l in range(self.num_layers):
            dists = torch.cdist(residual, self.codebooks[l])
            indices = dists.argmin(dim=1)
            quantized = self.codebooks[l][indices]
            codes.append(quantized)

            vq_loss = vq_loss + (
                (residual.detach() - quantized) ** 2).sum(dim=-1).mean()
            commit_loss = commit_loss + (
                (residual - quantized.detach()) ** 2).sum(dim=-1).mean()

            if self.training:
                self._revive_dead_codes(l, residual, indices)

            residual = residual - quantized.detach()

        z_hat = torch.stack(codes).sum(dim=0)
        z_hat_st = z + (z_hat - z).detach()
        x_hat = self.decoder(z_hat_st)
        recon_loss = ((x - x_hat) ** 2).sum(dim=-1).mean()

        total_loss = recon_loss + vq_loss + self.beta * commit_loss
        combined_loss = vq_loss + self.beta * commit_loss
        return total_loss, recon_loss, combined_loss

    @torch.no_grad()
    def compute_codebook_usage(self, data: torch.Tensor):
        z = self.encoder(data)
        x = z
        usage = []
        for l in range(self.num_layers):
            dists = torch.cdist(x, self.codebooks[l])
            indices = dists.argmin(dim=1)
            usage.append(indices.unique().numel())
            x = x - self.codebooks[l][indices]
        return usage

    @torch.no_grad()
    def compute_recon_error(self, data: torch.Tensor):
        z = self.encoder(data)
        x = z
        for l in range(self.num_layers):
            dists = torch.cdist(x, self.codebooks[l])
            indices = dists.argmin(dim=1)
            x = x - self.codebooks[l][indices]
        z_hat = z - x
        x_hat = self.decoder(z_hat)
        return ((data - x_hat) ** 2).sum(dim=-1).mean()


def train_rqvae(rqvae, data, num_epochs, batch_size, lr, device,
                patience=10, revival_start_epoch: int = 0):
    optimizer = torch.optim.Adam(rqvae.parameters(), lr=lr)
    N = data.shape[0]
    best_total = float('inf')
    patience_counter = 0
    best_state = None

    rqvae.revival_enabled = (revival_start_epoch == 0)

    for epoch in range(num_epochs):
        if epoch == revival_start_epoch:
            rqvae.revival_enabled = True

        perm = torch.randperm(N, device=device)
        loss_sum, recon_sum, quant_sum = 0.0, 0.0, 0.0
        n_batches = 0

        rqvae.train()
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = data[perm[start:end]]

            total, recon, quant = rqvae(batch)  # quant = vq + β·commit
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            loss_sum += total.item()
            recon_sum += recon.item()
            quant_sum += quant.item()
            n_batches += 1

        avg_total = loss_sum / n_batches
        usage = rqvae.compute_codebook_usage(data)
        usage_str = ', '.join(
            [f'L{i+1}={u}/{rqvae.codebook_size}'
             for i, u in enumerate(usage)])

        # Early-stop on TOTAL loss (recon + vq + β·commit), not recon alone,
        # because decoder is dropped after Stage 1 — we care about
        # quantization quality, not decoder fidelity.
        if avg_total < best_total:
            best_total = avg_total
            patience_counter = 0
            best_state = {k: v.clone() for k, v in rqvae.state_dict().items()}
            marker = ' *'
        else:
            patience_counter += 1
            marker = ''

        print(f"Epoch {epoch:3d}: loss={avg_total:.6f} "
              f"(recon={recon_sum/n_batches:.6f} "
              f"quant(vq+β·commit)={quant_sum/n_batches:.6f}) "
              f"usage=[{usage_str}] "
              f"revive={rqvae.revival_enabled}{marker}")

        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch} "
                  f"(best total={best_total:.6f})")
            break

    if best_state is not None:
        rqvae.load_state_dict(best_state)


def main():
    original_argv = list(sys.argv)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', type=str, default=None,
                        choices=DATASET_CHOICES)
    parser.add_argument('--max_layers', type=int, default=8)
    parser.add_argument('--codebook_size', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--encoder_hidden', type=int, default=128)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--rq_epochs', type=int, default=100)
    parser.add_argument('--rq_lr', type=float, default=1e-3)
    parser.add_argument('--rq_patience', type=int, default=10)
    parser.add_argument('--rq_batch_size', type=int, default=2048)
    parser.add_argument('--init_method', type=str, default='kmeans',
                        choices=['kmeans', 'random'])
    parser.add_argument('--kmeans_iters', type=int, default=20)
    parser.add_argument('--revival_mu', type=float, default=0.99,
                        help='EMA decay for usage tracking (Jukebox default 0.99)')
    parser.add_argument('--revival_threshold', type=float, default=1.0,
                        help='EMA usage < threshold → code considered dead')
    parser.add_argument('--revival_start_epoch', type=int, default=0,
                        help='Epoch to start dead-code revival (warmup)')
    parser.add_argument('--no_revival', action='store_true',
                        help='Disable dead-code revival entirely')
    parser.add_argument('--output_dir', type=str, default='./pretrained')
    stage1_args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    config = get_config()

    if stage1_args.dataset is not None:
        apply_dataset_config(config, stage1_args.dataset)

    seed_explicit = any(
        a == '--seed' or a.startswith('--seed=') for a in original_argv)
    if not seed_explicit:
        config.SEED = DEFAULT_SEED

    set_seed(config.SEED)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    os.makedirs(stage1_args.output_dir, exist_ok=True)

    _, _, _, meta_info = load_data(config)
    user_profile_dict = meta_info['user_profile_vectors']

    train_user_ids = meta_info['train_user_ids']
    train_profiles_list = [user_profile_dict[uid] for uid in train_user_ids
                           if uid in user_profile_dict]
    train_profiles = torch.tensor(
        np.array(train_profiles_list)).float().to(device)
    print(f"Training profiles: {train_profiles.shape[0]} users, "
          f"dim={train_profiles.shape[1]}")

    L = stage1_args.max_layers
    K = stage1_args.codebook_size
    d = config.PROFILE_DIM
    ld = stage1_args.latent_dim

    print(f"\nRQ-VAE: L={L}, K={K}, profile_dim={d}, latent_dim={ld}, "
          f"encoder_hidden={stage1_args.encoder_hidden}, "
          f"beta={stage1_args.beta}, epochs={stage1_args.rq_epochs}, "
          f"init={stage1_args.init_method}, "
          f"revival={'off' if stage1_args.no_revival else 'on'} "
          f"(mu={stage1_args.revival_mu}, "
          f"thr={stage1_args.revival_threshold}, "
          f"start_epoch={stage1_args.revival_start_epoch}), "
          f"seed={config.SEED}")

    rqvae = RQVAE(
        input_dim=d, latent_dim=ld, num_layers=L,
        codebook_size=K,
        encoder_hidden=stage1_args.encoder_hidden,
        beta=stage1_args.beta,
        revival_mu=stage1_args.revival_mu,
        revival_threshold=stage1_args.revival_threshold,
        revival_enabled=not stage1_args.no_revival,
    ).to(device)

    if stage1_args.init_method == 'kmeans':
        print(f"Running k-means init in latent space "
              f"(iters={stage1_args.kmeans_iters}; note: encoder is random "
              f"at this point, so this is k-means on a random projection)")
        rqvae.init_kmeans(train_profiles,
                          kmeans_iters=stage1_args.kmeans_iters,
                          seed=config.SEED)
    else:
        rqvae.init_random_sample(train_profiles)

    train_rqvae(rqvae, train_profiles,
                num_epochs=stage1_args.rq_epochs,
                batch_size=stage1_args.rq_batch_size,
                lr=stage1_args.rq_lr,
                device=device,
                patience=stage1_args.rq_patience,
                revival_start_epoch=stage1_args.revival_start_epoch)

    enc_path = os.path.join(stage1_args.output_dir, 'encoder.pt')
    torch.save(rqvae.encoder.state_dict(), enc_path)
    print(f"Encoder -> {enc_path}")

    codebook_sizes = []
    for i in range(L):
        cb = rqvae.codebooks[i].detach().cpu()
        save_path = os.path.join(stage1_args.output_dir,
                                 f'codebook_layer{i+1}.pt')
        torch.save(cb, save_path)
        codebook_sizes.append(cb.shape[0])
        print(f"Layer {i+1}: {cb.shape[0]} codes (dim={cb.shape[1]}) "
              f"-> {save_path}")

    rqvae.eval()
    with torch.no_grad():
        total, recon, quant = rqvae(train_profiles)
    final_usage = rqvae.compute_codebook_usage(train_profiles)
    final_usage = [int(u) for u in final_usage]
    usage_frac = sum(final_usage) / (L * K) if K > 0 else 0.0
    recon_err = rqvae.compute_recon_error(train_profiles)

    meta = {
        'num_layers': L,
        'codebook_size': K,
        'codebook_sizes': codebook_sizes,
        'profile_dim': d,
        'latent_dim': ld,
        'encoder_hidden': stage1_args.encoder_hidden,
        'beta': stage1_args.beta,
        'rq_lr': stage1_args.rq_lr,
        'rq_epochs': stage1_args.rq_epochs,
        'rq_patience': stage1_args.rq_patience,
        'rq_batch_size': stage1_args.rq_batch_size,
        'init_method': stage1_args.init_method,
        'kmeans_iters': stage1_args.kmeans_iters,
        'revival_enabled': not stage1_args.no_revival,
        'revival_mu': stage1_args.revival_mu,
        'revival_threshold': stage1_args.revival_threshold,
        'revival_start_epoch': stage1_args.revival_start_epoch,
        'seed': config.SEED,
        'final_total_loss': float(total.item()),
        'final_recon_loss': float(recon.item()),
        'final_quant_loss': float(quant.item()),
        'final_end_to_end_recon_error': float(recon_err.item()),
        'final_usage_per_layer': final_usage,
        'final_usage_fraction': float(usage_frac),
    }
    meta_path = os.path.join(stage1_args.output_dir, 'meta.pt')
    torch.save(meta, meta_path)
    print(f"Meta -> {meta_path}")

    print(f"\nFinal: total={total.item():.6f} "
          f"recon={recon.item():.6f} "
          f"quant(vq+β·commit)={quant.item():.6f} "
          f"end2end_recon={recon_err.item():.6f} "
          f"usage_frac={usage_frac:.3f} "
          f"usage={final_usage}")
    print(f"Stage 1 done. Outputs saved to {stage1_args.output_dir}")


if __name__ == "__main__":
    main()
