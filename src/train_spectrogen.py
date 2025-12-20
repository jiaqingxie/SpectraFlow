"""
SpectroGen training with physical prior VAE.
- Uses CrossModalVAE with use_physical_prior=True.
- KL term uses physical prior derived from source/target spectrum statistics.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import CrossModalVAE
from train import PairedModalDataset
from utils import (
    compute_params,
    compute_prior_distribution,
    compute_kl_loss_with_prior,
    get_device,
    kl_annealing,
    reconstruction_loss,
)


def batch_compute_params(spec_batch):
    """Compute physical params for a batch of 1D spectra tensor (B, L)."""
    params_list = []
    for spec in spec_batch.cpu().numpy():
        params_list.append(compute_params(spec))
    return params_list


def batch_compute_bandwidth_std(spec_batch):
    """Fast computation of only bandwidth and std for a batch (B, L)."""
    specs_np = spec_batch.cpu().numpy()
    batch_size = specs_np.shape[0]
    bandwidths = np.zeros(batch_size, dtype=np.float32)
    stds = np.zeros(batch_size, dtype=np.float32)
    
    for i, spec in enumerate(specs_np):
        stds[i] = np.std(spec)
        max_intensity = np.max(spec)
        if max_intensity > 0:
            half_max = max_intensity / 2.0
            indices_above = np.where(spec >= half_max)[0]
            if len(indices_above) > 0:
                fwhm = indices_above[-1] - indices_above[0]
                bandwidths[i] = fwhm / len(spec) if len(spec) > 0 else 0.1
            else:
                bandwidths[i] = 0.1
        else:
            bandwidths[i] = 0.1
    
    return bandwidths, stds


def params_to_vec(params):
    """Convert params dict to fixed 7-dim vector."""
    return np.array(
        [
            params.get('mean', 0.0),
            params.get('std', 0.0),
            params.get('bandwidth', 0.0),
            len(params.get('peak_positions', [])),
            params.get('max_intensity', 0.0),
            params.get('energy_range', (0.0, 0.0))[0],
            params.get('energy_range', (0.0, 0.0))[1],
        ],
        dtype=np.float32,
    )


class SpectroGenTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.train_steps = 0

        self.model = CrossModalVAE(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            latent_dim=config.latent_dim,
            image_size=config.resize_shape,
            num_modes=config.num_modes,
            use_physical_prior=True,
            physical_dim=7,
        ).to(device)

        def _init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.model.apply(_init_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.mode_map = {'ir': 0, 'uv': 1, 'raman': 2}

    def train_step(self, batch, target_mode):
        self.model.train()
        self.train_steps += 1

        source_batch = batch[0].to(self.device)
        target_batch = batch[1].to(self.device)
        
        # Use pre-computed physical params from dataset if available (batch[8])
        if len(batch) > 8 and batch[8] is not None:
            phys_src_vec = batch[8].to(self.device)  # (B, 7) - pre-computed
        else:
            # Fallback: compute on-the-fly (slower)
            src_orig = batch[6]
            phys_src = batch_compute_params(src_orig)
            phys_src_vec = torch.from_numpy(np.array([params_to_vec(p) for p in phys_src], dtype=np.float32)).to(self.device)
        
        # For target, compute only bandwidth and std (faster than full compute_params)
        tgt_orig = batch[7]
        tgt_bw, tgt_std = batch_compute_bandwidth_std(tgt_orig)

        target_mode_idx = self.mode_map[target_mode]
        recon, mu, logvar = self.model(source_batch, target_mode=target_mode_idx, physical_params=phys_src_vec)

        recon_loss = reconstruction_loss(recon, target_batch)

        # Vectorized prior computation from pre-computed params
        # phys_src_vec shape: (B, 7) where [:, 1] = std, [:, 2] = bandwidth
        src_std = phys_src_vec[:, 1].cpu().numpy()  # (B,)
        src_bw = phys_src_vec[:, 2].cpu().numpy()   # (B,)
        
        # Vectorized prior computation: prior_mu = log(tgt_bw / (src_bw + eps) + eps)
        prior_mu = np.log(tgt_bw / (src_bw + 1e-8) + 1e-8)
        # prior_logvar = log((src_std^2 + tgt_std^2) + eps)
        prior_std_sq = src_std ** 2 + tgt_std ** 2
        prior_logvar = np.log(prior_std_sq + 1e-8)
        
        # Convert to tensors and expand to match (batch_size, latent_dim)
        prior_mu = torch.from_numpy(prior_mu).to(mu.dtype).to(self.device)
        prior_logvar = torch.from_numpy(prior_logvar).to(logvar.dtype).to(self.device)
        prior_mu = prior_mu.unsqueeze(1).expand_as(mu)
        prior_logvar = prior_logvar.unsqueeze(1).expand_as(logvar)

        kl_loss = compute_kl_loss_with_prior(mu, logvar, prior_mu, prior_logvar)

        beta = kl_annealing(self.train_steps, self.config.beta_max, k=0.1, x0=500)
        total_loss = recon_loss + beta * kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'beta': beta,
        }

    @torch.no_grad()
    def eval_step(self, batch, target_mode):
        self.model.eval()
        source_batch = batch[0].to(self.device)
        target_batch = batch[1].to(self.device)
        
        # Use pre-computed physical params from dataset if available
        if len(batch) > 8 and batch[8] is not None:
            phys_src_vec = batch[8].to(self.device)
        else:
            src_orig = batch[6]
            phys_src = batch_compute_params(src_orig)
            phys_src_vec = torch.from_numpy(np.array([params_to_vec(p) for p in phys_src], dtype=np.float32)).to(self.device)
        
        # Fast computation for target
        tgt_orig = batch[7]
        tgt_bw, tgt_std = batch_compute_bandwidth_std(tgt_orig)

        target_mode_idx = self.mode_map[target_mode]
        recon, mu, logvar = self.model(source_batch, target_mode=target_mode_idx, physical_params=phys_src_vec)

        recon_loss = reconstruction_loss(recon, target_batch)
        
        # Vectorized prior computation
        src_std = phys_src_vec[:, 1].cpu().numpy()
        src_bw = phys_src_vec[:, 2].cpu().numpy()
        prior_mu = np.log(tgt_bw / (src_bw + 1e-8) + 1e-8)
        prior_std_sq = src_std ** 2 + tgt_std ** 2
        prior_logvar = np.log(prior_std_sq + 1e-8)
        
        prior_mu = torch.from_numpy(prior_mu).to(mu.dtype).to(self.device)
        prior_logvar = torch.from_numpy(prior_logvar).to(logvar.dtype).to(self.device)
        prior_mu = prior_mu.unsqueeze(1).expand_as(mu)
        prior_logvar = prior_logvar.unsqueeze(1).expand_as(logvar)
        kl_loss = compute_kl_loss_with_prior(mu, logvar, prior_mu, prior_logvar)

        return {'recon_loss': recon_loss.item(), 'kl_loss': kl_loss.item()}

    def save_checkpoint(self, path):
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_steps': self.train_steps,
            },
            path,
        )


def get_paired_loaders_phys(source_csv, target_csv, batch_size, heatmap_size, resize_shape, out_channels, seed=42):
    dataset = PairedModalDataset(
        source_csv,
        target_csv,
        source_size=None,
        target_size=None,
        heatmap_size=heatmap_size,
        resize_shape=resize_shape,
        out_channels=out_channels,
        use_h5=True,
    )
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_modality_pair(trainer, train_loader, val_loader, source_mode, target_mode, epochs, save_dir, seed):
    os.makedirs(save_dir, exist_ok=True)
    best_val = float('inf')
    print(f"\nTraining SpectroGen (physical prior): {source_mode} -> {target_mode}")
    print("=" * 60)
    for epoch in range(epochs):
        trainer.model.train()
        train_losses = []
        for batch in train_loader:
            losses = trainer.train_step(batch, target_mode)
            train_losses.append(losses['total_loss'])
        trainer.model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                eval_res = trainer.eval_step(batch, target_mode)
                val_losses.append(eval_res['recon_loss'])
        avg_train = float(np.mean(train_losses)) if train_losses else 0.0
        avg_val = float(np.mean(val_losses)) if val_losses else 0.0
        print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train:.6f} | Val recon: {avg_val:.6f}")
        if avg_val < best_val:
            best_val = avg_val
            ckpt = os.path.join(save_dir, f'spectrogen_{source_mode}2{target_mode}_best_seed{seed}.pt')
            trainer.save_checkpoint(ckpt)
            print(f"  -> Saved best (val recon {best_val:.6f})")
    print(f"Training completed. Best val recon: {best_val:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Train SpectroGen (VAE with physical prior)")
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--beta_max', type=float, default=0.001)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--source_mode', type=str, required=True, choices=['ir', 'uv', 'raman'])
    parser.add_argument('--target_mode', type=str, required=True, choices=['ir', 'uv', 'raman'])
    parser.add_argument('--heatmap_size', type=int, default=3600)
    parser.add_argument('--resize_shape', type=int, nargs=2, default=[60, 60])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = get_device(args.cpu)
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    source_csv = data_dir / f'{args.source_mode}_broaden_processed.csv'
    target_csv = data_dir / f'{args.target_mode}_broaden_processed.csv'
    if not source_csv.exists() or not target_csv.exists():
        print("Error: data files not found")
        print(f"  {source_csv}")
        print(f"  {target_csv}")
        return

    class Config:
        in_channels = 1
        hidden_channels = args.hidden_channels
        latent_dim = args.latent_dim
        num_modes = 3
        resize_shape = tuple(args.resize_shape)
        learning_rate = args.learning_rate
        beta_max = args.beta_max

    config = Config()
    trainer = SpectroGenTrainer(config, device)
    train_loader, val_loader, _ = get_paired_loaders_phys(
        str(source_csv),
        str(target_csv),
        batch_size=args.batch_size,
        heatmap_size=args.heatmap_size,
        resize_shape=config.resize_shape,
        out_channels=1,
        seed=args.seed,
    )
    train_modality_pair(
        trainer,
        train_loader,
        val_loader,
        args.source_mode,
        args.target_mode,
        epochs=args.epochs,
        save_dir=args.save_dir,
        seed=args.seed,
    )
    print("SpectroGen training finished.")


if __name__ == '__main__':
    main()

