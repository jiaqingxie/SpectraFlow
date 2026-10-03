"""
Baseline training script without per-sample min/max normalization.
This version reshapes spectra directly into 2D heatmaps and trains the
CrossModalVAE end-to-end without rescaling the dynamic range.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
from scipy import interpolate
import argparse
import os
from pathlib import Path
import h5py

from model import CrossModalVAE
from utils import (
    get_device, kl_divergence_loss, reconstruction_loss,
    kl_annealing
)


class PairedModalDatasetBaseline(Dataset):
    """Paired modality dataset without per-sample normalization."""

    def __init__(self, source_csv, target_csv, target_size=3600,
                 resize_shape=(60, 60), out_channels=1, use_h5=True):
        source_h5 = source_csv.replace('.csv', '.h5')
        target_h5 = target_csv.replace('.csv', '.h5')

        source_heatmaps_raw = None
        target_heatmaps_raw = None

        if use_h5 and os.path.exists(source_h5) and os.path.exists(target_h5):
            with h5py.File(source_h5, 'r') as f:
                source_spectra = f['spectra'][:]
                if 'heatmaps_unscaled' in f:
                    source_heatmaps_raw = f['heatmaps_unscaled'][:].astype(np.float32)
            with h5py.File(target_h5, 'r') as f:
                target_spectra = f['spectra'][:]
                if 'heatmaps_unscaled' in f:
                    target_heatmaps_raw = f['heatmaps_unscaled'][:].astype(np.float32)
        else:
            source_df = pd.read_csv(source_csv, header=None)
            source_spectra = source_df.iloc[1:, 1:].values.astype(np.float32)

            target_df = pd.read_csv(target_csv, header=None)
            target_spectra = target_df.iloc[1:, 1:].values.astype(np.float32)

        assert len(source_spectra) == len(target_spectra), (
            "Source and target data lengths must match: "
            f"{len(source_spectra)} vs {len(target_spectra)}"
        )

        self.target_size = target_size
        self.resize_shape = resize_shape
        self.out_channels = out_channels

        self.source_interpolated = [self._interpolate(spec) for spec in source_spectra]
        self.target_interpolated = [self._interpolate(spec) for spec in target_spectra]

        if source_heatmaps_raw is not None:
            self.source_heatmaps = np.expand_dims(source_heatmaps_raw, axis=1)
        else:
            self.source_heatmaps = [self._reshape_to_heatmap(spec) for spec in self.source_interpolated]

        if target_heatmaps_raw is not None:
            self.target_heatmaps = np.expand_dims(target_heatmaps_raw, axis=1)
        else:
            self.target_heatmaps = [self._reshape_to_heatmap(spec) for spec in self.target_interpolated]

    def _interpolate(self, spectrum):
        if spectrum.shape[-1] == self.target_size:
            return spectrum.astype(np.float32)
        x_old = np.linspace(0, 1, len(spectrum))
        x_new = np.linspace(0, 1, self.target_size)
        return interpolate.interp1d(
            x_old, spectrum, kind='linear', fill_value='extrapolate', bounds_error=False
        )(x_new).astype(np.float32)

    def _reshape_to_heatmap(self, spectrum_1d):
        heatmap = spectrum_1d.reshape(self.resize_shape)
        return np.expand_dims(heatmap, axis=0)

    def __len__(self):
        return len(self.source_heatmaps)

    def __getitem__(self, idx):
        source_heatmap = torch.tensor(self.source_heatmaps[idx]).float()
        target_heatmap = torch.tensor(self.target_heatmaps[idx]).float()
        return source_heatmap, target_heatmap


def get_paired_loaders_baseline(source_csv, target_csv, batch_size=32,
                                target_size=3600, resize_shape=(60, 60),
                                out_channels=1, use_h5=True):
    dataset = PairedModalDatasetBaseline(
        source_csv, target_csv,
        target_size=target_size,
        resize_shape=resize_shape,
        out_channels=out_channels,
        use_h5=use_h5
    )

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class CrossModalVAETrainerBaseline:
    """Trainer for baseline setup without per-sample normalization."""

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
            use_physical_prior=False,
            physical_dim=7,
            clamp_output=False
        ).to(device)

        def _init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.model.apply(_init_weights)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.mode_map = {'ir': 0, 'uv': 1, 'raman': 2}

    def train_step(self, source_batch, target_batch, source_mode, target_mode):
        self.model.train()
        self.train_steps += 1

        source_batch = source_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        target_mode_idx = self.mode_map[target_mode]

        recon, mu, logvar = self.model(
            source_batch, target_mode=target_mode_idx, physical_params=None
        )

        recon_loss = reconstruction_loss(recon, target_batch)
        kl_loss = kl_divergence_loss(mu, logvar)

        beta = kl_annealing(
            self.train_steps, self.config.beta_max, k=0.1, x0=500
        ) if self.model.training else self.config.beta_max

        total_loss = recon_loss + beta * kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'beta': beta
        }

    @torch.no_grad()
    def eval_step(self, source_batch, target_batch, target_mode):
        self.model.eval()

        source_batch = source_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        target_mode_idx = self.mode_map[target_mode]

        recon, mu, logvar = self.model(
            source_batch, target_mode=target_mode_idx, physical_params=None
        )

        recon_loss = reconstruction_loss(recon, target_batch)
        kl_loss = kl_divergence_loss(mu, logvar)

        return {
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'recon': recon.cpu(),
            'target': target_batch.cpu()
        }

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
        }, path)


def train_modality_pair_baseline(trainer, train_loader, val_loader,
                                 source_mode, target_mode, epochs=20,
                                 save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')

    print(f"\nTraining baseline {source_mode} -> {target_mode}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        trainer.model.train()
        train_losses = {'total': [], 'recon': [], 'kl': []}

        for source_batch, target_batch in train_loader:
            losses = trainer.train_step(
                source_batch, target_batch, source_mode, target_mode
            )
            train_losses['total'].append(losses['total_loss'])
            train_losses['recon'].append(losses['recon_loss'])
            train_losses['kl'].append(losses['kl_loss'])

        trainer.model.eval()
        val_losses = {'recon': [], 'kl': []}

        with torch.no_grad():
            for source_batch, target_batch in val_loader:
                eval_results = trainer.eval_step(
                    source_batch, target_batch, target_mode
                )
                val_losses['recon'].append(eval_results['recon_loss'])
                val_losses['kl'].append(eval_results['kl_loss'])

        avg_train_loss = np.mean(train_losses['total'])
        avg_val_recon = np.mean(val_losses['recon'])

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f} "
              f"(Recon: {np.mean(train_losses['recon']):.6f}, "
              f"KL: {np.mean(train_losses['kl']):.6f})")
        print(f"  Val Recon: {avg_val_recon:.6f}")

        if avg_val_recon < best_val_loss:
            best_val_loss = avg_val_recon
            checkpoint_path = os.path.join(
                save_dir,
                f'vae_baseline_{source_mode}2{target_mode}_best.pt'
            )
            trainer.save_checkpoint(checkpoint_path)
            print(f"  -> Saved best baseline model (val_loss: {best_val_loss:.6f})")

    print(f"Baseline training completed. Best val loss: {best_val_loss:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description='Train baseline Cross-Modal VAE without normalization')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed CSV files')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='Learning rate')
    parser.add_argument('--beta_max', type=float, default=0.001, help='Max KL weight')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--modes', nargs='+', default=['ir', 'uv', 'raman'],
                        help='Modalities to include')
    args = parser.parse_args()

    device = get_device(args.cpu)
    print(f"Using device: {device}")

    class Config:
        in_channels = 1
        hidden_channels = args.hidden_channels
        latent_dim = args.latent_dim
        num_modes = 3
        resize_shape = (60, 60)
        learning_rate = args.learning_rate
        beta_max = args.beta_max
        batch_size = args.batch_size

    config = Config()

    trainer = CrossModalVAETrainerBaseline(config, device)

    data_dir = Path(args.data_dir)
    mode_pairs = []
    for source_mode in args.modes:
        for target_mode in args.modes:
            mode_pairs.append((source_mode, target_mode))

    for source_mode, target_mode in mode_pairs:
        source_file = f'{source_mode}_broaden_processed.csv'
        target_file = f'{target_mode}_broaden_processed.csv'

        source_csv = data_dir / source_file
        target_csv = data_dir / target_file

        source_h5 = data_dir / f'{source_mode}_broaden_processed.h5'
        target_h5 = data_dir / f'{target_mode}_broaden_processed.h5'

        if not (source_csv.exists() or source_h5.exists()) or not (target_csv.exists() or target_h5.exists()):
            print(f"Skipping {source_mode} -> {target_mode}: data files not found")
            continue

        train_loader, val_loader, _ = get_paired_loaders_baseline(
            str(source_csv), str(target_csv),
            batch_size=config.batch_size,
            target_size=3600,
            resize_shape=config.resize_shape,
            out_channels=config.in_channels,
            use_h5=True
        )

        train_modality_pair_baseline(
            trainer, train_loader, val_loader,
            source_mode, target_mode,
            epochs=args.epochs,
            save_dir=args.save_dir
        )

    print("All baseline trainings completed!")


if __name__ == '__main__':
    main()
