"""
Training pipeline:
- UNet translator: direct IR/UV/Raman heatmap-to-heatmap regression
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import get_paired_loaders
from utils import get_device


class FiLM2D(nn.Module):
    """Feature-wise linear modulation for 2D feature maps."""

    def __init__(self, embedding_dim: int, feature_dim: int):
        super().__init__()
        self.scale = nn.Linear(embedding_dim, feature_dim)
        self.shift = nn.Linear(embedding_dim, feature_dim)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        scale = self.scale(embedding).unsqueeze(-1).unsqueeze(-1)
        shift = self.shift(embedding).unsqueeze(-1).unsqueeze(-1)
        return scale * x + shift


class UNetTranslator(nn.Module):
    """Simple conditional UNet for 2D heatmap translation."""

    def __init__(self, in_channels: int = 1, hidden_channels: int = 128, num_modes: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        embed_dim = hidden_channels
        self.mode_embed = nn.Embedding(num_modes, embed_dim)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.film1 = FiLM2D(embed_dim, hidden_channels)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.SiLU(),
        )
        self.film2 = FiLM2D(embed_dim, hidden_channels * 2)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.SiLU(),
        )
        self.film3 = FiLM2D(embed_dim, hidden_channels * 4)

        self.middle = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.SiLU(),
        )
        self.film_middle = FiLM2D(embed_dim, hidden_channels * 4)

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels * 4, hidden_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.SiLU(),
        )
        self.film_dec3 = FiLM2D(embed_dim, hidden_channels * 2)

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels * 4, hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.film_dec2 = FiLM2D(embed_dim, hidden_channels)

        self.decoder1 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, target_mode: int) -> torch.Tensor:
        if isinstance(target_mode, torch.Tensor):
            mode_idx = target_mode.long()
        else:
            mode_idx = torch.tensor([target_mode] * x.size(0), device=x.device, dtype=torch.long)

        cond = self.mode_embed(mode_idx)

        h1 = self.film1(self.encoder1(x), cond)
        h2 = self.film2(self.encoder2(h1), cond)
        h3 = self.film3(self.encoder3(h2), cond)

        h_mid = self.film_middle(self.middle(h3), cond)

        h3_up = self.decoder3(h_mid)
        h3_up = self.film_dec3(h3_up, cond)
        h3_up = torch.cat([h3_up, h2], dim=1)

        h2_up = self.decoder2(h3_up)
        h2_up = self.film_dec2(h2_up, cond)
        h2_up = torch.cat([h2_up, h1], dim=1)

        out = self.decoder1(h2_up)
        return torch.sigmoid(out)  # keep outputs in [0,1]


class UNetTrainer:
    """Trainer for the UNet translator pipeline."""

    def __init__(self, config, device):
        self.device = device
        self.mode_map = {'ir': 0, 'uv': 1, 'raman': 2}
        self.model = UNetTranslator(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            num_modes=config.num_modes,
        ).to(device)

        def _init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.model.apply(_init_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.train_steps = 0

    def train_step(self, source_batch, target_batch, target_mode):
        self.model.train()
        self.train_steps += 1

        source_batch = source_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        target_mode_idx = self.mode_map[target_mode]

        pred = self.model(source_batch, target_mode_idx)
        l1 = F.l1_loss(pred, target_batch)
        l2 = F.mse_loss(pred, target_batch)
        loss = 0.6 * l1 + 0.4 * l2

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {'loss': loss.item(), 'l1': l1.item(), 'l2': l2.item()}

    @torch.no_grad()
    def eval_step(self, source_batch, target_batch, target_mode):
        self.model.eval()

        source_batch = source_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        target_mode_idx = self.mode_map[target_mode]

        pred = self.model(source_batch, target_mode_idx)
        l1 = F.l1_loss(pred, target_batch)
        l2 = F.mse_loss(pred, target_batch)
        loss = 0.6 * l1 + 0.4 * l2

        return {'loss': loss.item(), 'l1': l1.item(), 'l2': l2.item()}

    def save_checkpoint(self, path):
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_steps': self.train_steps,
            },
            path,
        )


def train_unet_pair(
    trainer: UNetTrainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    source_mode: str,
    target_mode: str,
    epochs: int,
    save_dir: str,
    seed: int,
):
    os.makedirs(save_dir, exist_ok=True)
    best_val = float('inf')

    print(f"\nTraining UNet: {source_mode} -> {target_mode}")
    print("=" * 60)

    for epoch in range(epochs):
        trainer.model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            source_batch, target_batch = batch[0], batch[1]
            loss_dict = trainer.train_step(source_batch, target_batch, target_mode)
            train_losses.append(loss_dict['loss'])
            pbar.set_postfix({'loss': f"{loss_dict['loss']:.6f}"})

        trainer.model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                source_batch, target_batch = batch[0], batch[1]
                loss_dict = trainer.eval_step(source_batch, target_batch, target_mode)
                val_losses.append(loss_dict['loss'])

        avg_train = float(np.mean(train_losses)) if train_losses else 0.0
        avg_val = float(np.mean(val_losses)) if val_losses else 0.0

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train:.6f}")
        print(f"  Val Loss:   {avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            ckpt = os.path.join(save_dir, f'unet_{source_mode}2{target_mode}_best_seed{seed}.pt')
            trainer.save_checkpoint(ckpt)
            print(f"  -> Saved best model (val_loss: {best_val:.6f})")

    print(f"Training completed. Best val loss: {best_val:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Train UNet translator pipeline")
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Processed data directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='Learning rate')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint directory (base)')
    parser.add_argument('--no_dataset_subdir', action='store_true',
                        help='Do not auto-create a dataset-specific subdirectory under save_dir (default: False). '
                             'By default, if --data_dir path contains "qm9s", checkpoints will be saved under save_dir/qm9s/.')
    parser.add_argument('--source_mode', type=str, required=True, choices=['ir', 'uv', 'raman'],
                        help='Source modality')
    parser.add_argument('--target_mode', type=str, required=True, choices=['ir', 'uv', 'raman'],
                        help='Target modality')
    parser.add_argument('--source_csv', type=str, default=None,
                        help='Custom source CSV filename (optional, overrides default naming)')
    parser.add_argument('--target_csv', type=str, default=None,
                        help='Custom target CSV filename (optional, overrides default naming)')
    parser.add_argument('--heatmap_size', type=int, default=3600,
                       help='Heatmap size (must be a perfect square, e.g., 3600=60x60, 1024=32x32)')
    parser.add_argument('--resize_shape', type=int, nargs=2, default=[60, 60],
                       help='Heatmap reshape size, e.g., 60 60 or 32 32')
    parser.add_argument('--source_size', type=int, default=None, help='Optional source spectrum length before heatmap')
    parser.add_argument('--target_size', type=int, default=None, help='Optional target spectrum length before heatmap')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # If training on qm9s, save checkpoints into a separate subfolder to avoid collisions with other runs.
    # Users can disable this behavior via --no_dataset_subdir or fully override via --save_dir.
    # Check both data_dir and source/target CSV paths for 'qm9s'
    is_qm9s = False
    if not args.no_dataset_subdir:
        if 'qm9s' in str(args.data_dir).lower():
            is_qm9s = True
        elif args.source_csv and 'qm9' in str(args.source_csv).lower():
            is_qm9s = True
        elif args.target_csv and 'qm9' in str(args.target_csv).lower():
            is_qm9s = True
    
    if is_qm9s:
        args.save_dir = os.path.join(args.save_dir, 'qm9s')
        print(f"[save_dir] Detected qm9s dataset. Saving checkpoints to: {args.save_dir}")

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
    # Use custom filenames if provided, otherwise use default naming
    source_csv = Path(args.source_csv) if args.source_csv and os.path.isabs(args.source_csv) else (data_dir / args.source_csv) if args.source_csv else (data_dir / f'{args.source_mode}_broaden_processed.csv')
    target_csv = Path(args.target_csv) if args.target_csv and os.path.isabs(args.target_csv) else (data_dir / args.target_csv) if args.target_csv else (data_dir / f'{args.target_mode}_broaden_processed.csv')

    if not source_csv.exists() or not target_csv.exists():
        print("Error: data files not found")
        print(f"  {source_csv}")
        print(f"  {target_csv}")
        return

    train_loader, val_loader, _ = get_paired_loaders(
        str(source_csv),
        str(target_csv),
        batch_size=args.batch_size,
        source_size=args.source_size,
        target_size=args.target_size,
        heatmap_size=args.heatmap_size,
        resize_shape=tuple(args.resize_shape),
        out_channels=1,
        use_h5=True,
        seed=args.seed,
    )

    class Config:
        in_channels = 1
        hidden_channels = args.hidden_channels
        num_modes = 3
        learning_rate = args.learning_rate

    config = Config()
    trainer = UNetTrainer(config, device)
    train_unet_pair(
        trainer,
        train_loader,
        val_loader,
        args.source_mode,
        args.target_mode,
        epochs=args.epochs,
        save_dir=args.save_dir,
        seed=args.seed,
    )

    print("Training completed!")


if __name__ == '__main__':
    main()

