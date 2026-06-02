"""
Train Encoder–Decoder (seq2seq) baseline: BiGRU + Bahdanau attention + GRU decoder.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model_seq2seq import Seq2SeqSpectrumTranslator
from train import get_paired_loaders
from utils import get_device


class Seq2SeqTrainer:
    """Trainer for the encoder–decoder baseline."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.train_steps = 0
        self.mode_map = {'ir': 0, 'uv': 1, 'raman': 2}

        self.model = Seq2SeqSpectrumTranslator(
            in_channels=config.in_channels,
            image_size=config.resize_shape,
            patch_size=config.patch_size,
            hidden_dim=config.hidden_dim,
            encoder_depth=config.encoder_depth,
            decoder_depth=config.decoder_depth,
            num_heads=config.num_heads,
            num_modes=config.num_modes,
            dropout=config.dropout,
            clamp_output=True,
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.05,
        )

    def train_step(self, source_batch, target_batch, target_mode):
        self.model.train()
        self.train_steps += 1

        source_batch = source_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        target_mode_idx = self.mode_map[target_mode]

        pred = self.model(source_batch, target_mode=target_mode_idx)
        l1_loss = F.l1_loss(pred, target_batch)
        l2_loss = F.mse_loss(pred, target_batch)
        total_loss = 0.6 * l1_loss + 0.4 * l2_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'l1': l1_loss.item(),
            'l2': l2_loss.item(),
        }

    @torch.no_grad()
    def eval_step(self, source_batch, target_batch, target_mode):
        self.model.eval()

        source_batch = source_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        target_mode_idx = self.mode_map[target_mode]

        pred = self.model(source_batch, target_mode=target_mode_idx)
        l1_loss = F.l1_loss(pred, target_batch)
        l2_loss = F.mse_loss(pred, target_batch)
        total_loss = 0.6 * l1_loss + 0.4 * l2_loss

        return {
            'loss': total_loss.item(),
            'l1': l1_loss.item(),
            'l2': l2_loss.item(),
            'pred': pred.cpu(),
            'target': target_batch.cpu(),
        }

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_steps': self.train_steps,
        }, path)


def train_modality_pair(trainer, train_loader, val_loader, source_mode, target_mode,
                        epochs=50, save_dir='checkpoints', seed=42):
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')

    print(f"\nTraining seq2seq (Encoder–Decoder) baseline: {source_mode} -> {target_mode}")
    print(f"{'=' * 60}")

    for epoch in range(epochs):
        train_losses = []
        train_l1_losses = []
        train_l2_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            source_batch, target_batch = batch[0], batch[1]
            losses = trainer.train_step(source_batch, target_batch, target_mode)
            train_losses.append(losses['loss'])
            train_l1_losses.append(losses['l1'])
            train_l2_losses.append(losses['l2'])
            pbar.set_postfix({'loss': f"{losses['loss']:.6f}"})

        val_losses = []
        val_l1_losses = []
        val_l2_losses = []

        with torch.no_grad():
            for batch in val_loader:
                source_batch, target_batch = batch[0], batch[1]
                eval_results = trainer.eval_step(source_batch, target_batch, target_mode)
                val_losses.append(eval_results['loss'])
                val_l1_losses.append(eval_results['l1'])
                val_l2_losses.append(eval_results['l2'])

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Train L1:   {np.mean(train_l1_losses):.6f}")
        print(f"  Train L2:   {np.mean(train_l2_losses):.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        print(f"  Val L1:     {np.mean(val_l1_losses):.6f}")
        print(f"  Val L2:     {np.mean(val_l2_losses):.6f}")

        trainer.scheduler.step()
        current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(
                save_dir,
                f'seq2seq_{source_mode}2{target_mode}_best_seed{seed}.pt'
            )
            trainer.save_checkpoint(checkpoint_path)
            print(f"  -> Saved best seq2seq model (val_loss: {best_val_loss:.6f})")

    print(f"Seq2seq training completed. Best val loss: {best_val_loss:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Encoder–Decoder (seq2seq) baseline')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed CSV files')
    parser.add_argument('--source_csv', type=str, default=None,
                        help='Custom source CSV filename (optional, absolute or relative to data_dir)')
    parser.add_argument('--target_csv', type=str, default=None,
                        help='Custom target CSV filename (optional, absolute or relative to data_dir)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--depth', type=int, default=6,
                        help='Default encoder/decoder depth when --encoder_depth/--decoder_depth omitted')
    parser.add_argument('--encoder_depth', type=int, default=None,
                        help='Encoder layers (default: same as --depth)')
    parser.add_argument('--decoder_depth', type=int, default=None,
                        help='Kept for API compat; GRU+attention decoder is a single GRUCell loop (unused)')
    parser.add_argument('--num_heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint directory (base)')
    parser.add_argument('--no_dataset_subdir', action='store_true',
                        help='Do not auto-create a dataset-specific subdirectory under save_dir')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--source_mode', type=str, required=True, choices=['ir', 'uv', 'raman'],
                        help='Source modality')
    parser.add_argument('--target_mode', type=str, required=True, choices=['ir', 'uv', 'raman'],
                        help='Target modality')
    parser.add_argument('--heatmap_size', type=int, default=3600,
                        help='Heatmap size (must be a perfect square)')
    parser.add_argument('--resize_shape', type=int, nargs=2, default=[60, 60],
                        help='Heatmap reshape size, e.g., 60 60 or 32 32')
    parser.add_argument('--source_size', type=int, default=None,
                        help='Optional source spectrum length before heatmap')
    parser.add_argument('--target_size', type=int, default=None,
                        help='Optional target spectrum length before heatmap')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    enc_depth = args.encoder_depth if args.encoder_depth is not None else args.depth
    dec_depth = args.decoder_depth if args.decoder_depth is not None else args.depth

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
    print(f"Random seed: {args.seed}")
    print(f"Encoder depth: {enc_depth}, Decoder depth: {dec_depth}")

    class Config:
        in_channels = 1
        hidden_dim = args.hidden_dim
        encoder_depth = enc_depth
        decoder_depth = dec_depth
        num_heads = args.num_heads
        num_modes = 3
        resize_shape = tuple(args.resize_shape)
        patch_size = args.patch_size
        dropout = args.dropout
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        epochs = args.epochs
        batch_size = args.batch_size

    config = Config()
    trainer = Seq2SeqTrainer(config, device)

    data_dir = Path(args.data_dir)
    if args.source_csv:
        source_csv = Path(args.source_csv) if os.path.isabs(args.source_csv) else data_dir / args.source_csv
    else:
        source_csv = data_dir / f'{args.source_mode}_broaden_processed.csv'

    if args.target_csv:
        target_csv = Path(args.target_csv) if os.path.isabs(args.target_csv) else data_dir / args.target_csv
    else:
        target_csv = data_dir / f'{args.target_mode}_broaden_processed.csv'

    if not source_csv.exists() or not target_csv.exists():
        print("Error: Data files not found")
        print(f"Source: {source_csv}")
        print(f"Target: {target_csv}")
        return

    train_loader, val_loader, _ = get_paired_loaders(
        str(source_csv), str(target_csv),
        batch_size=config.batch_size,
        source_size=args.source_size,
        target_size=args.target_size,
        heatmap_size=args.heatmap_size,
        resize_shape=config.resize_shape,
        out_channels=config.in_channels,
        use_h5=True,
        seed=args.seed
    )

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")

    train_modality_pair(
        trainer, train_loader, val_loader,
        args.source_mode, args.target_mode,
        epochs=args.epochs,
        save_dir=args.save_dir,
        seed=args.seed
    )

    print("Seq2seq training completed!")


if __name__ == '__main__':
    main()
