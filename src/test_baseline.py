"""
Baseline evaluation script without using per-sample min/max rescaling.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from model import CrossModalVAE
from utils import get_device
from train_baseline import get_paired_loaders_baseline


def calculate_metrics(original, reconstructed):
    original = original.flatten()
    reconstructed = reconstructed.flatten()

    mse = np.mean((original - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(original, reconstructed)

    try:
        pearson, _ = pearsonr(original, reconstructed)
    except Exception:
        pearson = np.nan

    mae = np.mean(np.abs(original - reconstructed))
    mape = np.mean(np.abs((original - reconstructed) / (original + 1e-8))) * 100

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'pearson': pearson
    }


def plot_comparison(original_list, reconstructed_list, save_dir, prefix,
                   source_mode, target_mode, num_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    num_samples = min(num_samples, len(original_list))

    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        orig = original_list[i]
        recon = reconstructed_list[i]

        x_axis = np.linspace(0, len(orig) - 1, len(orig))
        axes[i].plot(x_axis, orig, label='Original', alpha=0.7, linewidth=2)
        axes[i].plot(x_axis, recon, label='Reconstructed', alpha=0.7,
                     linewidth=2, linestyle='--')
        axes[i].set_xlabel('Wavenumber Index')
        axes[i].set_ylabel('Intensity')
        axes[i].set_title(f'Sample {i+1}: {source_mode} -> {target_mode} (baseline)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(
        save_dir, f'{prefix}_baseline_{source_mode}2{target_mode}_comparison.png'
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved baseline comparison plot to: {save_path}")


def test_modality_pair_baseline(model, test_loader, source_mode, target_mode,
                                device, save_dir='results'):
    model.eval()

    mode_map = {'ir': 0, 'uv': 1, 'raman': 2}
    target_mode_idx = mode_map[target_mode]

    all_metrics = {
        'mse': [], 'rmse': [], 'mae': [], 'mape': [],
        'r2': [], 'pearson': []
    }

    original_spectra = []
    reconstructed_spectra = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            source_batch, target_batch = batch

            source_batch = source_batch.to(device)

            recon, _, _ = model(
                source_batch, target_mode=target_mode_idx, physical_params=None
            )

            if batch_idx == 0:
                print(f"Recon range: [{recon.min().item():.4f}, {recon.max().item():.4f}]")
                print(f"Target range: [{target_batch.min().item():.4f}, {target_batch.max().item():.4f}]")

            recon_np = recon.cpu().numpy()
            target_np = target_batch.numpy()

            for i in range(recon_np.shape[0]):
                recon_spec = recon_np[i, 0].reshape(-1)
                target_spec = target_np[i, 0].reshape(-1)

                metrics = calculate_metrics(target_spec, recon_spec)
                for key, value in metrics.items():
                    if not np.isnan(value):
                        all_metrics[key].append(value)

                if len(original_spectra) < 5:
                    original_spectra.append(target_spec)
                    reconstructed_spectra.append(recon_spec)

    print(f"\n{'='*60}")
    print(f"Baseline Test Results: {source_mode} -> {target_mode}")
    print(f"{'='*60}")
    print(f"MSE:      {np.mean(all_metrics['mse']):.6e} ± {np.std(all_metrics['mse']):.6e}")
    print(f"RMSE:     {np.mean(all_metrics['rmse']):.6e} ± {np.std(all_metrics['rmse']):.6e}")
    print(f"MAE:      {np.mean(all_metrics['mae']):.6e} ± {np.std(all_metrics['mae']):.6e}")
    print(f"MAPE:     {np.mean(all_metrics['mape']):.4f}% ± {np.std(all_metrics['mape']):.4f}%")
    print(f"R²:       {np.mean(all_metrics['r2']):.6f} ± {np.std(all_metrics['r2']):.6f}")
    print(f"Pearson:  {np.mean(all_metrics['pearson']):.6f} ± {np.std(all_metrics['pearson']):.6f}")

    if len(original_spectra) > 0:
        plot_comparison(
            original_spectra, reconstructed_spectra,
            save_dir, 'test', source_mode, target_mode,
            num_samples=min(5, len(original_spectra))
        )

    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Test baseline Cross-Modal VAE without normalization'
    )
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed CSV files')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--source_mode', type=str, required=True,
                        choices=['ir', 'uv', 'raman'], help='Source modality')
    parser.add_argument('--target_mode', type=str, required=True,
                        choices=['ir', 'uv', 'raman'], help='Target modality')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--save_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    args = parser.parse_args()

    device = get_device(args.cpu)
    print(f"Using device: {device}")

    model = CrossModalVAE(
        in_channels=1,
        hidden_channels=args.hidden_channels,
        latent_dim=args.latent_dim,
        image_size=(60, 60),
        num_modes=3,
        use_physical_prior=False,
        physical_dim=7,
        clamp_output=False
    ).to(device)

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f'vae_baseline_{args.source_mode}2{args.target_mode}_best.pt'
    )

    if not os.path.exists(checkpoint_path):
        print(f"Error: Baseline checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded baseline checkpoint from: {checkpoint_path}")

    data_dir = Path(args.data_dir)
    source_file = f'{args.source_mode}_broaden_processed.csv'
    target_file = f'{args.target_mode}_broaden_processed.csv'

    source_csv = data_dir / source_file
    target_csv = data_dir / target_file

    if not source_csv.exists() and not (data_dir / f'{args.source_mode}_broaden_processed.h5').exists():
        print(f"Error: Source data not found: {source_csv}")
        return
    if not target_csv.exists() and not (data_dir / f'{args.target_mode}_broaden_processed.h5').exists():
        print(f"Error: Target data not found: {target_csv}")
        return

    _, _, test_loader = get_paired_loaders_baseline(
        str(source_csv), str(target_csv),
        batch_size=args.batch_size,
        target_size=3600,
        resize_shape=(60, 60),
        out_channels=1,
        use_h5=True
    )

    print(f"Baseline test dataset size: {len(test_loader.dataset)}")

    test_modality_pair_baseline(
        model, test_loader,
        args.source_mode, args.target_mode,
        device, args.save_dir
    )

    print("\nBaseline testing completed!")


if __name__ == '__main__':
    main()


