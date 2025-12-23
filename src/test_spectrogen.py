"""
Testing script for SpectroGen (VAE with physical prior).
Loads a SpectroGen checkpoint, runs on test split, and reports metrics plus comparison plots.
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr

from model import CrossModalVAE
from train_spectrogen import get_paired_loaders_phys, batch_compute_params
from train import PairedModalDataset
from torch.utils.data import DataLoader
from utils import inverse_heatmap_to_spectrum, get_device, reconstruction_loss


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


def calculate_metrics(original, reconstructed):
    original = original.flatten()
    reconstructed = reconstructed.flatten()
    mse = np.mean((original - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(original - reconstructed))
    mape = np.mean(np.abs((original - reconstructed) / (original + 1e-8))) * 100
    ss_res = np.sum((original - reconstructed) ** 2)
    ss_tot = np.sum((original - np.mean(original)) ** 2) + 1e-8
    r2 = 1 - ss_res / ss_tot
    # Pearson相关系数
    try:
        pearson, _ = pearsonr(original, reconstructed)
    except:
        pearson = np.nan
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "pearson": pearson}


def plot_comparison(targets, preds, save_dir, prefix, num_samples=6):
    os.makedirs(save_dir, exist_ok=True)
    n = min(num_samples, len(targets))
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
    if n == 1:
        axes = [axes]
    for i in range(n):
        x_axis = np.arange(len(targets[i]))
        axes[i].plot(x_axis, targets[i], label="Target", linewidth=2)
        axes[i].plot(x_axis, preds[i], label="Prediction", linewidth=2, linestyle="--")
        axes[i].set_title(f"Sample {i+1}")
        axes[i].set_xlabel("Index")
        axes[i].set_ylabel("Intensity")
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{prefix}_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to: {path}")


def load_model(ckpt_path, device, in_channels=1, hidden_channels=128, latent_dim=128, num_modes=3, image_size=(60, 60)):
    model = CrossModalVAE(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        latent_dim=latent_dim,
        image_size=image_size,
        num_modes=num_modes,
        use_physical_prior=True,
        physical_dim=7,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Test SpectroGen (VAE with physical prior)")
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default=None)
    parser.add_argument('--source_mode', type=str, required=True, choices=['ir', 'uv', 'raman'])
    parser.add_argument('--target_mode', type=str, required=True, choices=['ir', 'uv', 'raman'])
    parser.add_argument('--source_csv', type=str, default=None,
                       help='Custom source CSV filename (optional, overrides default naming)')
    parser.add_argument('--target_csv', type=str, default=None,
                       help='Custom target CSV filename (optional, overrides default naming)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--heatmap_size', type=int, default=3600,
                       help='Heatmap size (must be a perfect square, e.g., 3600=60x60, 1024=32x32)')
    parser.add_argument('--resize_shape', type=int, nargs=2, default=[60, 60],
                       help='Heatmap reshape size, e.g., 60 60 or 32 32')
    parser.add_argument('--no_split', action='store_true',
                       help='Do not split dataset; treat the whole provided CSV/H5 as the test set. '
                            'Note: For qm9s dataset, split is always used unless this flag is set. '
                            'For other datasets, if --source_csv/--target_csv is provided, full dataset is used by default.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = get_device(args.cpu)
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    # Use custom filenames if provided, otherwise use default naming
    if args.source_csv:
        source_csv = Path(args.source_csv) if os.path.isabs(args.source_csv) else data_dir / args.source_csv
    else:
        source_csv = data_dir / f'{args.source_mode}_broaden_processed.csv'
    
    if args.target_csv:
        target_csv = Path(args.target_csv) if os.path.isabs(args.target_csv) else data_dir / args.target_csv
    else:
        target_csv = data_dir / f'{args.target_mode}_broaden_processed.csv'
    if not source_csv.exists() or not target_csv.exists():
        print("Error: data files not found")
        print(f"  {source_csv}")
        print(f"  {target_csv}")
        return

    # 创建测试数据加载器
    # For qm9s dataset, always use split (even if source_csv/target_csv provided)
    # For other datasets, use full dataset if source_csv/target_csv provided (unless --no_split is explicitly set)
    is_qm9s = False
    if 'qm9s' in str(args.data_dir).lower():
        is_qm9s = True
    elif args.source_csv and 'qm9' in str(args.source_csv).lower():
        is_qm9s = True
    elif args.target_csv and 'qm9' in str(args.target_csv).lower():
        is_qm9s = True
    
    # For qm9s, always split unless --no_split is explicitly set
    # For others, use full dataset if CSV files provided (unless --no_split is explicitly set)
    if is_qm9s:
        use_full_as_test = args.no_split  # qm9s: only use full if explicitly --no_split
    else:
        use_full_as_test = args.no_split or (args.source_csv is not None) or (args.target_csv is not None)
    
    if use_full_as_test:
        # When user provides explicit CSVs (often already "test split"), do NOT random-split again.
        dataset = PairedModalDataset(
            str(source_csv), str(target_csv),
            source_size=None,
            target_size=None,
            heatmap_size=args.heatmap_size,
            resize_shape=tuple(args.resize_shape),
            out_channels=1,
            use_h5=True
        )
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        print("Data split: using FULL dataset as test (no random_split).")
    else:
        _, _, test_loader = get_paired_loaders_phys(
            str(source_csv),
            str(target_csv),
            batch_size=args.batch_size,
            heatmap_size=args.heatmap_size,
            resize_shape=tuple(args.resize_shape),
            out_channels=1,
            seed=args.seed,
        )
        print(f"Data split: using random_split(test subset) to match training split (seed={args.seed}).")
    
    print(f"Test dataset size: {len(test_loader.dataset)}")

    if args.checkpoint_name:
        ckpt_path = Path(args.checkpoint_dir) / args.checkpoint_name
    else:
        ckpt_path = Path(args.checkpoint_dir) / f'spectrogen_{args.source_mode}2{args.target_mode}_best_seed{args.seed}.pt'
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        return

    model = load_model(
        ckpt_path,
        device,
        in_channels=1,
        hidden_channels=args.hidden_channels,
        latent_dim=args.latent_dim,
        num_modes=3,
        image_size=tuple(args.resize_shape),
    )
    target_mode_idx = {'ir': 0, 'uv': 1, 'raman': 2}[args.target_mode]

    all_metrics = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            source_batch = batch[0].to(device)
            target_batch = batch[1]
            tgt_min = batch[4].numpy()
            tgt_max = batch[5].numpy()
            tgt_orig = batch[7].numpy()

            phys_src = batch_compute_params(batch[6])
            phys_tensor = torch.tensor([params_to_vec(p) for p in phys_src], dtype=torch.float32, device=device)

            recon, mu, logvar = model(source_batch, target_mode=target_mode_idx, physical_params=phys_tensor)
            recon = recon.cpu().numpy()

            # recon loss for logging
            recon_loss = reconstruction_loss(torch.tensor(recon), target_batch).item()

            bs = recon.shape[0]
            for i in range(bs):
                pred_spec = inverse_heatmap_to_spectrum(recon[i, 0], tgt_min[i], tgt_max[i])
                tgt_spec = tgt_orig[i]
                if len(pred_spec) != len(tgt_spec):
                    x_old = np.linspace(0, 1, len(pred_spec))
                    x_new = np.linspace(0, 1, len(tgt_spec))
                    pred_spec = np.interp(x_new, x_old, pred_spec)
                metrics = calculate_metrics(tgt_spec, pred_spec)
                metrics["recon_loss"] = recon_loss
                all_metrics.append(metrics)
                all_targets.append(tgt_spec)
                all_preds.append(pred_spec)

    if all_metrics:
        keys = all_metrics[0].keys()
        mean_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
        print("Test metrics (mean over test set):")
        for k, v in mean_metrics.items():
            print(f"  {k}: {v:.6f}")

    if all_targets and all_preds:
        plot_comparison(
            all_targets,
            all_preds,
            save_dir=args.results_dir,
            prefix=f'spectrogen_{args.source_mode}2{args.target_mode}',
            num_samples=min(6, len(all_targets)),
        )

    print("SpectroGen testing completed.")


if __name__ == '__main__':
    main()

