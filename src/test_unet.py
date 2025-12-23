"""
Test script for UNet translator.
Evaluates saved checkpoints on paired modality data and saves comparison plots.
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr

from train_unet import UNetTranslator
from train import get_paired_loaders, PairedModalDataset
from torch.utils.data import DataLoader
from utils import inverse_heatmap_to_spectrum, get_device


def calculate_metrics(original, reconstructed):
    """Compute basic regression metrics on 1D spectra."""
    original = original.flatten()
    reconstructed = reconstructed.flatten()

    mse = np.mean((original - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(original - reconstructed))
    mape = np.mean(np.abs((original - reconstructed) / (original + 1e-8))) * 100

    # R^2
    ss_res = np.sum((original - reconstructed) ** 2)
    ss_tot = np.sum((original - np.mean(original)) ** 2) + 1e-8
    r2 = 1 - ss_res / ss_tot

    # Pearson correlation coefficient
    try:
        pearson, _ = pearsonr(original, reconstructed)
    except:
        pearson = np.nan

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "pearson": pearson,
    }


def plot_comparison(original_list, reconstructed_list, save_dir, prefix, num_samples=6):
    """Plot original vs reconstructed spectra."""
    os.makedirs(save_dir, exist_ok=True)
    num_samples = min(num_samples, len(original_list))

    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        orig = original_list[i]
        recon = reconstructed_list[i]
        x_axis = np.arange(len(orig))
        axes[i].plot(x_axis, orig, label="Target", alpha=0.8, linewidth=2)
        axes[i].plot(x_axis, recon, label="UNet Pred", alpha=0.8, linewidth=2, linestyle="--")
        axes[i].set_xlabel("Index")
        axes[i].set_ylabel("Intensity")
        axes[i].legend()
        axes[i].grid(alpha=0.3)
        axes[i].set_title(f"Sample {i+1}")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to: {save_path}")


def load_model(checkpoint_path, device, in_channels=1, hidden_channels=128, num_modes=3):
    """Load UNetTranslator from checkpoint."""
    model = UNetTranslator(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_modes=num_modes,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def evaluate(model, test_loader, target_mode_idx, device):
    """Run evaluation on the test set."""
    all_metrics = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            source_heatmap = batch[0].to(device)
            target_heatmap = batch[1]
            target_min = batch[4].numpy()
            target_max = batch[5].numpy()
            target_spec = batch[7].numpy()  # original-length target spectrum

            pred_heatmap = model(source_heatmap, target_mode_idx).cpu().numpy()
            target_heatmap_np = target_heatmap.numpy()

            batch_size = pred_heatmap.shape[0]
            for i in range(batch_size):
                pred_spec = inverse_heatmap_to_spectrum(
                    pred_heatmap[i, 0],
                    target_min[i],
                    target_max[i],
                )
                target_spec_i = target_spec[i]

                # If lengths mismatch (should rarely happen), interpolate to target length
                if len(pred_spec) != len(target_spec_i):
                    x_old = np.linspace(0, 1, len(pred_spec))
                    x_new = np.linspace(0, 1, len(target_spec_i))
                    pred_spec = np.interp(x_new, x_old, pred_spec)

                metrics = calculate_metrics(target_spec_i, pred_spec)
                all_metrics.append(metrics)
                all_targets.append(target_spec_i)
                all_preds.append(pred_spec)

    # Aggregate metrics
    if not all_metrics:
        return {}, [], []

    keys = all_metrics[0].keys()
    mean_metrics = {}
    for k in keys:
        values = [m[k] for m in all_metrics]
        # Use nanmean for metrics that might contain NaN (e.g., pearson)
        mean_metrics[k] = float(np.nanmean(values))
    return mean_metrics, all_targets, all_preds


def main():
    parser = argparse.ArgumentParser(description="Test UNet translator")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory containing UNet checkpoints")
    parser.add_argument("--source_mode", type=str, required=True, choices=["ir", "uv", "raman"], help="Source modality")
    parser.add_argument("--target_mode", type=str, required=True, choices=["ir", "uv", "raman"], help="Target modality")
    parser.add_argument("--source_csv", type=str, default=None, help="Custom source CSV filename (optional, overrides default naming)")
    parser.add_argument("--target_csv", type=str, default=None, help="Custom target CSV filename (optional, overrides default naming)")
    parser.add_argument("--checkpoint_name", type=str, default=None, help="Checkpoint filename (optional)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing")
    parser.add_argument("--hidden_channels", type=int, default=128, help="Hidden channels (must match training)")
    parser.add_argument("--heatmap_size", type=int, default=3600,
                       help="Heatmap size (must be a perfect square, e.g., 3600=60x60, 1024=32x32)")
    parser.add_argument("--resize_shape", type=int, nargs=2, default=[60, 60],
                       help="Heatmap reshape size, e.g., 60 60 or 32 32")
    parser.add_argument("--source_size", type=int, default=None, help="Optional source spectrum length before heatmap")
    parser.add_argument("--target_size", type=int, default=None, help="Optional target spectrum length before heatmap")
    parser.add_argument("--no_split", action="store_true",
                       help="Do not split dataset; treat the whole provided CSV/H5 as the test set. "
                            "Note: For qm9s dataset, split is always used unless this flag is set. "
                            "For other datasets, if --source_csv/--target_csv is provided, full dataset is used by default.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save plots")
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
        source_csv = data_dir / f"{args.source_mode}_broaden_processed.csv"
    
    if args.target_csv:
        target_csv = Path(args.target_csv) if os.path.isabs(args.target_csv) else data_dir / args.target_csv
    else:
        target_csv = data_dir / f"{args.target_mode}_broaden_processed.csv"

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
            source_size=args.source_size,
            target_size=args.target_size,
            heatmap_size=args.heatmap_size,
            resize_shape=tuple(args.resize_shape),
            out_channels=1,
            use_h5=True
        )
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        print("Data split: using FULL dataset as test (no random_split).")
    else:
        _, _, test_loader = get_paired_loaders(
            str(source_csv),
            str(target_csv),
            batch_size=args.batch_size,
            source_size=args.source_size,
            target_size=args.target_size,
            heatmap_size=args.heatmap_size,
            resize_shape=tuple(args.resize_shape),
            out_channels=1,
            use_h5=True,
            seed=args.seed,  # Must match training seed for consistent data split
        )
        print(f"Data split: using random_split(test subset) to match training split (seed={args.seed}).")
    
    print(f"Test dataset size: {len(test_loader.dataset)}")

    # Resolve checkpoint path
    if args.checkpoint_name is not None:
        ckpt_path = Path(args.checkpoint_dir) / args.checkpoint_name
    else:
        ckpt_path = Path(args.checkpoint_dir) / f"unet_{args.source_mode}2{args.target_mode}_best_seed{args.seed}.pt"

    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        return

    model = load_model(
        ckpt_path,
        device,
        in_channels=1,
        hidden_channels=args.hidden_channels,
        num_modes=3,
    )

    target_mode_idx = {"ir": 0, "uv": 1, "raman": 2}[args.target_mode]
    metrics, targets, preds = evaluate(model, test_loader, target_mode_idx, device)

    if metrics:
        print("Test metrics:")
        for k, v in metrics.items():
            if np.isnan(v):
                print(f"  {k}: NaN")
            else:
                print(f"  {k}: {v:.6f}")

    # Save comparison plot for a few samples
    if targets and preds:
        plot_comparison(
            targets,
            preds,
            save_dir=args.results_dir,
            prefix=f"unet_{args.source_mode}2{args.target_mode}",
            num_samples=min(6, len(targets)),
        )

    print("Testing completed.")


if __name__ == "__main__":
    main()

