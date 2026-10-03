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
from scipy.spatial.distance import jensenshannon

from model import CrossModalVAE
from train_spectrogen import get_paired_loaders_phys, batch_compute_params
from train import PairedModalDataset
from torch.utils.data import DataLoader
from utils import (
    inverse_heatmap_to_spectrum,
    get_device,
    reconstruction_loss,
    paired_dataset_use_random_split_by_default,
)


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


def calculate_psnr(original, reconstructed):
    """计算PSNR (Peak Signal-to-Noise Ratio)"""
    original = original.flatten()
    reconstructed = reconstructed.flatten()
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_val = np.max(original)
    if max_val == 0:
        return np.nan
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr


def calculate_ssim_1d(original, reconstructed):
    """计算1D数据的SSIM (Structural Similarity Index)"""
    original = original.flatten()
    reconstructed = reconstructed.flatten()

    # 确保长度一致
    if len(original) != len(reconstructed):
        min_len = min(len(original), len(reconstructed))
        original = original[:min_len]
        reconstructed = reconstructed[:min_len]

    # 计算均值和方差
    mu1 = np.mean(original)
    mu2 = np.mean(reconstructed)
    sigma1_sq = np.var(original)
    sigma2_sq = np.var(reconstructed)
    sigma12 = np.mean((original - mu1) * (reconstructed - mu2))

    # SSIM参数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 计算SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    if denominator == 0:
        return np.nan

    ssim = numerator / denominator
    return ssim


def calculate_js_divergence(original, reconstructed):
    """计算Jensen-Shannon Divergence"""
    original = original.flatten()
    reconstructed = reconstructed.flatten()

    # 归一化为概率分布（确保非负）
    original_norm = original - np.min(original) + 1e-10
    reconstructed_norm = reconstructed - np.min(reconstructed) + 1e-10

    original_prob = original_norm / np.sum(original_norm)
    reconstructed_prob = reconstructed_norm / np.sum(reconstructed_norm)

    # 计算JS Divergence
    js_div = jensenshannon(original_prob, reconstructed_prob)
    return js_div


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

    # PSNR
    psnr = calculate_psnr(original, reconstructed)

    # SSIM
    ssim = calculate_ssim_1d(original, reconstructed)

    # JS Divergence
    js_div = calculate_js_divergence(original, reconstructed)

    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "pearson": pearson,
            "psnr": psnr, "ssim": ssim, "js_div": js_div}


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
    parser.add_argument('--dump_index', type=int, nargs='+', default=None,
                       help='List of sample indices (within this test run) to dump target/pred spectra to CSV')
    parser.add_argument('--no_split', action='store_true',
                       help='Do not split dataset; treat the whole provided CSV/H5 as the test set. '
                            'Note: For QM9S / QMe14S, the same random_split test subset as training is used unless this flag is set. '
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

    if paired_dataset_use_random_split_by_default(
        args.data_dir, args.source_csv, args.target_csv
    ):
        use_full_as_test = args.no_split
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
    all_sources = []
    sample_indices = []

    with torch.no_grad():
        for batch in test_loader:
            source_batch = batch[0].to(device)
            target_batch = batch[1]
            tgt_min = batch[4].numpy()
            tgt_max = batch[5].numpy()
            tgt_orig = batch[7].numpy()
            src_orig = batch[6].numpy()

            phys_src = batch_compute_params(batch[6])
            phys_tensor = torch.tensor([params_to_vec(p) for p in phys_src], dtype=torch.float32, device=device)

            recon, mu, logvar = model(source_batch, target_mode=target_mode_idx, physical_params=phys_tensor)
            recon = recon.cpu().numpy()

            # recon loss for logging
            recon_loss = reconstruction_loss(torch.tensor(recon), target_batch).item()

            bs = recon.shape[0]
            base_idx = len(all_targets)  # running global index within this test run
            for i in range(bs):
                pred_spec = inverse_heatmap_to_spectrum(recon[i, 0], tgt_min[i], tgt_max[i])
                tgt_spec = tgt_orig[i]
                src_spec = src_orig[i]
                if len(pred_spec) != len(tgt_spec):
                    x_old = np.linspace(0, 1, len(pred_spec))
                    x_new = np.linspace(0, 1, len(tgt_spec))
                    pred_spec = np.interp(x_new, x_old, pred_spec)
                if len(src_spec) != len(tgt_spec):
                    x_old = np.linspace(0, 1, len(src_spec))
                    x_new = np.linspace(0, 1, len(tgt_spec))
                    src_spec = np.interp(x_new, x_old, src_spec)
                metrics = calculate_metrics(tgt_spec, pred_spec)
                metrics["recon_loss"] = recon_loss
                all_metrics.append(metrics)
                all_targets.append(tgt_spec)
                all_preds.append(pred_spec)
                all_sources.append(src_spec)
                sample_indices.append(base_idx + i)

    if all_metrics:
        keys = all_metrics[0].keys()
        # Filter out NaN values when calculating mean
        mean_metrics = {}
        std_metrics = {}
        for k in keys:
            values = [m[k] for m in all_metrics if not np.isnan(m[k])]
            if values:
                mean_metrics[k] = float(np.mean(values))
                std_metrics[k] = float(np.std(values))
            else:
                mean_metrics[k] = np.nan
                std_metrics[k] = np.nan

        print("\n" + "=" * 60)
        print(f"Test Results (SpectroGen): {args.source_mode} -> {args.target_mode}")
        print("=" * 60)
        # Print metrics in a specific order
        metric_order = ['mse', 'rmse', 'mae', 'mape', 'r2', 'pearson', 'psnr', 'ssim', 'js_div', 'recon_loss']
        for k in metric_order:
            if k in mean_metrics and not np.isnan(mean_metrics[k]):
                if k == 'mape':
                    print(f"{k.upper():8s}: {mean_metrics[k]:.4f}% ± {std_metrics[k]:.4f}%")
                elif k == 'recon_loss':
                    print(f"{k.upper():8s}: {mean_metrics[k]:.6e} ± {std_metrics[k]:.6e}")
                else:
                    print(f"{k.upper():8s}: {mean_metrics[k]:.6f} ± {std_metrics[k]:.6f}")
        # Print any other metrics not in the order list
        for k in sorted(keys):
            if k not in metric_order and not np.isnan(mean_metrics[k]):
                print(f"{k.upper():8s}: {mean_metrics[k]:.6f} ± {std_metrics[k]:.6f}")

    if sample_indices and all_targets and all_preds:
        # 显示保存对比图时对应的样本索引（当前测试运行中的顺序索引）
        num_show = min(6, len(sample_indices))
        print(f"\nSample indices used for comparison plots (within this test run): {sample_indices[:num_show]}")

    # 将预测和目标按索引保存，便于与原始CSV行号对照
    if sample_indices and all_targets and all_preds:
        os.makedirs(args.results_dir, exist_ok=True)
        preds_arr = np.vstack(all_preds)
        targets_arr = np.vstack(all_targets)
        idx_arr = np.array(sample_indices, dtype=int)

        preds_arr = np.vstack(all_preds)
        targets_arr = np.vstack(all_targets)
        sources_arr = np.vstack(all_sources)

        def _save_matrix(path, idx, matrix):
            with open(path, "w", encoding="utf-8") as f:
                header = "index," + ",".join([f"v{i}" for i in range(matrix.shape[1])])
                f.write(header + "\n")
                for i_row, row in zip(idx, matrix):
                    f.write(f"{i_row}," + ",".join(f"{v:.6e}" for v in row) + "\n")

        preds_path = Path(args.results_dir) / f"spectrogen_{args.source_mode}2{args.target_mode}_preds.csv"
        targets_path = Path(args.results_dir) / f"spectrogen_{args.source_mode}2{args.target_mode}_targets.csv"
        sources_path = Path(args.results_dir) / f"spectrogen_{args.source_mode}2{args.target_mode}_sources.csv"
        _save_matrix(preds_path, idx_arr, preds_arr)
        _save_matrix(targets_path, idx_arr, targets_arr)
        _save_matrix(sources_path, idx_arr, sources_arr)
        print(f"\nSaved predictions to: {preds_path}")
        print(f"Saved targets to:      {targets_path}")
        print(f"Saved sources to:      {sources_path}")

        # 单文件汇总：index, point, target, pred, source
        combined_path = Path(args.results_dir) / f"spectrogen_{args.source_mode}2{args.target_mode}_combined.csv"
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write("index,point,target,pred,source\n")
            for idx_val, tgt_row, pred_row, src_row in zip(idx_arr, targets_arr, preds_arr, sources_arr):
                for j, (tv, pv, sv) in enumerate(zip(tgt_row, pred_row, src_row)):
                    f.write(f"{idx_val},{j},{tv:.6e},{pv:.6e},{sv:.6e}\n")
        print(f"Saved combined target/pred/source to: {combined_path}")

        # 保存每个样本的R2列表（顺序与index一致）
        r2_list = np.array([m['r2'] for m in all_metrics], dtype=float)
        r2_path = Path(args.results_dir) / f"spectrogen_{args.source_mode}2{args.target_mode}_r2_per_sample.csv"
        with open(r2_path, "w", encoding="utf-8") as f:
            f.write("index,r2\n")
            for idx_val, r2_val in zip(idx_arr, r2_list):
                f.write(f"{idx_val},{r2_val:.6f}\n")
        print(f"Saved per-sample R2 to: {r2_path}")

        # 保存per-sample PSNR, SSIM, JS Divergence
        psnr_list = np.array([m.get('psnr', np.nan) for m in all_metrics], dtype=float)
        ssim_list = np.array([m.get('ssim', np.nan) for m in all_metrics], dtype=float)
        js_div_list = np.array([m.get('js_div', np.nan) for m in all_metrics], dtype=float)
        metrics_path = Path(args.results_dir) / f"spectrogen_{args.source_mode}2{args.target_mode}_metrics_per_sample.csv"
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("index,psnr,ssim,js_div\n")
            for idx_val, psnr_val, ssim_val, js_val in zip(idx_arr, psnr_list, ssim_list, js_div_list):
                f.write(f"{idx_val},{psnr_val:.6f},{ssim_val:.6f},{js_val:.6f}\n")
        print(f"Saved per-sample metrics (PSNR, SSIM, JS Div) to: {metrics_path}")

        # 如果用户指定 dump_index，则单独导出对应样本的 target/pred 光谱
        if args.dump_index:
            for want_idx in args.dump_index:
                if want_idx in idx_arr:
                    pos = np.where(idx_arr == want_idx)[0][0]
                    tgt_row = targets_arr[pos]
                    pred_row = preds_arr[pos]
                    src_row = sources_arr[pos]
                    out_path = Path(args.results_dir) / f"spectrogen_{args.source_mode}2{args.target_mode}_idx{want_idx}.csv"
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write("target,pred,source\n")
                        for tv, pv, sv in zip(tgt_row, pred_row, src_row):
                            f.write(f"{tv:.6e},{pv:.6e},{sv:.6e}\n")
                    print(f"Dumped target/pred for index {want_idx} -> {out_path}")
                else:
                    print(f"Warning: requested index {want_idx} not found in this test run.")

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
