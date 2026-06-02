"""
Test the Encoder–Decoder (seq2seq) baseline: BiGRU + Bahdanau attention + GRU decoder.
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from model_seq2seq import Seq2SeqSpectrumTranslator
from train import PairedModalDataset, get_paired_loaders
from utils import get_device, inverse_heatmap_to_spectrum, paired_dataset_use_random_split_by_default


def calculate_psnr(original, reconstructed):
    original = original.flatten()
    reconstructed = reconstructed.flatten()
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_val = np.max(original)
    if max_val == 0:
        return np.nan
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim_1d(original, reconstructed):
    original = original.flatten()
    reconstructed = reconstructed.flatten()

    if len(original) != len(reconstructed):
        min_len = min(len(original), len(reconstructed))
        original = original[:min_len]
        reconstructed = reconstructed[:min_len]

    mu1 = np.mean(original)
    mu2 = np.mean(reconstructed)
    sigma1_sq = np.var(original)
    sigma2_sq = np.var(reconstructed)
    sigma12 = np.mean((original - mu1) * (reconstructed - mu2))

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    if denominator == 0:
        return np.nan
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    return numerator / denominator


def calculate_js_divergence(original, reconstructed):
    original = original.flatten()
    reconstructed = reconstructed.flatten()
    original_norm = original - np.min(original) + 1e-10
    reconstructed_norm = reconstructed - np.min(reconstructed) + 1e-10
    original_prob = original_norm / np.sum(original_norm)
    reconstructed_prob = reconstructed_norm / np.sum(reconstructed_norm)
    return jensenshannon(original_prob, reconstructed_prob)


def calculate_metrics(original, reconstructed):
    original = original.flatten()
    reconstructed = reconstructed.flatten()

    mse = np.mean((original - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(original - reconstructed))
    mape = np.mean(np.abs((original - reconstructed) / (original + 1e-8))) * 100
    r2 = r2_score(original, reconstructed)

    try:
        pearson, _ = pearsonr(original, reconstructed)
    except Exception:
        pearson = np.nan

    psnr = calculate_psnr(original, reconstructed)
    ssim = calculate_ssim_1d(original, reconstructed)
    js_div = calculate_js_divergence(original, reconstructed)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'pearson': pearson,
        'psnr': psnr,
        'ssim': ssim,
        'js_div': js_div,
    }


def _resample_1d(y, new_len):
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) == new_len:
        return y
    x_old = np.linspace(0.0, 1.0, len(y))
    x_new = np.linspace(0.0, 1.0, new_len)
    return np.interp(x_new, x_old, y).astype(np.float64)


def dtw_distance_sakoe_chiba(a, b, window_ratio=0.12):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = len(a)
    if n == 0 or len(b) != n:
        return np.nan
    r = max(1, int(window_ratio * n))
    inf = 1e30
    dtw = np.full((n + 1, n + 1), inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        j_min = max(1, i - r)
        j_max = min(n, i + r)
        for j in range(j_min, j_max + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    if dtw[n, n] >= inf * 0.5:
        return np.nan
    return float(np.sqrt(dtw[n, n]))


def compute_dtw_metric(target, pred, subsample=200, window_ratio=0.12):
    target = np.asarray(target, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    L = int(min(subsample, len(target), len(pred)))
    if L < 4:
        return np.nan
    a = _resample_1d(target, L)
    b = _resample_1d(pred, L)
    return dtw_distance_sakoe_chiba(a, b, window_ratio=window_ratio)


def peak_matching_metrics(target, pred, prominence_rel=0.05, distance=5, top_k=30):
    target = np.asarray(target, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    scale = float(np.max(target) - np.min(target) + 1e-8)

    prom = prominence_rel * scale
    peaks_gt, _ = find_peaks(target, prominence=prom, distance=distance)
    peaks_pr, _ = find_peaks(pred, prominence=prom, distance=distance)

    n_gt_raw = len(peaks_gt)
    n_pr_raw = len(peaks_pr)
    count_match = 1.0 if n_gt_raw == n_pr_raw else 0.0

    if n_gt_raw == 0 and n_pr_raw == 0:
        return {
            'peak_count_match': count_match,
            'peak_pos_mae': 0.0,
            'peak_height_mae': 0.0,
        }

    if n_gt_raw == 0 or n_pr_raw == 0:
        return {
            'peak_count_match': count_match,
            'peak_pos_mae': np.nan,
            'peak_height_mae': np.nan,
        }

    h_gt = target[peaks_gt]
    h_pr = pred[peaks_pr]
    if len(peaks_gt) > top_k:
        order = np.argsort(h_gt)[::-1][:top_k]
        peaks_gt = peaks_gt[order]
        h_gt = h_gt[order]
    if len(peaks_pr) > top_k:
        order = np.argsort(h_pr)[::-1][:top_k]
        peaks_pr = peaks_pr[order]
        h_pr = h_pr[order]

    n_gt = len(peaks_gt)
    n_pr = len(peaks_pr)
    len_idx = max(len(target) - 1, 1)
    cost = np.zeros((n_gt, n_pr), dtype=np.float64)
    for i in range(n_gt):
        for j in range(n_pr):
            cost[i, j] = (
                abs(float(peaks_gt[i]) - float(peaks_pr[j])) / len_idx
                + abs(float(h_gt[i]) - float(h_pr[j])) / scale
            )

    row_ind, col_ind = linear_sum_assignment(cost)
    pos_errs = [abs(float(peaks_gt[i]) - float(peaks_pr[j])) for i, j in zip(row_ind, col_ind)]
    h_errs = [abs(float(h_gt[i]) - float(h_pr[j])) for i, j in zip(row_ind, col_ind)]

    return {
        'peak_count_match': count_match,
        'peak_pos_mae': float(np.mean(pos_errs)) if pos_errs else np.nan,
        'peak_height_mae': float(np.mean(h_errs)) if h_errs else np.nan,
    }


def plot_comparison(original_list, reconstructed_list, save_dir, source_mode, target_mode, num_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    num_samples = min(num_samples, len(original_list))

    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        orig = original_list[i]
        recon = reconstructed_list[i]
        x_axis = np.linspace(0, len(orig) - 1, len(orig))

        axes[i].plot(x_axis, orig, label='Original', alpha=0.8, linewidth=2)
        axes[i].plot(x_axis, recon, label='Seq2seq', alpha=0.8, linewidth=2, linestyle='--')
        axes[i].set_xlabel('Wavenumber Index')
        axes[i].set_ylabel('Intensity')
        axes[i].set_title(f'Sample {i + 1}: {source_mode} -> {target_mode} (seq2seq)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(
        save_dir, f'seq2seq_{source_mode}2{target_mode}_comparison.png'
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved seq2seq comparison plot to: {save_path}")


def test_modality_pair(
    model,
    test_loader,
    source_mode,
    target_mode,
    device,
    save_dir='results',
    extra_metrics=True,
    dtw_subsample=200,
    dtw_window_ratio=0.12,
    peak_prominence_rel=0.05,
    peak_distance=5,
    peak_top_k=30,
):
    model.eval()
    mode_map = {'ir': 0, 'uv': 1, 'raman': 2}
    target_mode_idx = mode_map[target_mode]

    all_metrics = {
        'mse': [], 'rmse': [], 'mae': [], 'mape': [],
        'r2': [], 'pearson': [], 'psnr': [], 'ssim': [], 'js_div': []
    }
    if extra_metrics:
        all_metrics['dtw'] = []
        all_metrics['peak_count_match'] = []
        all_metrics['peak_pos_mae'] = []
        all_metrics['peak_height_mae'] = []
    original_spectra = []
    predicted_spectra = []

    with torch.no_grad():
        for batch in test_loader:
            source_batch = batch[0].to(device)
            target_batch = batch[1]
            target_min_batch = batch[4]
            target_max_batch = batch[5]

            pred = model(source_batch, target_mode=target_mode_idx).cpu().numpy()
            target_np = target_batch.numpy()

            for i in range(pred.shape[0]):
                pred_heatmap = pred[i, 0]
                target_heatmap = target_np[i, 0]
                spec_min = float(target_min_batch[i])
                spec_max = float(target_max_batch[i])

                pred_spec = inverse_heatmap_to_spectrum(pred_heatmap, spec_min, spec_max)
                target_spec = inverse_heatmap_to_spectrum(target_heatmap, spec_min, spec_max)

                metrics = calculate_metrics(target_spec, pred_spec)
                if extra_metrics:
                    metrics['dtw'] = compute_dtw_metric(
                        target_spec, pred_spec,
                        subsample=dtw_subsample,
                        window_ratio=dtw_window_ratio,
                    )
                    metrics.update(
                        peak_matching_metrics(
                            target_spec, pred_spec,
                            prominence_rel=peak_prominence_rel,
                            distance=peak_distance,
                            top_k=peak_top_k,
                        )
                    )
                for key in all_metrics:
                    if key not in metrics:
                        continue
                    value = metrics[key]
                    if isinstance(value, (float, np.floating)) and np.isnan(value):
                        continue
                    all_metrics[key].append(value)

                if len(original_spectra) < 5:
                    original_spectra.append(target_spec)
                    predicted_spectra.append(pred_spec)

    print(f"\n{'=' * 60}")
    print(f"Seq2seq Test Results: {source_mode} -> {target_mode}")
    print(f"{'=' * 60}")
    print(f"MSE:      {np.mean(all_metrics['mse']):.6e} ± {np.std(all_metrics['mse']):.6e}")
    print(f"RMSE:     {np.mean(all_metrics['rmse']):.6e} ± {np.std(all_metrics['rmse']):.6e}")
    print(f"MAE:      {np.mean(all_metrics['mae']):.6e} ± {np.std(all_metrics['mae']):.6e}")
    print(f"MAPE:     {np.mean(all_metrics['mape']):.4f}% ± {np.std(all_metrics['mape']):.4f}%")
    print(f"R²:       {np.mean(all_metrics['r2']):.6f} ± {np.std(all_metrics['r2']):.6f}")
    print(f"Pearson:  {np.mean(all_metrics['pearson']):.6f} ± {np.std(all_metrics['pearson']):.6f}")
    if len(all_metrics['psnr']) > 0:
        print(f"PSNR:     {np.mean(all_metrics['psnr']):.6f} ± {np.std(all_metrics['psnr']):.6f}")
    if len(all_metrics['ssim']) > 0:
        print(f"SSIM:     {np.mean(all_metrics['ssim']):.6f} ± {np.std(all_metrics['ssim']):.6f}")
    if len(all_metrics['js_div']) > 0:
        print(f"JS Div:   {np.mean(all_metrics['js_div']):.6f} ± {np.std(all_metrics['js_div']):.6f}")
    if extra_metrics and all_metrics.get('dtw'):
        dtw_vals = np.asarray(all_metrics['dtw'], dtype=np.float64)
        print(f"DTW:      {np.nanmean(dtw_vals):.6f} ± {np.nanstd(dtw_vals):.6f}  "
              f"(subsample<={dtw_subsample}, Sakoe-Chiba w={dtw_window_ratio:.3f})")
    if extra_metrics and all_metrics.get('peak_count_match'):
        pcm = np.asarray(all_metrics['peak_count_match'])
        print(f"Peak cnt acc: {np.mean(pcm):.4f}  "
              f"(1 if raw #peaks equal on target vs pred; prominence_rel={peak_prominence_rel}, dist={peak_distance})")
    if extra_metrics and all_metrics.get('peak_pos_mae'):
        pp = np.asarray(all_metrics['peak_pos_mae'], dtype=np.float64)
        print(f"Peak pos MAE (matched): {np.nanmean(pp):.4f} ± {np.nanstd(pp):.4f}  (index units, top_k={peak_top_k})")
    if extra_metrics and all_metrics.get('peak_height_mae'):
        ph = np.asarray(all_metrics['peak_height_mae'], dtype=np.float64)
        print(f"Peak hgt MAE (matched): {np.nanmean(ph):.6e} ± {np.nanstd(ph):.6e}")

    if original_spectra:
        plot_comparison(original_spectra, predicted_spectra, save_dir, source_mode, target_mode)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='Test Encoder–Decoder (seq2seq) baseline')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed CSV files')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--source_mode', type=str, required=True, choices=['ir', 'uv', 'raman'],
                        help='Source modality')
    parser.add_argument('--target_mode', type=str, required=True, choices=['ir', 'uv', 'raman'],
                        help='Target modality')
    parser.add_argument('--source_csv', type=str, default=None,
                        help='Custom source CSV filename (optional, absolute or relative to data_dir)')
    parser.add_argument('--target_csv', type=str, default=None,
                        help='Custom target CSV filename (optional, absolute or relative to data_dir)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--depth', type=int, default=6,
                        help='Default encoder/decoder depth when --encoder_depth/--decoder_depth omitted')
    parser.add_argument('--encoder_depth', type=int, default=None,
                        help='Encoder layers (default: same as --depth)')
    parser.add_argument('--decoder_depth', type=int, default=None,
                        help='Kept for API compat; unused for GRU+attention decoder')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
    parser.add_argument('--save_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--heatmap_size', type=int, default=3600,
                        help='Heatmap size (must be a perfect square)')
    parser.add_argument('--resize_shape', type=int, nargs=2, default=[60, 60],
                        help='Heatmap reshape size')
    parser.add_argument('--source_size', type=int, default=None,
                        help='Optional source spectrum length before heatmap')
    parser.add_argument('--target_size', type=int, default=None,
                        help='Optional target spectrum length before heatmap')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_split', action='store_true',
                        help='Use the full provided dataset as test set')
    parser.add_argument('--no_extra_metrics', action='store_true',
                        help='Disable DTW / peak-based extra metrics')
    parser.add_argument('--dtw_subsample', type=int, default=200,
                        help='Resample spectra to at most this length before DTW')
    parser.add_argument('--dtw_window_ratio', type=float, default=0.12,
                        help='Sakoe-Chiba band width ratio for DTW')
    parser.add_argument('--peak_prominence_rel', type=float, default=0.05,
                        help='Relative prominence threshold for find_peaks, relative to target dynamic range')
    parser.add_argument('--peak_distance', type=int, default=5,
                        help='Minimum distance between detected peaks')
    parser.add_argument('--peak_top_k', type=int, default=30,
                        help='Match at most top-K peaks by intensity for peak metrics')
    args = parser.parse_args()

    enc_depth = args.encoder_depth if args.encoder_depth is not None else args.depth
    dec_depth = args.decoder_depth if args.decoder_depth is not None else args.depth

    device = get_device(args.cpu)
    print(f"Using device: {device}")

    model = Seq2SeqSpectrumTranslator(
        in_channels=1,
        image_size=tuple(args.resize_shape),
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        encoder_depth=enc_depth,
        decoder_depth=dec_depth,
        num_heads=args.num_heads,
        num_modes=3,
        clamp_output=True,
    ).to(device)

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f'seq2seq_{args.source_mode}2{args.target_mode}_best_seed{args.seed}.pt'
    )
    if not os.path.exists(checkpoint_path):
        print(f"Error: seq2seq checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        print(f"[warning] Unexpected checkpoint keys ignored: {unexpected_keys}")
    if missing_keys:
        print(f"[warning] Missing model keys when loading checkpoint: {missing_keys}")
    print(f"Loaded seq2seq checkpoint from: {checkpoint_path}")

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

    if paired_dataset_use_random_split_by_default(
        args.data_dir, args.source_csv, args.target_csv
    ):
        use_full_as_test = args.no_split
    else:
        use_full_as_test = args.no_split or (args.source_csv is not None) or (args.target_csv is not None)

    if use_full_as_test:
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
            str(source_csv), str(target_csv),
            batch_size=args.batch_size,
            source_size=args.source_size,
            target_size=args.target_size,
            heatmap_size=args.heatmap_size,
            resize_shape=tuple(args.resize_shape),
            out_channels=1,
            seed=args.seed
        )
        print("Data split: using random_split(test subset) to match training split.")

    print(f"Test dataset size: {len(test_loader.dataset)}")
    test_modality_pair(
        model, test_loader,
        args.source_mode, args.target_mode,
        device, args.save_dir,
        extra_metrics=not args.no_extra_metrics,
        dtw_subsample=args.dtw_subsample,
        dtw_window_ratio=args.dtw_window_ratio,
        peak_prominence_rel=args.peak_prominence_rel,
        peak_distance=args.peak_distance,
        peak_top_k=args.peak_top_k,
    )
    print("\nSeq2seq testing completed!")


if __name__ == '__main__':
    main()
