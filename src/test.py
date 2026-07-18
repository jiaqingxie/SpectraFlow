"""
测试跨模态VAE模型在不同模态转换上的性能
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon

from model import CrossModalVAE
from utils import (
    inverse_heatmap_to_spectrum,
    get_device, load_checkpoint,
    paired_dataset_use_random_split_by_default,
)
from train import PairedModalDataset, get_paired_loaders


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
    """
    计算评估指标
    
    Returns:
        dict: 包含MSE, R2, Pearson相关系数等指标
    """
    original = original.flatten()
    reconstructed = reconstructed.flatten()
    
    # MSE
    mse = np.mean((original - reconstructed) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # R²
    r2 = r2_score(original, reconstructed)
    
    # Pearson相关系数
    try:
        pearson, _ = pearsonr(original, reconstructed)
    except:
        pearson = np.nan
    
    # MAE
    mae = np.mean(np.abs(original - reconstructed))
    
    # 相对误差百分比
    mape = np.mean(np.abs((original - reconstructed) / (original + 1e-8))) * 100
    
    # PSNR
    psnr = calculate_psnr(original, reconstructed)
    
    # SSIM
    ssim = calculate_ssim_1d(original, reconstructed)
    
    # JS Divergence
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
        'js_div': js_div
    }


def has_invalid_core_metrics(metrics):
    """如果任一核心指标为 NaN，则整笔样本不纳入 summary。"""
    core_metric_keys = (
        'mse', 'rmse', 'mae', 'mape',
        'r2', 'pearson', 'psnr', 'ssim', 'js_div',
    )
    for key in core_metric_keys:
        value = metrics.get(key)
        if value is None or np.isnan(value):
            return True
    return False


def plot_comparison(original_list, reconstructed_list, save_dir, prefix, 
                   source_mode, target_mode, num_samples=5):
    """
    绘制原始和重建光谱的对比图
    
    Args:
        original_list: 原始光谱列表
        reconstructed_list: 重建光谱列表
        save_dir: 保存目录
        prefix: 文件名前缀
        source_mode: 源模态
        target_mode: 目标模态
        num_samples: 绘制的样本数量
    """
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
        axes[i].set_title(f'Sample {i+1}: {source_mode} -> {target_mode}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{prefix}_{source_mode}2{target_mode}_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to: {save_path}")


def test_modality_pair(model, test_loader, source_mode, target_mode, device, 
                      save_dir='results', dump_index=None):
    """
    测试单个模态对的转换性能
    
    Args:
        model: 训练好的VAE模型
        test_loader: 测试数据加载器
        source_mode: 源模态
        target_mode: 目标模态
        device: 计算设备
        save_dir: 结果保存目录
    """
    model.eval()
    
    mode_map = {'ir': 0, 'uv': 1, 'raman': 2}
    target_mode_idx = mode_map[target_mode]
    
    all_metrics = {
        'mse': [], 'rmse': [], 'mae': [], 'mape': [],
        'r2': [], 'pearson': [], 'psnr': [], 'ssim': [], 'js_div': []
    }
    
    original_spectra = []
    reconstructed_spectra = []
    all_targets = []
    all_preds = []
    all_sources = []
    sample_indices = []
    total_samples = 0
    skipped_invalid_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            source_batch, target_batch = batch[0], batch[1]
            source_min_batch = batch[2]
            source_max_batch = batch[3]
            target_min_batch = batch[4]
            target_max_batch = batch[5]
            physical_params = batch[8] if len(batch) > 8 else None
            
            source_batch = source_batch.to(device)
            physical_params = physical_params.to(device) if physical_params is not None else None
            
            # 前向传播
            recon, _, _ = model(
                source_batch,
                target_mode=target_mode_idx,
                physical_params=physical_params,
                sample_latent=False,
            )
            
            # 调试：检查输出范围
            if batch_idx == 0:
                print(f"Recon range: [{recon.min().item():.4f}, {recon.max().item():.4f}]")
                print(f"Target range: [{target_batch.min().item():.4f}, {target_batch.max().item():.4f}]")
            
            # 转换为numpy
            recon_np = recon.cpu().numpy()
            target_np = target_batch.numpy()
            
            # 将热图转换回光谱
            base_idx = total_samples
            for i in range(recon_np.shape[0]):
                total_samples += 1
                recon_heatmap = recon_np[i, 0]  # (H, W)
                target_heatmap = target_np[i, 0]
                source_heatmap = source_batch[i, 0].cpu().numpy()
                
                spec_min = float(target_min_batch[i])
                spec_max = float(target_max_batch[i])
                src_min = float(source_min_batch[i])
                src_max = float(source_max_batch[i])
                
                # 转换回1D光谱
                recon_spec = inverse_heatmap_to_spectrum(recon_heatmap, spec_min, spec_max)
                target_spec = inverse_heatmap_to_spectrum(target_heatmap, spec_min, spec_max)
                source_spec = inverse_heatmap_to_spectrum(source_heatmap, src_min, src_max)
                
                # 计算指标
                metrics = calculate_metrics(target_spec, recon_spec)
                if has_invalid_core_metrics(metrics):
                    skipped_invalid_samples += 1
                    continue
                for key in all_metrics:
                    all_metrics[key].append(metrics[key])
                all_targets.append(target_spec)
                all_preds.append(recon_spec)
                all_sources.append(source_spec)
                sample_indices.append(base_idx + i)
                
                # 保存前几个样本用于可视化
                if len(original_spectra) < 5:
                    original_spectra.append(target_spec)
                    reconstructed_spectra.append(recon_spec)
    
    # 打印平均指标
    print(f"\n{'='*60}")
    print(f"Test Results: {source_mode} -> {target_mode}")
    print(f"{'='*60}")
    print(f"Valid samples: {len(all_metrics['mse'])}/{total_samples} "
          f"(skipped invalid core-metric samples: {skipped_invalid_samples})")
    if len(all_metrics['mse']) == 0:
        print("Error: no valid samples remain after filtering invalid metrics.")
        return all_metrics
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
    
    # 绘制对比图
    if len(original_spectra) > 0:
        plot_comparison(
            original_spectra, reconstructed_spectra,
            save_dir, 'test',
            source_mode, target_mode,
            num_samples=min(5, len(original_spectra))
        )

    # 保存全量预测/目标/输入，及指定索引的对照
    if sample_indices and all_targets and all_preds and all_sources:
        os.makedirs(save_dir, exist_ok=True)
        preds_arr = np.vstack(all_preds)
        targets_arr = np.vstack(all_targets)
        sources_arr = np.vstack(all_sources)
        idx_arr = np.array(sample_indices, dtype=int)

        def _save_matrix(path, idx, matrix):
            with open(path, "w", encoding="utf-8") as f:
                header = "index," + ",".join([f"v{i}" for i in range(matrix.shape[1])])
                f.write(header + "\n")
                for i_row, row in zip(idx, matrix):
                    f.write(f"{i_row}," + ",".join(f"{v:.6e}" for v in row) + "\n")

        preds_path = Path(save_dir) / f"vae_{source_mode}2{target_mode}_preds.csv"
        targets_path = Path(save_dir) / f"vae_{source_mode}2{target_mode}_targets.csv"
        sources_path = Path(save_dir) / f"vae_{source_mode}2{target_mode}_sources.csv"
        _save_matrix(preds_path, idx_arr, preds_arr)
        _save_matrix(targets_path, idx_arr, targets_arr)
        _save_matrix(sources_path, idx_arr, sources_arr)
        print(f"\nSaved predictions to: {preds_path}")
        print(f"Saved targets to:      {targets_path}")
        print(f"Saved sources to:      {sources_path}")

        # 保存每个样本的R2列表（顺序与index一致）
        r2_list = np.array(all_metrics['r2'], dtype=float)
        r2_path = Path(save_dir) / f"vae_{source_mode}2{target_mode}_r2_per_sample.csv"
        with open(r2_path, "w", encoding="utf-8") as f:
            f.write("index,r2\n")
            for idx_val, r2_val in zip(idx_arr, r2_list):
                if not np.isnan(r2_val):
                    f.write(f"{idx_val},{r2_val:.6f}\n")
        print(f"Saved per-sample R2 to: {r2_path}")

        # 保存per-sample PSNR, SSIM, JS Divergence
        psnr_list = np.array(all_metrics['psnr'], dtype=float)
        ssim_list = np.array(all_metrics['ssim'], dtype=float)
        js_div_list = np.array(all_metrics['js_div'], dtype=float)
        metrics_path = Path(save_dir) / f"vae_{source_mode}2{target_mode}_metrics_per_sample.csv"
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("index,psnr,ssim,js_div\n")
            for idx_val, psnr_val, ssim_val, js_val in zip(idx_arr, psnr_list, ssim_list, js_div_list):
                f.write(f"{idx_val},{psnr_val:.6f},{ssim_val:.6f},{js_val:.6f}\n")
        print(f"Saved per-sample metrics (PSNR, SSIM, JS Div) to: {metrics_path}")

        if dump_index:
            for want_idx in dump_index:
                if want_idx in idx_arr:
                    pos = np.where(idx_arr == want_idx)[0][0]
                    tgt_row = targets_arr[pos]
                    pred_row = preds_arr[pos]
                    src_row = sources_arr[pos]
                    out_path = Path(save_dir) / f"vae_{source_mode}2{target_mode}_idx{want_idx}.csv"
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write("target,pred,source\n")
                        for tv, pv, sv in zip(tgt_row, pred_row, src_row):
                            f.write(f"{tv:.6e},{pv:.6e},{sv:.6e}\n")
                    print(f"Dumped target/pred for index {want_idx} -> {out_path}")
                else:
                    print(f"Warning: requested index {want_idx} not found in this test run.")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='Test Cross-Modal VAE')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing processed CSV files')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--source_mode', type=str, required=True,
                       choices=['ir', 'uv', 'raman'],
                       help='Source modality')
    parser.add_argument('--target_mode', type=str, required=True,
                       choices=['ir', 'uv', 'raman'],
                       help='Target modality')
    parser.add_argument('--source_csv', type=str, default=None,
                       help='Custom source CSV filename (optional, overrides default naming)')
    parser.add_argument('--target_csv', type=str, default=None,
                       help='Custom target CSV filename (optional, overrides default naming)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--heatmap_size', type=int, default=3600,
                       help='Heatmap size (must be a perfect square, e.g., 3600=60x60, 1024=32x32)')
    parser.add_argument('--resize_shape', type=int, nargs=2, default=[60, 60],
                       help='Heatmap reshape size, e.g., 60 60 or 32 32')
    parser.add_argument('--preserve_spectral_order', action='store_true',
                       help='Directly reshape spectra without patch-wise reordering')
    parser.add_argument('--source_size', type=int, default=None,
                       help='Optional source spectrum length before heatmap (if None, uses heatmap_size)')
    parser.add_argument('--target_size', type=int, default=None,
                       help='Optional target spectrum length before heatmap (if None, uses heatmap_size)')
    parser.add_argument('--save_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--no_split', action='store_true',
                       help='Do not split dataset; treat the whole provided CSV/H5 as the test set. '
                            'Note: For QM9S / QMe14S, the same random_split test subset as training is used unless this flag is set. '
                            'For other datasets, if --source_csv/--target_csv is provided, full dataset is used by default.')
    parser.add_argument('--use_split_test', action='store_true',
                       help='Force random_split and evaluate only the test subset, even when --source_csv/--target_csv is provided.')
    parser.add_argument('--dump_index', type=int, nargs='+', default=None,
                       help='List of sample indices (within this test run) to dump target/pred spectra to CSV')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    device = get_device(args.cpu)
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")

    # 设置随机种子确保数据分割和模型行为一致
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 创建模型
    model = CrossModalVAE(
        in_channels=1,
        hidden_channels=args.hidden_channels,
        latent_dim=args.latent_dim,
        image_size=tuple(args.resize_shape),
        num_modes=3,
        use_physical_prior=False,  # 与训练时保持一致
        physical_dim=7
    ).to(device)
    
    # 加载检查点
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f'vae_{args.source_mode}2{args.target_mode}_best_seed{args.seed}.pt'
    )

    # 向后兼容旧的未带seed命名
    if not os.path.exists(checkpoint_path):
        checkpoint_path_old = os.path.join(
            args.checkpoint_dir,
            f'vae_{args.source_mode}2{args.target_mode}_best.pt'
        )
        if os.path.exists(checkpoint_path_old):
            checkpoint_path = checkpoint_path_old
            print(f"Warning: Using old checkpoint format (without seed). Consider retraining with --seed {args.seed}")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 允许非严格加载，避免键不匹配导致报错
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded checkpoint from: {checkpoint_path}")
    
    # 加载测试数据
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
        print(f"Error: Data files not found")
        print(f"Source: {source_csv}")
        print(f"Target: {target_csv}")
        return
    
    if args.use_split_test and args.no_split:
        raise ValueError("--use_split_test and --no_split cannot be used together.")

    # QM9S / QMe14S：默认与训练相同的 test 子集
    if args.use_split_test:
        use_full_as_test = False
    elif paired_dataset_use_random_split_by_default(
        args.data_dir, args.source_csv, args.target_csv
    ):
        use_full_as_test = args.no_split
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
            use_h5=True,
            preserve_spectral_order=args.preserve_spectral_order
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
            seed=args.seed,
            preserve_spectral_order=args.preserve_spectral_order
        )
        print("Data split: using random_split(test subset) to match training split.")
    
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # 测试
    metrics = test_modality_pair(
        model, test_loader,
        args.source_mode, args.target_mode,
        device, args.save_dir,
        dump_index=args.dump_index
    )
    
    print("\nTesting completed!")


if __name__ == '__main__':
    main()

