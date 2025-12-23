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

from model import CrossModalVAE
from utils import (
    inverse_heatmap_to_spectrum,
    get_device, load_checkpoint
)
from train import PairedModalDataset, get_paired_loaders


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
                      save_dir='results'):
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
        'r2': [], 'pearson': []
    }
    
    original_spectra = []
    reconstructed_spectra = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            source_batch, target_batch = batch[0], batch[1]
            target_min_batch = batch[4]
            target_max_batch = batch[5]
            physical_params = batch[8] if len(batch) > 8 else None
            
            source_batch = source_batch.to(device)
            physical_params = physical_params.to(device) if physical_params is not None else None
            
            # 前向传播
            recon, _, _ = model(source_batch, target_mode=target_mode_idx, physical_params=physical_params)
            
            # 调试：检查输出范围
            if batch_idx == 0:
                print(f"Recon range: [{recon.min().item():.4f}, {recon.max().item():.4f}]")
                print(f"Target range: [{target_batch.min().item():.4f}, {target_batch.max().item():.4f}]")
            
            # 转换为numpy
            recon_np = recon.cpu().numpy()
            target_np = target_batch.numpy()
            
            # 将热图转换回光谱
            for i in range(recon_np.shape[0]):
                recon_heatmap = recon_np[i, 0]  # (H, W)
                target_heatmap = target_np[i, 0]
                
                spec_min = float(target_min_batch[i])
                spec_max = float(target_max_batch[i])
                
                # 转换回1D光谱
                recon_spec = inverse_heatmap_to_spectrum(recon_heatmap, spec_min, spec_max)
                target_spec = inverse_heatmap_to_spectrum(target_heatmap, spec_min, spec_max)
                
                # 计算指标
                metrics = calculate_metrics(target_spec, recon_spec)
                for key in all_metrics:
                    if not np.isnan(metrics[key]):
                        all_metrics[key].append(metrics[key])
                
                # 保存前几个样本用于可视化
                if len(original_spectra) < 5:
                    original_spectra.append(target_spec)
                    reconstructed_spectra.append(recon_spec)
    
    # 打印平均指标
    print(f"\n{'='*60}")
    print(f"Test Results: {source_mode} -> {target_mode}")
    print(f"{'='*60}")
    print(f"MSE:      {np.mean(all_metrics['mse']):.6e} ± {np.std(all_metrics['mse']):.6e}")
    print(f"RMSE:     {np.mean(all_metrics['rmse']):.6e} ± {np.std(all_metrics['rmse']):.6e}")
    print(f"MAE:      {np.mean(all_metrics['mae']):.6e} ± {np.std(all_metrics['mae']):.6e}")
    print(f"MAPE:     {np.mean(all_metrics['mape']):.4f}% ± {np.std(all_metrics['mape']):.4f}%")
    print(f"R²:       {np.mean(all_metrics['r2']):.6f} ± {np.std(all_metrics['r2']):.6f}")
    print(f"Pearson:  {np.mean(all_metrics['pearson']):.6f} ± {np.std(all_metrics['pearson']):.6f}")
    
    # 绘制对比图
    if len(original_spectra) > 0:
        plot_comparison(
            original_spectra, reconstructed_spectra,
            save_dir, 'test',
            source_mode, target_mode,
            num_samples=min(5, len(original_spectra))
        )
    
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
    parser.add_argument('--source_size', type=int, default=None,
                       help='Optional source spectrum length before heatmap (if None, uses heatmap_size)')
    parser.add_argument('--target_size', type=int, default=None,
                       help='Optional target spectrum length before heatmap (if None, uses heatmap_size)')
    parser.add_argument('--save_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--no_split', action='store_true',
                       help='Do not split dataset; treat the whole provided CSV/H5 as the test set. '
                            'Note: For qm9s dataset, split is always used unless this flag is set. '
                            'For other datasets, if --source_csv/--target_csv is provided, full dataset is used by default.')
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
    
    # 测试
    metrics = test_modality_pair(
        model, test_loader,
        args.source_mode, args.target_mode,
        device, args.save_dir
    )
    
    print("\nTesting completed!")


if __name__ == '__main__':
    main()

