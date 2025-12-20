"""
测试Flow Matching模型在不同模态转换上的性能
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from model_flow import ConditionalFlowMatching
from utils import (
    inverse_heatmap_to_spectrum,
    get_device
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


def plot_flow_process(source_heatmaps, target_heatmaps, path_heatmaps_list, save_dir,
                     source_mode, target_mode, num_samples=6, num_steps_show=5, 
                     target_min_list=None, target_max_list=None):
    """
    可视化Flow Matching的生成过程
    展示从源模态到目标模态的转换过程
    
    Args:
        source_heatmaps: 源热图列表 (list of (H, W) arrays) - 归一化后的[0,1]
        target_heatmaps: 目标热图列表 (list of (H, W) arrays) - 归一化后的[0,1]
        path_heatmaps_list: 每个样本的转换路径列表 (list of lists, each inner list contains (H, W) arrays)
        save_dir: 保存目录
        source_mode: 源模态
        target_mode: 目标模态
        num_samples: 展示的样本数量（行数）
        num_steps_show: 展示的步骤数（列数，包括起点和终点）
        target_min_list: 目标光谱最小值列表（用于denormalize）
        target_max_list: 目标光谱最大值列表（用于denormalize）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples = min(num_samples, len(source_heatmaps))
    
    # 创建6行5列的子图
    fig, axes = plt.subplots(num_samples, num_steps_show, figsize=(15, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        path = path_heatmaps_list[i]
        source_hm = source_heatmaps[i]
        target_hm = target_heatmaps[i]
        
        # 选择要展示的步骤（包括起点和终点）
        total_steps = len(path)
        step_indices = [0]  # 起点
        if total_steps > 1:
            # 中间步骤均匀分布
            for j in range(1, num_steps_show - 1):
                idx = int(j * (total_steps - 1) / (num_steps_show - 1))
                step_indices.append(min(idx, total_steps - 1))
        if total_steps > 1:
            step_indices.append(total_steps - 1)  # 终点
        
        # 确保正好5个步骤
        step_indices = step_indices[:num_steps_show]
        while len(step_indices) < num_steps_show and total_steps > 1:
            step_indices.append(total_steps - 1)
        
        # Denormalize热图（如果提供了min/max值）
        if target_min_list is not None and target_max_list is not None:
            spec_min = target_min_list[i]
            spec_max = target_max_list[i]
            denormalize_fn = lambda hm: hm * (spec_max - spec_min + 1e-8) + spec_min
        else:
            denormalize_fn = lambda hm: hm  # 不denormalize
        
        # 计算该样本所有热图的范围用于统一显示
        all_heatmaps = [denormalize_fn(source_hm)] + [denormalize_fn(target_hm)]
        if len(path) > 0:
            all_heatmaps.extend([denormalize_fn(p) for p in path])
        vmin = min([hm.min() for hm in all_heatmaps])
        vmax = max([hm.max() for hm in all_heatmaps])
        
        # 绘制每个步骤
        for j, step_idx in enumerate(step_indices):
            if j == 0:
                # 第一列：源光谱（IR）
                heatmap = denormalize_fn(source_hm)
                title = f'{source_mode.upper()}\n(t=0)'
            elif j == num_steps_show - 1:
                # 最后一列：目标光谱（Raman）
                heatmap = denormalize_fn(target_hm)
                title = f'{target_mode.upper()}\n(t=1.0)'
            else:
                # 中间步骤：转换过程
                if step_idx < len(path):
                    heatmap = denormalize_fn(path[step_idx])
                    t_val = step_idx / (total_steps - 1) if total_steps > 1 else 0.0
                    title = f't={t_val:.2f}'
                else:
                    heatmap = denormalize_fn(target_hm)
                    title = f't=1.0'
            
            ax = axes[i, j]
            # 显示60x60热图（patch），使用'imshow'直接显示2D图像（denormalize后的）
            im = ax.imshow(heatmap, cmap='viridis', aspect='equal', interpolation='nearest', 
                          vmin=vmin, vmax=vmax)  # 使用denormalize后的范围
            ax.set_title(title, fontsize=10, fontweight='bold' if j in [0, num_steps_show-1] else 'normal')
            ax.axis('off')
            ax.set_aspect('equal')  # 保持60x60的方形比例
            
            # 只在第一行最后一列添加颜色条
            if i == 0 and j == num_steps_show - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')
        
        # 在左侧添加样本编号
        axes[i, 0].text(-0.15, 0.5, f'Sample {i+1}', 
                       transform=axes[i, 0].transAxes, 
                       fontsize=12, fontweight='bold',
                       va='center', ha='right')
    
    plt.suptitle(f'Flow Matching Process: {source_mode.upper()} → {target_mode.upper()}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = os.path.join(save_dir, f'flow_{source_mode}2{target_mode}_process.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved flow process visualization to: {save_path}")


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
        axes[i].plot(x_axis, recon, label='Generated (Flow Matching)', alpha=0.7, 
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
                      save_dir='results', num_steps=100, use_rk4=True):
    """
    测试单个模态对的转换性能
    
    Args:
        model: 训练好的Flow Matching模型
        test_loader: 测试数据加载器
        source_mode: 源模态
        target_mode: 目标模态
        device: 计算设备
        save_dir: 结果保存目录
        num_steps: ODE求解步数
        use_rk4: 是否使用RK4方法（更准确但更慢）
    """
    model.eval()
    
    mode_map = {'ir': 0, 'uv': 1, 'raman': 2}
    target_mode_idx = mode_map[target_mode]
    
    all_metrics = {
        'mse': [], 'rmse': [], 'mae': [], 'mape': [],
        'r2': [], 'pearson': []
    }
    
    original_spectra = []
    generated_spectra = []
    
    # 用于Flow过程可视化的数据
    source_heatmaps_for_flow = []
    target_heatmaps_for_flow = []
    path_heatmaps_list = []
    target_min_list_for_flow = []
    target_max_list_for_flow = []
    
    # 使用tqdm显示测试进度
    pbar = tqdm(test_loader, desc=f"Testing {source_mode}->{target_mode}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            source_batch, target_batch = batch[0], batch[1]
            source_min_batch = batch[2]
            source_max_batch = batch[3]
            target_min_batch = batch[4]
            target_max_batch = batch[5]
            
            source_batch = source_batch.to(device)
            
            # 收集前6个样本用于Flow过程可视化
            collect_flow_data = len(source_heatmaps_for_flow) < 6
            
            if collect_flow_data:
                # 使用Flow Matching生成，并获取完整路径
                generated, path = model.sample(
                    source_batch[:6-len(source_heatmaps_for_flow)] if len(source_heatmaps_for_flow) < 6 else source_batch,
                    target_mode=target_mode_idx,
                    num_steps=num_steps,
                    use_rk4=use_rk4,
                    return_path=True
                )
                
                # 处理路径数据
                batch_size_collect = min(6 - len(source_heatmaps_for_flow), source_batch.size(0))
                path_np = [p.cpu().numpy() for p in path]
                
                for i in range(batch_size_collect):
                    source_hm = source_batch[i, 0].cpu().numpy()  # (H, W)
                    target_hm = target_batch[i, 0].numpy()  # (H, W)
                    path_hms = [p[i, 0] for p in path_np]  # List of (H, W)
                    spec_min = float(target_min_batch[i])
                    spec_max = float(target_max_batch[i])
                    
                    source_heatmaps_for_flow.append(source_hm)
                    target_heatmaps_for_flow.append(target_hm)
                    path_heatmaps_list.append(path_hms)
                    target_min_list_for_flow.append(spec_min)
                    target_max_list_for_flow.append(spec_max)
                
                # 继续处理完整batch用于指标计算
                generated = model.sample(
                    source_batch,
                    target_mode=target_mode_idx,
                    num_steps=num_steps,
                    use_rk4=use_rk4,
                    return_path=False
                )
            else:
                # 使用Flow Matching生成（不返回路径）
                generated = model.sample(
                    source_batch,
                    target_mode=target_mode_idx,
                    num_steps=num_steps,
                    use_rk4=use_rk4,
                    return_path=False
                )
            
            # 转换为numpy
            gen_np = generated.cpu().numpy()
            target_np = target_batch.numpy()
            
            # 将热图转换回光谱
            for i in range(gen_np.shape[0]):
                gen_heatmap = gen_np[i, 0]  # (H, W)
                target_heatmap = target_np[i, 0]
                
                spec_min = float(target_min_batch[i])
                spec_max = float(target_max_batch[i])
                
                # 转换回1D光谱
                gen_spec = inverse_heatmap_to_spectrum(gen_heatmap, spec_min, spec_max)
                target_spec = inverse_heatmap_to_spectrum(target_heatmap, spec_min, spec_max)
                
                # 计算指标
                metrics = calculate_metrics(target_spec, gen_spec)
                for key in all_metrics:
                    if not np.isnan(metrics[key]):
                        all_metrics[key].append(metrics[key])
                
                # 保存前几个样本用于可视化
                if len(original_spectra) < 5:
                    original_spectra.append(target_spec)
                    generated_spectra.append(gen_spec)
            
            # 更新进度条，显示当前R²分数（如果有的话）
            if len(all_metrics['r2']) > 0:
                current_r2 = np.mean(all_metrics['r2'])
                pbar.set_postfix({'R²': f"{current_r2:.4f}", 'samples': len(all_metrics['r2'])})
            else:
                pbar.set_postfix({'samples': len(all_metrics['mse'])})
    
    # 打印平均指标
    print(f"\n{'='*60}")
    print(f"Test Results (Flow Matching): {source_mode} -> {target_mode}")
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
            original_spectra, generated_spectra,
            save_dir, 'flow',
            source_mode, target_mode,
            num_samples=min(5, len(original_spectra))
        )
    
    # 绘制Flow Matching过程可视化
    if len(source_heatmaps_for_flow) > 0:
        plot_flow_process(
            source_heatmaps_for_flow,
            target_heatmaps_for_flow,
            path_heatmaps_list,
            save_dir,
            source_mode,
            target_mode,
            num_samples=min(6, len(source_heatmaps_for_flow)),
            num_steps_show=5,
            target_min_list=target_min_list_for_flow if len(target_min_list_for_flow) > 0 else None,
            target_max_list=target_max_list_for_flow if len(target_max_list_for_flow) > 0 else None
        )
    
    return all_metrics


def generate_process_only(model, test_loader, source_mode, target_mode, device, 
                          save_dir='results', num_steps=150, use_rk4=True):
    """
    快速生成Flow Matching过程可视化（只处理6个样本，不运行完整测试）
    """
    model.eval()
    
    mode_map = {'ir': 0, 'uv': 1, 'raman': 2}
    target_mode_idx = mode_map[target_mode]
    
    source_heatmaps_for_flow = []
    target_heatmaps_for_flow = []
    path_heatmaps_list = []
    target_min_list_for_flow = []
    target_max_list_for_flow = []
    
    print(f"Generating flow process visualization: {source_mode} -> {target_mode}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(source_heatmaps_for_flow) >= 6:
                break
                
            source_batch, target_batch = batch[0], batch[1]
            target_min_batch = batch[4]
            target_max_batch = batch[5]
            
            source_batch = source_batch.to(device)
            
            # 只处理需要的样本数
            batch_size_collect = min(6 - len(source_heatmaps_for_flow), source_batch.size(0))
            source_batch_collect = source_batch[:batch_size_collect]
            
            # 使用Flow Matching生成，并获取完整路径
            generated, path = model.sample(
                source_batch_collect,
                target_mode=target_mode_idx,
                num_steps=num_steps,
                use_rk4=use_rk4,
                return_path=True
            )
            
            # 处理路径数据
            path_np = [p.cpu().numpy() for p in path]
            
            for i in range(batch_size_collect):
                source_hm = source_batch_collect[i, 0].cpu().numpy()  # (H, W)
                target_hm = target_batch[i, 0].numpy()  # (H, W)
                path_hms = [p[i, 0] for p in path_np]  # List of (H, W)
                spec_min = float(target_min_batch[i])
                spec_max = float(target_max_batch[i])
                
                source_heatmaps_for_flow.append(source_hm)
                target_heatmaps_for_flow.append(target_hm)
                path_heatmaps_list.append(path_hms)
                target_min_list_for_flow.append(spec_min)
                target_max_list_for_flow.append(spec_max)
    
    # 绘制Flow Matching过程可视化
    if len(source_heatmaps_for_flow) > 0:
        plot_flow_process(
            source_heatmaps_for_flow,
            target_heatmaps_for_flow,
            path_heatmaps_list,
            save_dir,
            source_mode,
            target_mode,
            num_samples=min(6, len(source_heatmaps_for_flow)),
            num_steps_show=5,
            target_min_list=target_min_list_for_flow,
            target_max_list=target_max_list_for_flow
        )
        print(f"Process visualization generated successfully!")
    else:
        print("Error: No samples collected for visualization")


def main():
    parser = argparse.ArgumentParser(description='Test Flow Matching Model')
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
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--save_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--num_steps', type=int, default=150, help='Number of ODE steps (default: 150, more steps = better quality)')
    parser.add_argument('--use_rk4', action='store_true', help='Use RK4 ODE solver (more accurate but slower, default: True)')
    parser.add_argument('--no_rk4', action='store_true', help='Disable RK4 and use Euler method (faster but less accurate)')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--process_only', action='store_true',
                       help='Only generate process visualization (fast, no full test)')
    args = parser.parse_args()
    
    # 设置随机种子以确保结果可复现
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
    
    # 创建模型（2D架构）
    model = ConditionalFlowMatching(
        in_channels=1,
        hidden_channels=args.hidden_channels,
        num_modes=3,
        image_size=(60, 60),
        sigma_min=0.01
    ).to(device)
    
    # 加载检查点（支持带seed和不带seed的文件名）
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f'flow_{args.source_mode}2{args.target_mode}_best_seed{args.seed}.pt'
    )
    
    # 如果带seed的文件不存在，尝试加载不带seed的文件（向后兼容）
    if not os.path.exists(checkpoint_path):
        checkpoint_path_old = os.path.join(
            args.checkpoint_dir,
            f'flow_{args.source_mode}2{args.target_mode}_best.pt'
        )
        if os.path.exists(checkpoint_path_old):
            checkpoint_path = checkpoint_path_old
            print(f"Warning: Using old checkpoint format (without seed). Consider retraining with --seed {args.seed}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded checkpoint from: {checkpoint_path}")
    
    # 加载测试数据
    data_dir = Path(args.data_dir)
    source_file = f'{args.source_mode}_broaden_processed.csv'
    target_file = f'{args.target_mode}_broaden_processed.csv'
    
    source_csv = data_dir / source_file
    target_csv = data_dir / target_file
    
    if not source_csv.exists() or not target_csv.exists():
        print(f"Error: Data files not found")
        print(f"Source: {source_csv}")
        print(f"Target: {target_csv}")
        return
    
    # 创建测试数据加载器
    _, _, test_loader = get_paired_loaders(
        str(source_csv), str(target_csv),
        batch_size=args.batch_size,
        target_size=3600,
        resize_shape=(60, 60),
        out_channels=1,
        seed=args.seed
    )  
    
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # 确定是否使用RK4（默认True，使用RK4获得更准确的结果）
    # 如果指定了--no_rk4，则使用Euler；否则默认使用RK4（也可以通过--use_rk4显式指定）
    use_rk4 = not args.no_rk4  # 默认True，除非指定--no_rk4
    
    # 如果只生成process图，跳过完整测试
    if args.process_only:
        generate_process_only(
            model, test_loader,
            args.source_mode, args.target_mode,
            device, args.save_dir, args.num_steps, use_rk4
        )
        print("\nProcess visualization completed!")
    else:
        # 完整测试
        metrics = test_modality_pair(
            model, test_loader,
            args.source_mode, args.target_mode,
            device, args.save_dir, args.num_steps, use_rk4
        )
        print("\nTesting completed!")


if __name__ == '__main__':
    main()

