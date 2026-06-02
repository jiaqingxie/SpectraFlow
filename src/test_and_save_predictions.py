"""
测试 Flow Matching 模型并保存详细的预测值，用于后续区域误差分析
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from scipy import interpolate
import argparse
import os
from pathlib import Path
from tqdm import tqdm

from model_flow import ConditionalFlowMatching
from utils import (
    inverse_heatmap_to_spectrum,
    get_device
)
from train import PairedModalDataset, get_paired_loaders


def test_and_save_detailed(model, test_loader, source_mode, target_mode, device, 
                          save_dir='results', num_steps=150, use_rk4=True,
                          save_intermediate_t=None, num_intermediate_samples=5):
    """
    测试模型并保存详细的预测值、真实值和源值，以及中间时刻的结果
    
    Args:
        model: 训练好的Flow Matching模型
        test_loader: 测试数据加载器
        source_mode: 源模态
        target_mode: 目标模态
        device: 计算设备
        save_dir: 结果保存目录
        num_steps: ODE求解步数
        use_rk4: 是否使用RK4方法
        save_intermediate_t: 要保存的中间时刻列表，如 [0.0, 0.25, 0.5, 0.75, 1.0]，如果None则自动选择
        num_intermediate_samples: 如果save_intermediate_t为None，自动选择多少个中间时刻
    """
    model.eval()
    
    mode_map = {'ir': 0, 'uv': 1, 'raman': 2}
    target_mode_idx = mode_map[target_mode]
    
    # 存储所有数据
    all_targets = []
    all_preds = []
    all_sources = []
    sample_indices = []
    
    # 存储中间时刻的数据（随机选择样本）
    intermediate_data = {}  # {t_value: [(sample_idx, spectrum), ...]}
    
    # 随机选择要保存中间时刻的样本索引（在知道总样本数之前先不选）
    selected_indices_for_intermediate = set()
    total_samples_known = False
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Testing {source_mode} -> {target_mode} and saving detailed predictions...")
    if save_intermediate_t is not None:
        print(f"Saving intermediate steps at t={save_intermediate_t} for {num_intermediate_samples} random samples")
    pbar = tqdm(test_loader, desc=f"Testing {source_mode}->{target_mode}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            source_batch, target_batch = batch[0], batch[1]
            source_min_batch = batch[2]
            source_max_batch = batch[3]
            target_min_batch = batch[4]
            target_max_batch = batch[5]
            
            source_batch = source_batch.to(device)
            
            # 在第一个batch时，随机选择要保存中间时刻的样本索引
            if batch_idx == 0 and not total_samples_known:
                # 估算总样本数（可能不完全准确，但足够选择随机样本）
                estimated_total = len(test_loader.dataset)
                if estimated_total > 0:
                    # 随机选择样本索引
                    all_indices = list(range(estimated_total))
                    np.random.shuffle(all_indices)
                    selected_indices_for_intermediate = set(all_indices[:num_intermediate_samples])
                    total_samples_known = True
                    print(f"Randomly selected {len(selected_indices_for_intermediate)} samples for intermediate steps: {sorted(list(selected_indices_for_intermediate))[:10]}...")
            
            # 生成预测（获取完整路径以便保存中间时刻）
            generated, path = model.sample(
                source_batch,
                target_mode=target_mode_idx,
                num_steps=num_steps,
                use_rk4=use_rk4,
                return_path=True
            )
            
            # 转换为numpy
            gen_np = generated.cpu().numpy()
            target_np = target_batch.numpy()
            
            # 将热图转换回光谱
            base_idx = len(all_targets)
            # 获取原始长度的目标光谱（如果dataset提供了）
            target_original = batch[7].numpy() if len(batch) > 7 else None
            source_original = batch[6].numpy() if len(batch) > 6 else None
            
            # 处理路径数据（用于保存中间时刻）
            path_np = [p.cpu().numpy() for p in path] if path is not None else None
            total_path_steps = len(path_np) if path_np is not None else 0
            
            # 确定要保存的中间时刻（只在第一个batch时确定）
            if batch_idx == 0:
                if save_intermediate_t is None and path_np is not None:
                    # 自动选择均匀分布的中间时刻
                    if total_path_steps > 1:
                        step_indices = np.linspace(0, total_path_steps - 1, num_intermediate_samples, dtype=int)
                        save_intermediate_t = [i / (total_path_steps - 1) for i in step_indices]
                    else:
                        save_intermediate_t = []
                
                # 初始化中间时刻数据字典
                if path_np is not None and save_intermediate_t:
                    for t_val in save_intermediate_t:
                        intermediate_data[t_val] = []
            
            for i in range(gen_np.shape[0]):
                gen_heatmap = gen_np[i, 0]  # (H, W)
                target_heatmap = target_np[i, 0]
                source_heatmap_np = source_batch[i, 0].cpu().numpy()
                
                spec_min = float(target_min_batch[i])
                spec_max = float(target_max_batch[i])
                src_min = float(source_min_batch[i])
                src_max = float(source_max_batch[i])
                
                # 转换回1D光谱（从热图恢复）
                gen_spec = inverse_heatmap_to_spectrum(gen_heatmap, spec_min, spec_max)
                
                # 如果dataset提供了原始长度光谱，使用原始长度；否则从热图恢复
                if target_original is not None:
                    target_spec = target_original[i].copy()
                    # 如果预测长度和原始长度不同，插值预测到原始长度
                    if len(gen_spec) != len(target_spec):
                        x_old = np.linspace(0, 1, len(gen_spec))
                        x_new = np.linspace(0, 1, len(target_spec))
                        gen_spec = np.interp(x_new, x_old, gen_spec)
                else:
                    target_spec = inverse_heatmap_to_spectrum(target_heatmap, spec_min, spec_max)
                
                if source_original is not None:
                    source_spec = source_original[i].copy()
                else:
                    source_spec = inverse_heatmap_to_spectrum(source_heatmap_np, src_min, src_max)
                
                all_targets.append(target_spec)
                all_preds.append(gen_spec)
                all_sources.append(source_spec)
                sample_indices.append(base_idx + i)
                
                # 保存中间时刻的结果（只保存随机选择的样本）
                current_sample_idx = base_idx + i
                if path_np is not None and save_intermediate_t and current_sample_idx in selected_indices_for_intermediate:
                    for t_val in save_intermediate_t:
                        # 找到对应的路径步骤索引
                        if total_path_steps > 1:
                            step_idx = int(round(t_val * (total_path_steps - 1)))
                            step_idx = max(0, min(step_idx, total_path_steps - 1))
                        else:
                            step_idx = 0
                        
                        if step_idx < len(path_np):
                            intermediate_heatmap = path_np[step_idx][i, 0]  # (H, W)
                            intermediate_spec = inverse_heatmap_to_spectrum(
                                intermediate_heatmap, spec_min, spec_max
                            )
                            
                            # 如果目标长度不同，插值
                            if target_original is not None and len(intermediate_spec) != len(target_spec):
                                x_old = np.linspace(0, 1, len(intermediate_spec))
                                x_new = np.linspace(0, 1, len(target_spec))
                                intermediate_spec = np.interp(x_new, x_old, intermediate_spec)
                            
                            intermediate_data[t_val].append((base_idx + i, intermediate_spec))
    
    # 保存为CSV文件
    if len(all_targets) > 0:
        preds_arr = np.vstack(all_preds)
        targets_arr = np.vstack(all_targets)
        sources_arr = np.vstack(all_sources)
        idx_arr = np.array(sample_indices, dtype=int)
        
        def _save_matrix(path, idx, matrix):
            """保存矩阵到CSV，第一列是index，后续列是每个波数点的值"""
            with open(path, "w", encoding="utf-8") as f:
                header = "index," + ",".join([f"wavenumber_{i}" for i in range(matrix.shape[1])])
                f.write(header + "\n")
                for i_row, row in zip(idx, matrix):
                    f.write(f"{i_row}," + ",".join(f"{v:.6e}" for v in row) + "\n")
        
        # 保存预测值、真实值、源值
        preds_path = Path(save_dir) / f"flow_{source_mode}2{target_mode}_preds.csv"
        targets_path = Path(save_dir) / f"flow_{source_mode}2{target_mode}_targets.csv"
        sources_path = Path(save_dir) / f"flow_{source_mode}2{target_mode}_sources.csv"
        
        _save_matrix(preds_path, idx_arr, preds_arr)
        _save_matrix(targets_path, idx_arr, targets_arr)
        _save_matrix(sources_path, idx_arr, sources_arr)
        
        print(f"\n{'='*60}")
        print(f"Saved detailed predictions:")
        print(f"  Predictions: {preds_path}")
        print(f"  Targets:     {targets_path}")
        print(f"  Sources:     {sources_path}")
        print(f"  Total samples: {len(all_targets)}")
        print(f"  Spectrum length: {preds_arr.shape[1]} points")
        print(f"{'='*60}")
        
        # 保存误差矩阵（每个样本每个波数点的误差）
        error_arr = np.abs(targets_arr - preds_arr)  # 绝对误差
        error_path = Path(save_dir) / f"flow_{source_mode}2{target_mode}_errors.csv"
        _save_matrix(error_path, idx_arr, error_arr)
        print(f"  Errors (absolute): {error_path}")
        
        # 保存每个波数点的统计信息（用于区域分析）
        wavenumber_stats = {
            'mean_error': np.mean(error_arr, axis=0),
            'std_error': np.std(error_arr, axis=0),
            'median_error': np.median(error_arr, axis=0),
            'max_error': np.max(error_arr, axis=0),
            'min_error': np.min(error_arr, axis=0),
        }
        
        stats_path = Path(save_dir) / f"flow_{source_mode}2{target_mode}_wavenumber_stats.csv"
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write("wavenumber_index,mean_error,std_error,median_error,max_error,min_error\n")
            for i in range(preds_arr.shape[1]):
                f.write(f"{i},{wavenumber_stats['mean_error'][i]:.6e},"
                       f"{wavenumber_stats['std_error'][i]:.6e},"
                       f"{wavenumber_stats['median_error'][i]:.6e},"
                       f"{wavenumber_stats['max_error'][i]:.6e},"
                       f"{wavenumber_stats['min_error'][i]:.6e}\n")
        print(f"  Wavenumber statistics: {stats_path}")
        
        # 保存中间时刻的结果
        if intermediate_data:
            for t_val in sorted(intermediate_data.keys()):
                t_data = intermediate_data[t_val]
                if len(t_data) > 0:
                    # 保存为CSV：每行是一个样本，每列是一个波数点
                    intermediate_path = Path(save_dir) / f"flow_{source_mode}2{target_mode}_intermediate_t{t_val:.3f}.csv"
                    with open(intermediate_path, "w", encoding="utf-8") as f:
                        # 获取光谱长度
                        spec_len = len(t_data[0][1])
                        header = "index," + ",".join([f"wavenumber_{i}" for i in range(spec_len)])
                        f.write(header + "\n")
                        for sample_idx, spec in t_data:
                            f.write(f"{sample_idx}," + ",".join(f"{v:.6e}" for v in spec) + "\n")
                    print(f"  Intermediate step t={t_val:.3f}: {intermediate_path} ({len(t_data)} samples)")
        
        return preds_path, targets_path, sources_path, error_path, stats_path
    else:
        print("Error: No samples processed!")
        return None, None, None, None, None


def main():
    parser = argparse.ArgumentParser(description='Test Flow Matching Model and Save Detailed Predictions')
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
                       help='Custom source CSV filename (optional)')
    parser.add_argument('--target_csv', type=str, default=None,
                       help='Custom target CSV filename (optional)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--heatmap_size', type=int, default=3600,
                       help='Heatmap size (must be a perfect square)')
    parser.add_argument('--resize_shape', type=int, nargs=2, default=[60, 60],
                       help='Heatmap reshape size, e.g., 60 60')
    parser.add_argument('--source_size', type=int, default=None,
                       help='Optional source spectrum length before heatmap')
    parser.add_argument('--target_size', type=int, default=None,
                       help='Optional target spectrum length before heatmap')
    parser.add_argument('--no_split', action='store_true',
                       help='Do not split dataset; treat the whole provided CSV/H5 as the test set')
    parser.add_argument('--save_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--num_steps', type=int, default=150, help='Number of ODE steps')
    parser.add_argument('--use_rk4', action='store_true', help='Use RK4 ODE solver')
    parser.add_argument('--no_rk4', action='store_true', help='Disable RK4 and use Euler method')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--save_intermediate_t', type=float, nargs='+', default=None,
                       help='List of t values (0-1) to save intermediate steps, e.g., 0.0 0.25 0.5 0.75 1.0')
    parser.add_argument('--num_intermediate_samples', type=int, default=5,
                       help='Number of samples to save intermediate steps for (if --save_intermediate_t not specified, auto-selects t values)')
    args = parser.parse_args()
    
    # 设置随机种子
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
    
    # 创建模型
    model = ConditionalFlowMatching(
        in_channels=1,
        hidden_channels=args.hidden_channels,
        num_modes=3,
        image_size=tuple(args.resize_shape),
        sigma_min=0.01
    ).to(device)
    
    # 加载检查点
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f'flow_{args.source_mode}2{args.target_mode}_best_seed{args.seed}.pt'
    )
    
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
    
    # 确定是否使用RK4
    use_rk4 = not args.no_rk4
    
    # 运行测试并保存
    test_and_save_detailed(
        model, test_loader,
        args.source_mode, args.target_mode,
        device, args.save_dir, args.num_steps, use_rk4,
        save_intermediate_t=args.save_intermediate_t,
        num_intermediate_samples=args.num_intermediate_samples
    )
    
    print("\nTesting and saving completed!")


if __name__ == '__main__':
    main()
