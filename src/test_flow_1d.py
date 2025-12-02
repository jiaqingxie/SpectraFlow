"""
测试Flow Matching模型（1D版本）在不同模态转换上的性能
支持从Parquet或CSV/HDF5加载数据
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from model_flow_1d import ConditionalFlowMatching1D
from dataset_parquet import get_parquet_loaders
from train_flow_1d import get_paired_loaders_1d, FlowMatchingTrainer1D
from utils import get_device


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


def plot_comparison(source_spectra, target_spectra, generated_spectra, save_dir,
                   source_name, target_name, num_samples=6):
    """
    可视化生成的频谱与真实频谱的对比
    
    Args:
        source_spectra: 源频谱列表 (list of 1D arrays)
        target_spectra: 目标频谱列表 (list of 1D arrays)
        generated_spectra: 生成的频谱列表 (list of 1D arrays)
        save_dir: 保存目录
        source_name: 源模态名称
        target_name: 目标模态名称
        num_samples: 展示的样本数量
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples = min(num_samples, len(source_spectra))
    
    # 创建子图
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        source = source_spectra[i]
        target = target_spectra[i]
        generated = generated_spectra[i]
        
        ax = axes[i]
        
        # 绘制源、目标和生成的频谱
        x_source = np.arange(len(source))
        x_target = np.arange(len(target))
        x_generated = np.arange(len(generated))
        
        ax.plot(x_source, source, 'b-', alpha=0.5, label=f'Source ({source_name})', linewidth=1)
        ax.plot(x_target, target, 'g-', label=f'Target ({target_name})', linewidth=1.5)
        ax.plot(x_generated, generated, 'r--', label=f'Generated ({target_name})', linewidth=1.5)
        
        ax.set_xlabel('Index')
        ax.set_ylabel('Intensity')
        ax.set_title(f'Sample {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{source_name}2{target_name}_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def test_model(model, test_loader, device, target_mode=None, num_steps=100, use_rk4=True):
    """
    测试模型性能
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 设备
        target_mode: 目标模态索引
        num_steps: ODE求解步数
        use_rk4: 是否使用RK4方法
    
    Returns:
        dict: 包含所有样本的指标和预测结果
    """
    model.eval()
    
    all_metrics = []
    all_source = []
    all_target = []
    all_generated = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            source_batch, target_batch = batch[0], batch[1]
            
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            
            # 生成样本
            target_length = target_batch.shape[-1]
            generated = model.sample(
                source_batch,
                target_mode=target_mode,
                num_steps=num_steps,
                use_rk4=use_rk4,
                target_length=target_length
            )
            
            # 转换为numpy
            source_np = source_batch.cpu().numpy()
            target_np = target_batch.cpu().numpy()
            generated_np = generated.cpu().numpy()
            
            # 计算每个样本的指标
            batch_size = source_np.shape[0]
            for i in range(batch_size):
                metrics = calculate_metrics(target_np[i, 0], generated_np[i, 0])
                all_metrics.append(metrics)
                all_source.append(source_np[i, 0])
                all_target.append(target_np[i, 0])
                all_generated.append(generated_np[i, 0])
    
    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        if len(values) > 0:
            avg_metrics[key] = np.mean(values)
        else:
            avg_metrics[key] = np.nan
    
    return {
        'avg_metrics': avg_metrics,
        'all_metrics': all_metrics,
        'source_spectra': all_source,
        'target_spectra': all_target,
        'generated_spectra': all_generated
    }


def main():
    parser = argparse.ArgumentParser(description='Test Flow Matching Model (1D)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_type', type=str, default='parquet',
                       choices=['parquet', 'csv'],
                       help='Data source type: parquet or csv')
    parser.add_argument('--data_dir', type=str, 
                       default='/mnt/shared-storage-user/xiejiaqing/datasets/multimodal-spectroscopic-dataset',
                       help='Directory containing data files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--sigma_min', type=float, default=0.01, help='Minimum noise level')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    
    # Parquet数据参数
    parser.add_argument('--source_field', type=str, default='ir_spectra',
                       help='Source spectrum field (for parquet data)')
    parser.add_argument('--target_field', type=str, default='h_nmr_spectra',
                       choices=['h_nmr_spectra', 'c_nmr_spectra'],
                       help='Target spectrum field (for parquet data)')
    
    # CSV数据参数
    parser.add_argument('--source_mode', type=str, default=None,
                       choices=['ir', 'uv', 'raman'],
                       help='Source modality (for CSV data)')
    parser.add_argument('--target_mode', type=str, default=None,
                       choices=['ir', 'uv', 'raman'],
                       help='Target modality (for CSV data)')
    
    # 通用参数
    parser.add_argument('--source_size', type=int, default=1800,
                       help='Source spectrum length')
    parser.add_argument('--target_size', type=int, default=10000,
                       help='Target spectrum length')
    parser.add_argument('--num_steps', type=int, default=100,
                       help='Number of ODE steps for generation')
    parser.add_argument('--use_rk4', action='store_true',
                       help='Use RK4 method (more accurate but slower)')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--num_plot_samples', type=int, default=6,
                       help='Number of samples to plot')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    device = get_device(args.cpu)
    print(f"Using device: {device}")
    
    # 配置文件
    class Config:
        in_channels = 1
        hidden_channels = args.hidden_channels
        num_modes = 1
        learning_rate = 4e-4
        epochs = 50
        batch_size = args.batch_size
        sigma_min = args.sigma_min
    
    config = Config()
    
    # 创建模型
    model = ConditionalFlowMatching1D(
        in_channels=config.in_channels,
        hidden_channels=config.hidden_channels,
        num_modes=config.num_modes,
        sigma_min=config.sigma_min
    ).to(device)
    
    # 加载检查点
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    # 根据数据源类型加载测试数据
    if args.data_type == 'parquet':
        data_path = Path(args.data_dir)
        if not data_path.exists():
            print(f"Error: Data directory not found: {data_path}")
            return
        
        _, _, test_loader = get_parquet_loaders(
            parquet_path=str(data_path),
            source_field=args.source_field,
            target_field=args.target_field,
            batch_size=args.batch_size,
            source_size=args.source_size,
            target_size=args.target_size,
            normalize=True,
            seed=args.seed
        )
        
        source_name = args.source_field
        target_name = args.target_field
        
    else:
        if args.source_mode is None or args.target_mode is None:
            print("Error: --source_mode and --target_mode are required for CSV data")
            return
        
        data_dir = Path(args.data_dir)
        source_file = f'{args.source_mode}_broaden_processed.csv'
        target_file = f'{args.target_mode}_broaden_processed.csv'
        
        source_csv = data_dir / source_file
        target_csv = data_dir / target_file
        
        if not source_csv.exists() or not target_csv.exists():
            print(f"Error: Data files not found")
            return
        
        _, _, test_loader = get_paired_loaders_1d(
            str(source_csv), str(target_csv),
            batch_size=args.batch_size,
            source_size=args.source_size,
            target_size=args.target_size,
            use_h5=True,
            normalize=True,
            seed=args.seed
        )
        
        source_name = args.source_mode
        target_name = args.target_mode
    
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # 测试模型
    print(f"\nTesting model...")
    print(f"ODE steps: {args.num_steps}, RK4: {args.use_rk4}")
    
    results = test_model(
        model, test_loader, device,
        target_mode=None,
        num_steps=args.num_steps,
        use_rk4=args.use_rk4
    )
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"Test Results: {source_name} -> {target_name}")
    print(f"{'='*60}")
    avg_metrics = results['avg_metrics']
    for key, value in avg_metrics.items():
        if not np.isnan(value):
            print(f"{key.upper()}: {value:.6f}")
        else:
            print(f"{key.upper()}: NaN")
    
    # 保存结果
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 保存指标到文件
    metrics_file = os.path.join(args.save_dir, f'{source_name}2{target_name}_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Test Results: {source_name} -> {target_name}\n")
        f.write(f"{'='*60}\n")
        for key, value in avg_metrics.items():
            if not np.isnan(value):
                f.write(f"{key.upper()}: {value:.6f}\n")
            else:
                f.write(f"{key.upper()}: NaN\n")
    print(f"\nSaved metrics to {metrics_file}")
    
    # 绘制对比图
    print(f"\nPlotting comparison...")
    plot_comparison(
        results['source_spectra'][:args.num_plot_samples],
        results['target_spectra'][:args.num_plot_samples],
        results['generated_spectra'][:args.num_plot_samples],
        args.save_dir,
        source_name, target_name,
        num_samples=args.num_plot_samples
    )
    
    print(f"\nTesting completed! Results saved to {args.save_dir}")


if __name__ == '__main__':
    main()

