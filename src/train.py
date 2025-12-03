"""
训练跨模态VAE模型，实现IR、UV、Raman之间的光谱转换
包含物理先验（KL散度）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
from scipy import interpolate
import math
import random
import argparse
import os
from pathlib import Path
import h5py

from model import CrossModalVAE
from utils import (
    spectrum_to_heatmap, inverse_heatmap_to_spectrum,
    get_device, kl_divergence_loss, reconstruction_loss, kl_annealing,
    save_checkpoint, compute_params, compute_prior_distribution,
    compute_kl_loss_with_prior
)


class PairedModalDataset(Dataset):
    """
    配对模态数据集
    支持任意两个模态之间的配对
    支持从CSV或HDF5文件加载数据（HDF5更快）
    支持不同的输入输出长度（source_size和target_size）
    """
    def __init__(self, source_csv, target_csv, source_size=None, target_size=None, 
                 heatmap_size=3600, resize_shape=(60, 60), 
                 out_channels=1, use_h5=True):
        # 优先使用HDF5格式（如果存在且启用）
        source_h5 = source_csv.replace('.csv', '.h5')
        target_h5 = target_csv.replace('.csv', '.h5')
        
        if use_h5 and os.path.exists(source_h5) and os.path.exists(target_h5):
            print(f"Loading from HDF5 files (fast mode)")
            # 从HDF5加载
            with h5py.File(source_h5, 'r') as f:
                source_spectra = f['spectra'][:]
                self.source_x_axis = f['x_axis'][:]
                self.source_physical_params = f['physical_params'][:] if 'physical_params' in f else None
            self.source_data = source_spectra
            
            with h5py.File(target_h5, 'r') as f:
                target_spectra = f['spectra'][:]
                self.target_x_axis = f['x_axis'][:]
                self.target_physical_params = f['physical_params'][:] if 'physical_params' in f else None
            self.target_data = target_spectra
        else:
            print(f"Loading from CSV files (slow mode)")
            # 从CSV加载
            self.source_df = pd.read_csv(source_csv, header=None)
            self.source_x_axis = self.source_df.iloc[0, 1:].values.astype(np.float32)
            self.source_data = self.source_df.iloc[1:, 1:].values.astype(np.float32)
            
            self.target_df = pd.read_csv(target_csv, header=None)
            self.target_x_axis = self.target_df.iloc[0, 1:].values.astype(np.float32)
            self.target_data = self.target_df.iloc[1:, 1:].values.astype(np.float32)
        
        assert len(self.source_data) == len(self.target_data), \
            f"Source and target data lengths must match: {len(self.source_data)} vs {len(self.target_data)}"
        
        # 设置源和目标光谱长度（用于插值）
        # 如果未指定，使用heatmap_size（保持向后兼容）
        self.source_size = source_size if source_size is not None else heatmap_size
        self.target_size = target_size if target_size is not None else heatmap_size
        
        # 热图大小（模型输入输出必须是固定大小）
        self.heatmap_size = heatmap_size
        self.resize_shape = resize_shape
        self.out_channels = out_channels
        
        # 处理源模态数据：先插值到source_size，再插值到heatmap_size用于热图转换
        self.source_interpolated_original = [self._interpolate(spec, self.source_size) for spec in self.source_data]
        self.source_interpolated_for_heatmap = [self._interpolate(spec, self.heatmap_size) for spec in self.source_interpolated_original]
        self.source_min_vals = [spec.min() for spec in self.source_interpolated_for_heatmap]
        self.source_max_vals = [spec.max() for spec in self.source_interpolated_for_heatmap]
        self.source_heatmaps = [
            self._spectrum_to_heatmap(spec, minv, maxv)
            for spec, minv, maxv in zip(self.source_interpolated_for_heatmap, self.source_min_vals, self.source_max_vals)
        ]
        
        # 若H5未提供，则离线计算物理参数（基于原始插值后的光谱）
        if getattr(self, 'source_physical_params', None) is None:
            self.source_physical_params = []
            for spec in self.source_interpolated_original:
                params = compute_params(spec)
                physical_vec = np.array([
                    params.get('mean', 0),
                    params.get('std', 0),
                    params.get('bandwidth', 0),
                    len(params.get('peak_positions', [])),
                    params.get('max_intensity', 0),
                    params.get('energy_range', (0, 0))[0],
                    params.get('energy_range', (0, 0))[1],
                ], dtype=np.float32)
                physical_vec = np.nan_to_num(physical_vec, nan=0.0, posinf=1e6, neginf=-1e6)
                physical_vec = np.clip(physical_vec, -1e3, 1e3)
                self.source_physical_params.append(physical_vec)
        
        # 处理目标模态数据：先插值到target_size，再插值到heatmap_size用于热图转换
        self.target_interpolated_original = [self._interpolate(spec, self.target_size) for spec in self.target_data]
        self.target_interpolated_for_heatmap = [self._interpolate(spec, self.heatmap_size) for spec in self.target_interpolated_original]
        self.target_min_vals = [spec.min() for spec in self.target_interpolated_for_heatmap]
        self.target_max_vals = [spec.max() for spec in self.target_interpolated_for_heatmap]
        self.target_heatmaps = [
            self._spectrum_to_heatmap(spec, minv, maxv)
            for spec, minv, maxv in zip(self.target_interpolated_for_heatmap, self.target_min_vals, self.target_max_vals)
        ]
    
    def _interpolate(self, spectrum, target_length):
        """插值到指定长度"""
        if len(spectrum) == target_length:
            return spectrum.copy()
        x_old = np.linspace(0, 1, len(spectrum))
        x_new = np.linspace(0, 1, target_length)
        return interpolate.interp1d(x_old, spectrum, kind='linear', 
                                   fill_value='extrapolate', bounds_error=False)(x_new)
    
    def _spectrum_to_heatmap(self, spectrum_1d, min_val, max_val):
        """转换为热图"""
        norm = (spectrum_1d - min_val) / (max_val - min_val + 1e-8)
        side = int(math.sqrt(len(norm)))
        patch_size = max(p for p in [10, 8, 5, 4, 2, 1] if side % p == 0)
        num_patches_per_row = side // patch_size
        patches = norm.reshape(-1, patch_size, patch_size)
        rows = [
            np.concatenate(patches[i * num_patches_per_row:(i + 1) * num_patches_per_row], axis=1)
            for i in range(num_patches_per_row)
        ]
        heatmap = np.concatenate(rows, axis=0)
        return np.expand_dims(heatmap, axis=0)  # (1, H, W)
    
    def __len__(self):
        return len(self.source_heatmaps)
    
    def __getitem__(self, idx):
        source_heatmap = self.source_heatmaps[idx]
        target_heatmap = self.target_heatmaps[idx]
        source_min = self.source_min_vals[idx]
        source_max = self.source_max_vals[idx]
        target_min = self.target_min_vals[idx]
        target_max = self.target_max_vals[idx]
        
        return (
            torch.tensor(source_heatmap).float(),
            torch.tensor(target_heatmap).float(),
            torch.tensor(source_min).float(),
            torch.tensor(source_max).float(),
            torch.tensor(target_min).float(),
            torch.tensor(target_max).float(),
            torch.tensor(self.source_interpolated_original[idx]).float(),  # 原始长度的源光谱
            torch.tensor(self.target_interpolated_original[idx]).float(),  # 原始长度的目标光谱
            torch.tensor(self.source_physical_params[idx]).float(),  # 物理参数
        )


class CrossModalVAETrainer:
    """跨模态VAE训练器"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.train_steps = 0
        
        # 创建模型（暂时关闭物理先验，用标准VAE）
        self.model = CrossModalVAE(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            latent_dim=config.latent_dim,
            image_size=config.resize_shape,
            num_modes=config.num_modes,
            use_physical_prior=False,  # 重新开启物理先验
            physical_dim=7
        ).to(device)

        # 与SpectrumViT一致的权重初始化（Xavier + bias=0）
        def _init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        self.model.apply(_init_weights)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # 模态映射 (用于确定目标模态索引)
        self.mode_map = {'ir': 0, 'uv': 1, 'raman': 2}
        self.reverse_mode_map = {0: 'ir', 1: 'uv', 2: 'raman'}
    
    def train_step(self, source_batch, target_batch, physical_params, 
                   source_mode, target_mode):
        """
        训练一步（使用物理先验融合）
        
        Args:
            source_batch: 源模态数据 (B, C, H, W)
            target_batch: 目标模态数据 (B, C, H, W)
            physical_params: 物理参数 (B, 7)
            source_mode: 源模态名称 ('ir', 'uv', 'raman')
            target_mode: 目标模态名称 ('ir', 'uv', 'raman')
        """
        self.model.train()
        self.train_steps += 1
        
        source_batch = source_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        physical_params = physical_params.to(self.device) if physical_params is not None else None
        target_mode_idx = self.mode_map[target_mode]
        
        # 输入检查
        if torch.isnan(source_batch).any() or torch.isinf(source_batch).any():
            print("[DEBUG] source_batch has NaN/Inf before forward")
        if physical_params is not None and (torch.isnan(physical_params).any() or torch.isinf(physical_params).any()):
            print("[DEBUG] physical_params has NaN/Inf before forward")

        # 前向传播（融合物理参数）
        recon, mu, logvar = self.model(source_batch, target_mode=target_mode_idx, physical_params=physical_params)
        
        # 调试：检查中间值
        if torch.isnan(recon).any() or torch.isinf(recon).any():
            print(f"[DEBUG] recon has NaN/Inf! recon range: [{recon.min().item():.4f}, {recon.max().item():.4f}]")
            print(f"[DEBUG] mu range: [{mu.min().item():.4f}, {mu.max().item():.4f}]")
            print(f"[DEBUG] logvar range: [{logvar.min().item():.4f}, {logvar.max().item():.4f}]")
            if physical_params is not None:
                print(f"[DEBUG] physical_params range: [{physical_params.min().item():.4f}, {physical_params.max().item():.4f}]")
        
        # 计算重建损失（在热图0-1尺度上）
        recon_loss = reconstruction_loss(recon, target_batch)
        
        # 使用标准KL散度损失（先验p(z) = N(0, 1)）
        kl_loss = kl_divergence_loss(mu, logvar)
        
        # KL权重退火（从0逐步增加，避免初期KL支配）
        # 修改退火策略：更快地接近beta_max，但初始beta更小
        beta = kl_annealing(self.train_steps, self.config.beta_max, 
                          k=0.1, x0=500) if self.model.training else self.config.beta_max
        
        total_loss = recon_loss + beta * kl_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'beta': beta
        }
    
    @torch.no_grad()
    def eval_step(self, source_batch, target_batch, target_mode, physical_params=None):
        """
        评估一步（使用物理先验融合）
        
        Args:
            source_batch: 源模态数据
            target_batch: 目标模态数据
            target_mode: 目标模态
            physical_params: 物理参数
        """
        self.model.eval()
        
        source_batch = source_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        physical_params = physical_params.to(self.device) if physical_params is not None else None
        target_mode_idx = self.mode_map[target_mode]
        
        recon, mu, logvar = self.model(source_batch, target_mode=target_mode_idx, physical_params=physical_params)
        
        recon_loss = reconstruction_loss(recon, target_batch)
        
        # 使用标准KL散度损失
        kl_loss = kl_divergence_loss(mu, logvar)
        
        return {
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'recon': recon.cpu(),
            'target': target_batch.cpu()
        }
    
    def save_checkpoint(self, path):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
        }, path)
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_steps = checkpoint.get('train_steps', 0)


def get_paired_loaders(source_csv, target_csv, batch_size=32, source_size=None, target_size=None,
                      heatmap_size=3600, resize_shape=(60, 60), out_channels=1, use_h5=True, seed=42):
    """
    获取配对数据加载器
    
    Args:
        source_csv: 源模态CSV文件路径
        target_csv: 目标模态CSV文件路径
        batch_size: 批次大小
        source_size: 源光谱长度（例如1800），如果为None则使用heatmap_size
        target_size: 目标光谱长度（例如10000），如果为None则使用heatmap_size
        heatmap_size: 热图大小（模型输入输出固定大小，默认3600）
        resize_shape: 热图形状（默认(60, 60)）
        out_channels: 输出通道数
        use_h5: 是否使用HDF5格式
        seed: 随机种子
    """
    dataset = PairedModalDataset(
        source_csv, target_csv, 
        source_size=source_size,
        target_size=target_size,
        heatmap_size=heatmap_size,
        resize_shape=resize_shape,
        out_channels=out_channels,
        use_h5=use_h5
    )
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # 使用固定的随机种子确保训练和测试时的分割一致
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_modality_pair(trainer, train_loader, val_loader, source_mode, target_mode, 
                       epochs=20, save_dir='checkpoints', seed=42):
    """训练单个模态对"""
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    print(f"\nTraining {source_mode} -> {target_mode}")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        # 训练
        trainer.model.train()
        train_losses = {'total': [], 'recon': [], 'kl': []}
        
        for batch in train_loader:
            source_batch, target_batch = batch[0], batch[1]
            physical_params = batch[8] if len(batch) > 8 else None  # 物理参数
            losses = trainer.train_step(
                source_batch, target_batch, physical_params,
                source_mode, target_mode
            )
            train_losses['total'].append(losses['total_loss'])
            train_losses['recon'].append(losses['recon_loss'])
            train_losses['kl'].append(losses['kl_loss'])
        
        # 验证
        trainer.model.eval()
        val_losses = {'recon': [], 'kl': []}
        
        with torch.no_grad():
            for batch in val_loader:
                source_batch, target_batch = batch[0], batch[1]
                physical_params = batch[8] if len(batch) > 8 else None
                eval_results = trainer.eval_step(
                    source_batch, target_batch, target_mode,
                    physical_params=physical_params
                )
                val_losses['recon'].append(eval_results['recon_loss'])
                val_losses['kl'].append(eval_results['kl_loss'])
        
        avg_train_loss = np.mean(train_losses['total'])
        avg_val_recon = np.mean(val_losses['recon'])
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f} "
              f"(Recon: {np.mean(train_losses['recon']):.6f}, "
              f"KL: {np.mean(train_losses['kl']):.6f})")
        print(f"  Val Recon: {avg_val_recon:.6f}")
        
        # 保存最佳模型
        if avg_val_recon < best_val_loss:
            best_val_loss = avg_val_recon
            checkpoint_path = os.path.join(
                save_dir, 
                f'vae_{source_mode}2{target_mode}_best_seed{seed}.pt'
            )
            trainer.save_checkpoint(checkpoint_path)
            print(f"  -> Saved best model (val_loss: {best_val_loss:.6f})")
    
    print(f"Training completed. Best val loss: {best_val_loss:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Cross-Modal VAE')
    parser.add_argument('--raw_data_dir', type=str, default=None,
                       help='Directory containing raw CSV files (will auto-process if provided)')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing processed CSV files')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='Learning rate')
    parser.add_argument('--beta_max', type=float, default=0.001, help='Max KL weight')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--modes', nargs='+', default=['ir', 'uv', 'raman'],
                       help='Modalities to train')
    parser.add_argument('--source_size', type=int, default=None,
                       help='Source spectrum length (e.g., 1800). If None, use heatmap_size.')
    parser.add_argument('--target_size', type=int, default=None,
                       help='Target spectrum length (e.g., 10000). If None, use heatmap_size.')
    parser.add_argument('--heatmap_size', type=int, default=3600,
                       help='Heatmap size for model input/output (must be perfect square, default 3600=60x60)')
    args = parser.parse_args()
    
    device = get_device(args.cpu)
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")

    # 设置随机种子保持可复现
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 配置文件
    class Config:
        in_channels = 1
        hidden_channels = args.hidden_channels
        latent_dim = args.latent_dim
        num_modes = 3
        resize_shape = (60, 60)
        learning_rate = args.learning_rate
        beta_max = args.beta_max
        batch_size = args.batch_size
    
    config = Config()
    
    # 如果需要，先处理原始数据
    if args.raw_data_dir is not None:
        print("Raw data directory provided, processing data first...")
        # 检查是否需要重新处理（检查HDF5文件）
        output_h5_files = [
            f'{args.data_dir}/ir_broaden_processed.h5',
            f'{args.data_dir}/uv_broaden_processed.h5',
            f'{args.data_dir}/raman_broaden_processed.h5'
        ]
        
        need_process = not all(os.path.exists(f) for f in output_h5_files)
        
        if need_process:
            print("Processing data to HDF5 format (fast)...")
            import subprocess
            import sys
            result = subprocess.run([
                sys.executable, 'src/process.py',
                '--data_dir', args.raw_data_dir,
                '--output_dir', args.data_dir,
                '--h5_only'  # 只保存HDF5，速度最快
            ], capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print(f"Error processing data: {result.stderr}")
                return
        else:
            print("HDF5 files already exist, skipping processing...")
    
    # 创建训练器
    trainer = CrossModalVAETrainer(config, device)
    
    # 定义所有模态对（双向转换）
    data_dir = Path(args.data_dir)
    mode_pairs = [
        ('ir', 'uv'),
        # ('ir', 'raman'),
        # ('uv', 'raman'),
        # ('uv', 'ir'),
        # ('ir', 'ir'),
        # ('raman', 'uv'),
    ]
    
    # 训练每个模态对
    for source_mode, target_mode in mode_pairs:
        # 处理文件名映射（支持HDF5和CSV）
        source_file = f'{source_mode}_broaden_processed.csv'
        target_file = f'{target_mode}_broaden_processed.csv'
        
        source_csv = data_dir / source_file
        target_csv = data_dir / target_file
        
        # 检查文件是否存在（优先检查HDF5）
        source_h5 = data_dir / f'{source_mode}_broaden_processed.h5'
        target_h5 = data_dir / f'{target_mode}_broaden_processed.h5'
        
        if os.path.exists(source_h5) and os.path.exists(target_h5):
            # 使用HDF5文件（即使CSV不存在也没关系）
            print(f"Using HDF5 files for {source_mode} -> {target_mode}")
        elif not source_csv.exists() or not target_csv.exists():
            print(f"Skipping {source_mode} -> {target_mode}: Files not found")
            print(f"  Looking for: {source_csv} or {source_h5} and {target_csv} or {target_h5}")
            continue
        
        # 创建数据加载器（会自动使用HDF5如果存在）
        train_loader, val_loader, test_loader = get_paired_loaders(
            str(source_csv), str(target_csv),
            batch_size=config.batch_size,
            source_size=args.source_size,
            target_size=args.target_size,
            heatmap_size=args.heatmap_size,
            resize_shape=config.resize_shape,
            out_channels=config.in_channels,
            use_h5=True,  # 启用HDF5优先加载
            seed=args.seed
        )
        
        # 训练
        train_modality_pair(
            trainer, train_loader, val_loader,
            source_mode, target_mode,
            epochs=args.epochs,
            save_dir=args.save_dir,
            seed=args.seed
        )
    
    print("All training completed!")


if __name__ == '__main__':
    main()

