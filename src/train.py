"""
训练跨模态VAE模型，实现IR、UV、Raman之间的光谱转换
包含物理先验（KL散度）
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
from scipy import interpolate
import math
import argparse
import os
from pathlib import Path

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
    """
    def __init__(self, source_csv, target_csv, target_size=3600, resize_shape=(60, 60), 
                 out_channels=1):
        # 加载源模态数据
        self.source_df = pd.read_csv(source_csv, header=None)
        self.source_x_axis = self.source_df.iloc[0, 1:].values.astype(np.float32)
        self.source_data = self.source_df.iloc[1:, 1:].values.astype(np.float32)
        
        # 加载目标模态数据
        self.target_df = pd.read_csv(target_csv, header=None)
        self.target_x_axis = self.target_df.iloc[0, 1:].values.astype(np.float32)
        self.target_data = self.target_df.iloc[1:, 1:].values.astype(np.float32)
        
        assert len(self.source_data) == len(self.target_data), \
            f"Source and target data lengths must match: {len(self.source_data)} vs {len(self.target_data)}"
        
        self.target_size = target_size
        self.resize_shape = resize_shape
        self.out_channels = out_channels
        
        # 处理源模态数据
        self.source_interpolated = [self._interpolate(spec) for spec in self.source_data]
        self.source_min_vals = [spec.min() for spec in self.source_interpolated]
        self.source_max_vals = [spec.max() for spec in self.source_interpolated]
        self.source_heatmaps = [
            self._spectrum_to_heatmap(spec, minv, maxv)
            for spec, minv, maxv in zip(self.source_interpolated, self.source_min_vals, self.source_max_vals)
        ]
        
        # 处理目标模态数据
        self.target_interpolated = [self._interpolate(spec) for spec in self.target_data]
        self.target_min_vals = [spec.min() for spec in self.target_interpolated]
        self.target_max_vals = [spec.max() for spec in self.target_interpolated]
        self.target_heatmaps = [
            self._spectrum_to_heatmap(spec, minv, maxv)
            for spec, minv, maxv in zip(self.target_interpolated, self.target_min_vals, self.target_max_vals)
        ]
    
    def _interpolate(self, spectrum):
        """插值到目标长度"""
        x_old = np.linspace(0, 1, len(spectrum))
        x_new = np.linspace(0, 1, self.target_size)
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
            torch.tensor(self.source_interpolated[idx]).float(),  # 原始光谱用于计算参数
            torch.tensor(self.target_interpolated[idx]).float(),  # 原始光谱用于计算参数
        )


class CrossModalVAETrainer:
    """跨模态VAE训练器"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.train_steps = 0
        
        # 创建模型
        self.model = CrossModalVAE(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            latent_dim=config.latent_dim,
            image_size=config.resize_shape,
            num_modes=config.num_modes
        ).to(device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # 模态映射 (用于确定目标模态索引)
        self.mode_map = {'ir': 0, 'uv': 1, 'raman': 2}
        self.reverse_mode_map = {0: 'ir', 1: 'uv', 2: 'raman'}
    
    def train_step(self, source_batch, target_batch, source_spectra, target_spectra, 
                   source_mode, target_mode, use_physical_prior=True):
        """
        训练一步，包含物理先验计算
        
        Args:
            source_batch: 源模态数据 (B, C, H, W)
            target_batch: 目标模态数据 (B, C, H, W)
            source_spectra: 源模态原始光谱 (B, 3600)
            target_spectra: 目标模态原始光谱 (B, 3600)
            source_mode: 源模态名称 ('ir', 'uv', 'raman')
            target_mode: 目标模态名称 ('ir', 'uv', 'raman')
            use_physical_prior: 是否使用物理先验
        """
        self.model.train()
        self.train_steps += 1
        
        source_batch = source_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        target_mode_idx = self.mode_map[target_mode]
        
        # 前向传播
        recon, mu, logvar = self.model(source_batch, target_mode=target_mode_idx)
        
        # 计算损失
        recon_loss = reconstruction_loss(recon, target_batch)
        
        # 计算KL散度损失（带物理先验）
        # mu, logvar: VAE编码器输出的后验分布q(z|x)的参数
        # prior_mu, prior_logvar: 基于物理参数计算的先验分布p(z)的参数
        if use_physical_prior and source_spectra is not None and target_spectra is not None:
            # 计算物理先验参数（先验分布p(z)的参数）
            # 对批次中的每个样本计算参数，然后取平均
            batch_prior_mus = []
            batch_prior_logvars = []
            
            source_np = source_spectra.cpu().numpy() if isinstance(source_spectra, torch.Tensor) else source_spectra
            target_np = target_spectra.cpu().numpy() if isinstance(target_spectra, torch.Tensor) else target_spectra
            
            for i in range(min(len(source_np), len(target_np))):
                # 计算每个样本的物理参数
                source_params = compute_params(source_np[i])
                target_params = compute_params(target_np[i])
                # 基于物理参数计算先验分布p(z)的参数
                prior_mu_i, prior_logvar_i = compute_prior_distribution(source_params, target_params)
                batch_prior_mus.append(prior_mu_i)
                batch_prior_logvars.append(prior_logvar_i)
            
            # 转换为tensor（先验分布p(z)的参数）
            prior_mu = torch.tensor(np.mean(batch_prior_mus), 
                                   device=mu.device, dtype=mu.dtype) if batch_prior_mus else torch.zeros_like(mu[:, 0]).mean()
            prior_logvar = torch.tensor(np.mean(batch_prior_logvars), 
                                      device=logvar.device, dtype=logvar.dtype) if batch_prior_logvars else torch.zeros_like(logvar[:, 0]).mean()
            
            # 计算KL散度：KL(q(z|x) || p(z))
            # q(z|x)由编码器输出的mu和logvar定义
            # p(z)由物理先验的prior_mu和prior_logvar定义
            kl_loss = compute_kl_loss_with_prior(mu, logvar, prior_mu, prior_logvar)
        else:
            # 使用标准KL散度损失（先验p(z) = N(0, 1)）
            kl_loss = kl_divergence_loss(mu, logvar)
        
        # KL权重退火
        beta = kl_annealing(self.train_steps, self.config.beta_max, 
                          k=0.001, x0=500) if self.model.training else self.config.beta_max
        
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
    def eval_step(self, source_batch, target_batch, target_mode, 
                 source_spectra=None, target_spectra=None, use_physical_prior=False):
        """评估一步"""
        self.model.eval()
        
        source_batch = source_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        target_mode_idx = self.mode_map[target_mode]
        
        recon, mu, logvar = self.model(source_batch, target_mode=target_mode_idx)
        
        recon_loss = reconstruction_loss(recon, target_batch)
        
        # 计算KL损失（可选使用物理先验）
        if use_physical_prior and source_spectra is not None and target_spectra is not None:
            source_np = source_spectra.cpu().numpy() if isinstance(source_spectra, torch.Tensor) else source_spectra
            target_np = target_spectra.cpu().numpy() if isinstance(target_spectra, torch.Tensor) else target_spectra
            
            source_params = compute_params(source_np[0])
            target_params = compute_params(target_np[0])
            prior_mu, prior_logvar = compute_prior_distribution(source_params, target_params)
            
            prior_mu = torch.tensor(prior_mu, device=mu.device, dtype=mu.dtype)
            prior_logvar = torch.tensor(prior_logvar, device=logvar.device, dtype=logvar.dtype)
            kl_loss = compute_kl_loss_with_prior(mu, logvar, prior_mu, prior_logvar)
        else:
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


def get_paired_loaders(source_csv, target_csv, batch_size=32, target_size=3600, 
                      resize_shape=(60, 60), out_channels=1):
    """获取配对数据加载器"""
    dataset = PairedModalDataset(
        source_csv, target_csv, 
        target_size=target_size,
        resize_shape=resize_shape,
        out_channels=out_channels
    )
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_modality_pair(trainer, train_loader, val_loader, source_mode, target_mode, 
                       epochs=20, save_dir='checkpoints'):
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
            source_spectra = batch[6] if len(batch) > 6 else None  # 原始光谱数据
            target_spectra = batch[7] if len(batch) > 7 else None
            losses = trainer.train_step(
                source_batch, target_batch, 
                source_spectra, target_spectra,
                source_mode, target_mode, 
                use_physical_prior=True
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
                source_spectra = batch[6] if len(batch) > 6 else None
                target_spectra = batch[7] if len(batch) > 7 else None
                eval_results = trainer.eval_step(
                    source_batch, target_batch, target_mode,
                    source_spectra, target_spectra,
                    use_physical_prior=True
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
                f'vae_{source_mode}2{target_mode}_best.pt'
            )
            trainer.save_checkpoint(checkpoint_path)
            print(f"  -> Saved best model (val_loss: {best_val_loss:.6f})")
    
    print(f"Training completed. Best val loss: {best_val_loss:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Cross-Modal VAE')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing processed CSV files')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='Learning rate')
    parser.add_argument('--beta_max', type=float, default=0.01, help='Max KL weight')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--modes', nargs='+', default=['ir', 'uv', 'raman'],
                       help='Modalities to train')
    args = parser.parse_args()
    
    device = get_device(args.cpu)
    print(f"Using device: {device}")
    
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
    
    # 创建训练器
    trainer = CrossModalVAETrainer(config, device)
    
    # 定义所有模态对（双向转换）
    data_dir = Path(args.data_dir)
    mode_pairs = [
        ('ir', 'uv'),
        ('ir', 'raman'),
        ('uv', 'raman'),
        ('uv', 'ir'),
        ('raman', 'ir'),
        ('raman', 'uv'),
    ]
    
    # 训练每个模态对
    for source_mode, target_mode in mode_pairs:
        # 处理文件名映射
        source_file = f'{source_mode}_broaden_processed.csv'
        target_file = f'{target_mode}_broaden_processed.csv'
        
        source_csv = data_dir / source_file
        target_csv = data_dir / target_file
        
        # 检查文件是否存在
        if not source_csv.exists() or not target_csv.exists():
            print(f"Skipping {source_mode} -> {target_mode}: Files not found")
            continue
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = get_paired_loaders(
            str(source_csv), str(target_csv),
            batch_size=config.batch_size,
            target_size=3600,
            resize_shape=config.resize_shape,
            out_channels=config.in_channels
        )
        
        # 训练
        train_modality_pair(
            trainer, train_loader, val_loader,
            source_mode, target_mode,
            epochs=args.epochs,
            save_dir=args.save_dir
        )
    
    print("All training completed!")


if __name__ == '__main__':
    main()

