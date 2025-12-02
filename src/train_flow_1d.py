"""
训练Flow Matching模型（1D版本）
支持从CSV/HDF5或Parquet文件加载数据
支持不同输入输出长度的光谱转换
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
from scipy import interpolate
import argparse
import os
from pathlib import Path
import h5py
from tqdm import tqdm

from model_flow_1d import ConditionalFlowMatching1D
from dataset_parquet import get_parquet_loaders, ParquetSpectrumDataset
from utils import get_device


class PairedModalDataset1D(Dataset):
    """
    配对模态数据集（1D版本）
    直接处理1D光谱数据，支持不同输入输出长度
    支持从CSV或HDF5文件加载数据（HDF5更快）
    """
    def __init__(self, source_csv, target_csv, source_size=None, target_size=None, 
                 use_h5=True, normalize=True):
        """
        Args:
            source_csv: 源模态CSV文件路径
            target_csv: 目标模态CSV文件路径
            source_size: 源光谱长度（如果指定，会将源光谱插值到该长度）
            target_size: 目标光谱长度（如果指定，会将目标光谱插值到该长度）
            use_h5: 是否使用HDF5格式
            normalize: 是否归一化到[0, 1]
        """
        # 优先使用HDF5格式（如果存在且启用）
        source_h5 = source_csv.replace('.csv', '.h5')
        target_h5 = target_csv.replace('.csv', '.h5')
        
        if use_h5 and os.path.exists(source_h5) and os.path.exists(target_h5):
            print(f"Loading from HDF5 files (fast mode)")
            # 从HDF5加载
            with h5py.File(source_h5, 'r') as f:
                source_spectra = f['spectra'][:]
                self.source_x_axis = f['x_axis'][:]
            self.source_data = source_spectra
            
            with h5py.File(target_h5, 'r') as f:
                target_spectra = f['spectra'][:]
                self.target_x_axis = f['x_axis'][:]
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
        
        self.source_size = source_size
        self.target_size = target_size
        self.normalize = normalize
        
        # 处理源模态数据
        print(f"Processing source spectra...")
        self.source_spectra = []
        self.source_min_vals = []
        self.source_max_vals = []
        
        for spec in tqdm(self.source_data, desc="Processing source"):
            # 如果指定了长度，进行插值
            if source_size is not None and len(spec) != source_size:
                spec = self._interpolate(spec, source_size)
            
            # 归一化
            min_val = spec.min()
            max_val = spec.max()
            
            if normalize:
                spec_norm = (spec - min_val) / (max_val - min_val + 1e-8)
            else:
                spec_norm = spec
            
            self.source_spectra.append(spec_norm)
            self.source_min_vals.append(min_val)
            self.source_max_vals.append(max_val)
        
        # 处理目标模态数据
        print(f"Processing target spectra...")
        self.target_spectra = []
        self.target_min_vals = []
        self.target_max_vals = []
        
        for spec in tqdm(self.target_data, desc="Processing target"):
            # 如果指定了长度，进行插值
            if target_size is not None and len(spec) != target_size:
                spec = self._interpolate(spec, target_size)
            
            # 归一化
            min_val = spec.min()
            max_val = spec.max()
            
            if normalize:
                spec_norm = (spec - min_val) / (max_val - min_val + 1e-8)
            else:
                spec_norm = spec
            
            self.target_spectra.append(spec_norm)
            self.target_min_vals.append(min_val)
            self.target_max_vals.append(max_val)
        
        print(f"Source spectra shape: {self.source_spectra[0].shape}")
        print(f"Target spectra shape: {self.target_spectra[0].shape}")
    
    def _interpolate(self, spectrum, target_length):
        """插值到指定长度"""
        if len(spectrum) == target_length:
            return spectrum.copy()
        x_old = np.linspace(0, 1, len(spectrum))
        x_new = np.linspace(0, 1, target_length)
        return interpolate.interp1d(x_old, spectrum, kind='linear', 
                                   fill_value='extrapolate', bounds_error=False)(x_new)
    
    def __len__(self):
        return len(self.source_spectra)
    
    def __getitem__(self, idx):
        source_spec = self.source_spectra[idx]  # (L_source,)
        target_spec = self.target_spectra[idx]  # (L_target,)
        
        # 转换为tensor并添加channel维度: (1, L)
        source_tensor = torch.tensor(source_spec, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor(target_spec, dtype=torch.float32).unsqueeze(0)
        
        return source_tensor, target_tensor


def get_paired_loaders_1d(source_csv, target_csv, batch_size=32, source_size=None, 
                          target_size=None, use_h5=True, normalize=True, seed=42):
    """
    获取配对数据加载器（1D版本，从CSV/HDF5）
    
    Args:
        source_csv: 源模态CSV文件路径
        target_csv: 目标模态CSV文件路径
        batch_size: 批次大小
        source_size: 源光谱长度（例如1800）
        target_size: 目标光谱长度（例如10000）
        use_h5: 是否使用HDF5格式
        normalize: 是否归一化到[0, 1]
        seed: 随机种子
    """
    dataset = PairedModalDataset1D(
        source_csv, target_csv, 
        source_size=source_size,
        target_size=target_size,
        use_h5=use_h5,
        normalize=normalize
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


class FlowMatchingTrainer1D:
    """Flow Matching训练器（1D版本）"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.train_steps = 0
        
        # 创建模型（1D架构）
        self.model = ConditionalFlowMatching1D(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            num_modes=config.num_modes,
            sigma_min=config.sigma_min
        ).to(device)
        
        # 权重初始化
        def _init_weights(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        self.model.apply(_init_weights)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01
        )
        
        # 模态映射（用于CSV/HDF5数据）
        self.mode_map = {'ir': 0, 'uv': 1, 'raman': 2}
        self.reverse_mode_map = {0: 'ir', 1: 'uv', 2: 'raman'}
    
    def train_step(self, source_batch, target_batch, target_mode=None):
        """
        训练一步
        
        Args:
            source_batch: 源模态数据 (B, 1, L_source) - 1D光谱
            target_batch: 目标模态数据 (B, 1, L_target) - 1D光谱
            target_mode: 目标模态索引（对于NMR任务，可以设为None）
        """
        self.model.train()
        self.train_steps += 1
        
        source_batch = source_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        
        # 转换target_mode（如果是字符串）
        if target_mode is not None and isinstance(target_mode, str):
            target_mode = self.mode_map.get(target_mode, None)
        
        # 前向传播
        v_pred, v_true, x_t = self.model(
            x0=source_batch,
            x1=target_batch,
            t=None,  # 随机采样时间步
            target_mode=target_mode
        )
        
        # Flow Matching损失：混合损失（L1 + L2）
        l1_loss = F.l1_loss(v_pred, v_true)
        l2_loss = F.mse_loss(v_pred, v_true)
        velocity_loss = 0.6 * l1_loss + 0.4 * l2_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        velocity_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'loss': velocity_loss.item(),
        }
    
    @torch.no_grad()
    def eval_step(self, source_batch, target_batch, target_mode=None, num_steps=50, use_rk4=False):
        """
        评估一步
        
        Args:
            source_batch: 源模态数据 (B, 1, L_source)
            target_batch: 目标模态数据 (B, 1, L_target)
            target_mode: 目标模态索引
            num_steps: ODE求解步数
            use_rk4: 是否使用RK4方法
        """
        self.model.eval()
        
        source_batch = source_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        
        # 转换target_mode（如果是字符串）
        if target_mode is not None and isinstance(target_mode, str):
            target_mode = self.mode_map.get(target_mode, None)
        
        # 前向传播
        v_pred, v_true, x_t = self.model(
            x0=source_batch,
            x1=target_batch,
            t=None,
            target_mode=target_mode
        )
        
        # 损失
        l1_loss = F.l1_loss(v_pred, v_true)
        l2_loss = F.mse_loss(v_pred, v_true)
        loss = 0.6 * l1_loss + 0.4 * l2_loss
        
        # 生成样本（用于可视化）
        # 输出长度应该与target_batch相同
        target_length = target_batch.shape[-1]
        generated = self.model.sample(
            source_batch,
            target_mode=target_mode,
            num_steps=num_steps,
            use_rk4=use_rk4,
            target_length=target_length
        )
        
        # 生成损失
        gen_l1 = F.l1_loss(generated, target_batch)
        gen_l2 = F.mse_loss(generated, target_batch)
        gen_loss = 0.6 * gen_l1 + 0.4 * gen_l2
        
        return {
            'loss': loss.item(),
            'gen_loss': gen_loss.item(),
            'generated': generated.cpu(),
            'target': target_batch.cpu()
        }
    
    def save_checkpoint(self, path):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_steps': self.train_steps,
        }, path)
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_steps = checkpoint.get('train_steps', 0)


def train_modality_pair(trainer, train_loader, val_loader, source_name, target_name, 
                       epochs=20, save_dir='checkpoints', val_num_steps=50, seed=42):
    """训练单个模态对"""
    os.makedirs(save_dir, exist_ok=True)
    best_val_gen_loss = float('inf')
    
    print(f"\nTraining Flow Matching (1D): {source_name} -> {target_name}")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        # 训练
        trainer.model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            source_batch, target_batch = batch[0], batch[1]
            
            losses = trainer.train_step(
                source_batch, target_batch, target_mode=None
            )
            train_losses.append(losses['loss'])
            pbar.set_postfix({'loss': f"{losses['loss']:.6f}"})
        
        # 验证
        trainer.model.eval()
        val_losses = []
        val_gen_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                source_batch, target_batch = batch[0], batch[1]
                
                eval_results = trainer.eval_step(
                    source_batch, target_batch, target_mode=None,
                    num_steps=val_num_steps, use_rk4=False
                )
                val_losses.append(eval_results['loss'])
                val_gen_losses.append(eval_results['gen_loss'])
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_val_gen_loss = np.mean(val_gen_losses)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  Val Gen Loss: {avg_val_gen_loss:.6f}")
        
        # 更新学习率
        trainer.scheduler.step()
        current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6e}")
        
        # 保存最佳模型
        if avg_val_gen_loss < best_val_gen_loss:
            best_val_gen_loss = avg_val_gen_loss
            # 创建安全的文件名
            safe_target = target_name.replace('_', '-')
            checkpoint_path = os.path.join(
                save_dir, 
                f'flow_1d_{source_name}2{safe_target}_best_seed{seed}.pt'
            )
            trainer.save_checkpoint(checkpoint_path)
            print(f"  -> Saved best model (val_gen_loss: {best_val_gen_loss:.6f})")
    
    print(f"Training completed. Best val gen loss: {best_val_gen_loss:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Flow Matching Model (1D)')
    parser.add_argument('--data_type', type=str, default='parquet',
                       choices=['parquet', 'csv'],
                       help='Data source type: parquet or csv')
    parser.add_argument('--data_dir', type=str, 
                       default='/mnt/shared-storage-user/xiejiaqing/datasets/multimodal-spectroscopic-dataset',
                       help='Directory containing data files')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='Learning rate')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint directory')
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
                       help='Source spectrum length (default: 1800 for IR)')
    parser.add_argument('--target_size', type=int, default=10000,
                       help='Target spectrum length (default: 10000 for NMR)')
    parser.add_argument('--val_num_steps', type=int, default=50,
                       help='Number of ODE steps for validation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
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
    print(f"Data type: {args.data_type}")
    print(f"Source size: {args.source_size}")
    print(f"Target size: {args.target_size}")
    
    # 配置文件
    class Config:
        in_channels = 1
        hidden_channels = args.hidden_channels
        num_modes = 1  # 对于NMR任务，设为1（不需要模态嵌入）
        learning_rate = args.learning_rate
        epochs = args.epochs
        batch_size = args.batch_size
        sigma_min = args.sigma_min
    
    config = Config()
    
    # 创建训练器
    trainer = FlowMatchingTrainer1D(config, device)
    
    # 根据数据源类型加载数据
    if args.data_type == 'parquet':
        # 从Parquet加载
        print(f"\nLoading parquet data from: {args.data_dir}")
        data_path = Path(args.data_dir)
        if not data_path.exists():
            print(f"Error: Data directory not found: {data_path}")
            return
        
        train_loader, val_loader, test_loader = get_parquet_loaders(
            parquet_path=str(data_path),
            source_field=args.source_field,
            target_field=args.target_field,
            batch_size=config.batch_size,
            source_size=args.source_size,
            target_size=args.target_size,
            normalize=True,
            seed=args.seed
        )
        
        source_name = args.source_field
        target_name = args.target_field
        
    else:
        # 从CSV/HDF5加载
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
            print(f"Source: {source_csv}")
            print(f"Target: {target_csv}")
            return
        
        train_loader, val_loader, test_loader = get_paired_loaders_1d(
            str(source_csv), str(target_csv),
            batch_size=config.batch_size,
            source_size=args.source_size,
            target_size=args.target_size,
            use_h5=True,
            normalize=True,
            seed=args.seed
        )
        
        source_name = args.source_mode
        target_name = args.target_mode
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # 训练
    train_modality_pair(
        trainer, train_loader, val_loader,
        source_name, target_name,
        epochs=args.epochs,
        save_dir=args.save_dir,
        val_num_steps=args.val_num_steps,
        seed=args.seed
    )
    
    print("Training completed!")


if __name__ == '__main__':
    main()
