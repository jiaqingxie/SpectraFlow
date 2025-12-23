"""
训练Flow Matching模型，实现IR、UV、Raman之间的光谱转换
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm

from model_flow import ConditionalFlowMatching
from train import PairedModalDataset, get_paired_loaders
from utils import get_device


class FlowMatchingTrainer:
    """Flow Matching训练器"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.train_steps = 0
        
        # 创建模型（2D架构）
        self.model = ConditionalFlowMatching(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            num_modes=config.num_modes,
            image_size=config.resize_shape,
            sigma_min=config.sigma_min
        ).to(device)
        
        # 权重初始化（支持1D和2D卷积）
        def _init_weights(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
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
        
        # 模态映射
        self.mode_map = {'ir': 0, 'uv': 1, 'raman': 2}
        self.reverse_mode_map = {0: 'ir', 1: 'uv', 2: 'raman'}
    
    def train_step(self, source_batch, target_batch, target_mode, use_mixed_loss=False, gen_loss_weight=0.1, compute_gen_prob=0.1, train_gen_steps=100, use_rk4_in_train=False):
        """
        训练一步
        
        Args:
            source_batch: 源模态数据 (B, C, H, W) - 2D热图
            target_batch: 目标模态数据 (B, C, H, W) - 2D热图
            target_mode: 目标模态名称 ('ir', 'uv', 'raman')
            use_mixed_loss: 是否使用混合损失（velocity loss + generation loss）
            gen_loss_weight: 生成损失的权重（当use_mixed_loss=True时）
            compute_gen_prob: 计算生成损失的概率（0.0-1.0），用于加速训练
            train_gen_steps: 训练时生成损失的ODE步数（默认100，Euler方法）
            use_rk4_in_train: 训练时是否使用RK4（默认False，RK4占用更多GPU内存）
        """
        self.model.train()
        self.train_steps += 1
        
        source_batch = source_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        target_mode_idx = self.mode_map[target_mode]
        
        # 前向传播
        v_pred, v_true, x_t = self.model(
            x0=source_batch,
            x1=target_batch,
            t=None,  # 随机采样时间步
            target_mode=target_mode_idx
        )
        
        # Flow Matching损失：混合损失（L1 + L2）
        # L1损失保持稳定性，L2损失强调峰值强度（对Raman峰形有帮助）
        l1_loss = F.l1_loss(v_pred, v_true)
        l2_loss = F.mse_loss(v_pred, v_true)
        velocity_loss = 0.6 * l1_loss + 0.4 * l2_loss
        
        # 如果使用混合损失，同时优化生成损失
        # 但只在部分batch计算（通过概率控制）以加速训练
        gen_loss = None
        total_loss = velocity_loss
        
        if use_mixed_loss and np.random.rand() < compute_gen_prob:
            # 生成样本（训练时使用Euler方法避免GPU内存溢出，测试时再用RK4）
            generated = self.model.sample(
                source_batch,
                target_mode=target_mode_idx,
                num_steps=train_gen_steps,  # 默认100步，Euler方法
                use_rk4=use_rk4_in_train  # 默认False，避免GPU内存溢出
            )
            gen_l1 = F.l1_loss(generated, target_batch)
            gen_l2 = F.mse_loss(generated, target_batch)
            gen_loss = 0.6 * gen_l1 + 0.4 * gen_l2
            
            # 混合损失：velocity loss + generation loss
            # 注意：由于只在小部分batch计算，需要调整权重
            total_loss = velocity_loss + (gen_loss_weight / compute_gen_prob) * gen_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        result = {
            'loss': velocity_loss.item(),
            'total_loss': total_loss.item()
        }
        if gen_loss is not None:
            result['gen_loss'] = gen_loss.item()
        
        return result
    
    @torch.no_grad()
    def eval_step(self, source_batch, target_batch, target_mode, num_steps=50, use_rk4=False):
        """
        评估一步
        
        Args:
            source_batch: 源模态数据 (B, C, H, W) - 2D热图
            target_batch: 目标模态数据 (B, C, H, W) - 2D热图
            target_mode: 目标模态
            num_steps: ODE求解步数（验证时使用较少步数以加速）
            use_rk4: 是否使用RK4方法（验证时默认False以加速）
        """
        self.model.eval()
        
        source_batch = source_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        target_mode_idx = self.mode_map[target_mode]
        
        # 前向传播
        v_pred, v_true, x_t = self.model(
            x0=source_batch,
            x1=target_batch,
            t=None,
            target_mode=target_mode_idx
        )
        
        # 损失（与训练时一致，混合损失）
        l1_loss = F.l1_loss(v_pred, v_true)
        l2_loss = F.mse_loss(v_pred, v_true)
        loss = 0.6 * l1_loss + 0.4 * l2_loss
        
        # 生成样本（用于可视化）
        # 验证时使用Euler和较少步数以加速（测试时再用RK4和更多步数）
        generated = self.model.sample(
            source_batch,
            target_mode=target_mode_idx,
            num_steps=num_steps,
            use_rk4=use_rk4
        )
        
        # 生成损失（混合损失）
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


def train_modality_pair(trainer, train_loader, val_loader, source_mode, target_mode, 
                       epochs=20, save_dir='checkpoints', use_mixed_loss=False, gen_loss_weight=0.1, gen_loss_prob=0.1, val_num_steps=50, train_gen_steps=100, use_rk4_in_train=False, seed=42):
    """训练单个模态对"""
    os.makedirs(save_dir, exist_ok=True)
    # 使用val_gen_loss作为模型选择标准，因为它更直接反映最终生成质量
    best_val_gen_loss = float('inf')
    
    print(f"\nTraining Flow Matching: {source_mode} -> {target_mode}")
    if use_mixed_loss:
        print(f"Using mixed loss training (gen_loss_weight={gen_loss_weight})")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        # 训练
        trainer.model.train()
        train_losses = []
        train_gen_losses = []
        train_total_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            source_batch, target_batch = batch[0], batch[1]
            
            losses = trainer.train_step(
                source_batch, target_batch, target_mode,
                use_mixed_loss=use_mixed_loss,
                gen_loss_weight=gen_loss_weight,
                compute_gen_prob=gen_loss_prob,
                train_gen_steps=train_gen_steps,
                use_rk4_in_train=use_rk4_in_train
            )
            train_losses.append(losses['loss'])
            train_total_losses.append(losses['total_loss'])
            if 'gen_loss' in losses:
                train_gen_losses.append(losses['gen_loss'])
            
            pbar.set_postfix({'loss': f"{losses['loss']:.6f}"})
        
        # 验证
        trainer.model.eval()
        val_losses = []
        val_gen_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                source_batch, target_batch = batch[0], batch[1]
                
                eval_results = trainer.eval_step(
                    source_batch, target_batch, target_mode,
                    num_steps=val_num_steps, use_rk4=False  # 验证时用Euler加速
                )
                val_losses.append(eval_results['loss'])
                val_gen_losses.append(eval_results['gen_loss'])
        
        avg_train_loss = np.mean(train_losses)
        avg_train_total_loss = np.mean(train_total_losses)
        avg_val_loss = np.mean(val_losses)
        avg_val_gen_loss = np.mean(val_gen_losses)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Velocity Loss: {avg_train_loss:.6f}")
        if use_mixed_loss:
            print(f"  Train Total Loss: {avg_train_total_loss:.6f}")
            if train_gen_losses:
                avg_train_gen_loss = np.mean(train_gen_losses)
                print(f"  Train Gen Loss: {avg_train_gen_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  Val Gen Loss: {avg_val_gen_loss:.6f}")
        
        # 更新学习率
        trainer.scheduler.step()
        current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6e}")
        
        # 保存最佳模型（使用val_gen_loss，因为它更直接反映最终生成质量）
        if avg_val_gen_loss < best_val_gen_loss:
            best_val_gen_loss = avg_val_gen_loss
            checkpoint_path = os.path.join(
                save_dir, 
                f'flow_{source_mode}2{target_mode}_best_seed{seed}.pt'
            )
            trainer.save_checkpoint(checkpoint_path)
            print(f"  -> Saved best model (val_gen_loss: {best_val_gen_loss:.6f})")
    
    print(f"Training completed. Best val gen loss: {best_val_gen_loss:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Flow Matching Model')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing processed CSV files')
    parser.add_argument('--source_csv', type=str, default=None,
                       help='Custom source CSV filename (optional, overrides default naming; can be absolute or relative to data_dir)')
    parser.add_argument('--target_csv', type=str, default=None,
                       help='Custom target CSV filename (optional, overrides default naming; can be absolute or relative to data_dir)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='Learning rate (aligned with VAE, more stable)')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint directory (base)')
    parser.add_argument('--no_dataset_subdir', action='store_true',
                       help='Do not auto-create a dataset-specific subdirectory under save_dir (default: False). '
                            'By default, if --data_dir path contains "qm9s", checkpoints will be saved under save_dir/qm9s/.')
    parser.add_argument('--sigma_min', type=float, default=0.01, help='Minimum noise level')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--source_mode', type=str, required=True,
                       choices=['ir', 'uv', 'raman'],
                       help='Source modality')
    parser.add_argument('--target_mode', type=str, required=True,
                       choices=['ir', 'uv', 'raman'],
                       help='Target modality')
    parser.add_argument('--heatmap_size', type=int, default=3600,
                       help='Heatmap size (must be a perfect square, e.g., 3600=60x60, 1024=32x32)')
    parser.add_argument('--resize_shape', type=int, nargs=2, default=[60, 60],
                       help='Heatmap reshape size, e.g., 60 60 or 32 32')
    parser.add_argument('--source_size', type=int, default=None,
                       help='Optional source spectrum length before heatmap (if None, uses heatmap_size)')
    parser.add_argument('--target_size', type=int, default=None,
                       help='Optional target spectrum length before heatmap (if None, uses heatmap_size)')
    parser.add_argument('--use_mixed_loss', action='store_true',
                       help='Use mixed loss training (velocity loss + generation loss)')
    parser.add_argument('--gen_loss_weight', type=float, default=0.1,
                       help='Weight for generation loss in mixed loss training')
    parser.add_argument('--gen_loss_prob', type=float, default=0.1,
                       help='Probability of computing generation loss per batch (0.0-1.0, lower = faster training)')
    parser.add_argument('--val_num_steps', type=int, default=50,
                       help='Number of ODE steps for validation (default: 50, faster. Use RK4+more steps in test)')
    parser.add_argument('--train_gen_steps', type=int, default=100,
                       help='Number of ODE steps for generation loss in training (default: 100, Euler method, avoid GPU OOM. Use RK4+more steps in test)')
    parser.add_argument('--use_rk4_in_train', action='store_true',
                       help='Use RK4 in training generation loss (more accurate but uses more GPU memory, may cause OOM)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    # If training on qm9s, save checkpoints into a separate subfolder to avoid collisions with other runs.
    # Users can disable this behavior via --no_dataset_subdir or fully override via --save_dir.
    if (not args.no_dataset_subdir) and ('qm9s' in str(args.data_dir).lower()):
        args.save_dir = os.path.join(args.save_dir, 'qm9s')
        print(f"[save_dir] Detected qm9s dataset. Saving checkpoints to: {args.save_dir}")
    
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
    
    # 配置文件
    class Config:
        in_channels = 1
        hidden_channels = args.hidden_channels
        num_modes = 3
        resize_shape = tuple(args.resize_shape)
        learning_rate = args.learning_rate
        epochs = args.epochs
        batch_size = args.batch_size
        sigma_min = args.sigma_min
    
    config = Config()
    
    # 创建训练器
    trainer = FlowMatchingTrainer(config, device)
    
    # 加载数据
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
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = get_paired_loaders(
        str(source_csv), str(target_csv),
        batch_size=config.batch_size,
        source_size=args.source_size,
        target_size=args.target_size,
        heatmap_size=args.heatmap_size,
        resize_shape=config.resize_shape,
        out_channels=config.in_channels,
        use_h5=True,
        seed=args.seed
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    
    # 训练
    train_modality_pair(
        trainer, train_loader, val_loader,
        args.source_mode, args.target_mode,
        epochs=args.epochs,
        save_dir=args.save_dir,
        use_mixed_loss=args.use_mixed_loss,
        gen_loss_weight=args.gen_loss_weight,
        gen_loss_prob=args.gen_loss_prob,
        val_num_steps=args.val_num_steps,
        train_gen_steps=args.train_gen_steps,
        use_rk4_in_train=args.use_rk4_in_train,  # 默认False，避免GPU内存溢出
        seed=args.seed
    )
    
    print("Training completed!")


if __name__ == '__main__':
    main()

