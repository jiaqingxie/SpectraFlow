"""
训练Flow Matching模型（1D版本），从Parquet文件加载数据
支持从IR光谱生成H-NMR和C-NMR光谱
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

from model_flow_1d import ConditionalFlowMatching1D
from dataset_parquet import get_parquet_loaders
from utils import get_device


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

    def train_step(self, source_batch, target_batch, target_mode=None):
        """
        训练一步

        Args:
            source_batch: 源模态数据 (B, 1, L_source) - 1D光谱
            target_batch: 目标模态数据 (B, 1, L_target) - 1D光谱
            target_mode: 目标模态索引（对于NMR任务，可以设为None或固定值）
        """
        self.model.train()
        self.train_steps += 1

        source_batch = source_batch.to(self.device)
        target_batch = target_batch.to(self.device)

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


def train_modality_pair(trainer, train_loader, val_loader, source_field, target_field,
                       epochs=20, save_dir='checkpoints', val_num_steps=50, seed=42):
    """训练单个模态对"""
    os.makedirs(save_dir, exist_ok=True)
    best_val_gen_loss = float('inf')

    print(f"\nTraining Flow Matching (1D): {source_field} -> {target_field}")
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
            safe_target = target_field.replace('_', '-')
            checkpoint_path = os.path.join(
                save_dir,
                f'flow_1d_ir2{safe_target}_best_seed{seed}.pt'
            )
            trainer.save_checkpoint(checkpoint_path)
            print(f"  -> Saved best model (val_gen_loss: {best_val_gen_loss:.6f})")

    print(f"Training completed. Best val gen loss: {best_val_gen_loss:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Flow Matching Model (1D) from Parquet')
    parser.add_argument('--data_dir', type=str,
                       default='/mnt/shared-storage-user/xiejiaqing/datasets/multimodal-spectroscopic-dataset',
                       help='Directory containing parquet files')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='Learning rate')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--sigma_min', type=float, default=0.01, help='Minimum noise level')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--target_field', type=str, required=True,
                       choices=['h_nmr_spectra', 'c_nmr_spectra'],
                       help='Target spectrum field (h_nmr_spectra or c_nmr_spectra)')
    parser.add_argument('--source_field', type=str, default='ir_spectra',
                       help='Source spectrum field (default: ir_spectra)')
    parser.add_argument('--source_size', type=int, default=None,
                       help='Source spectrum length. If None, use original length.')
    parser.add_argument('--target_size', type=int, default=None,
                       help='Target spectrum length. If None, use original length.')
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
    print(f"Source field: {args.source_field}")
    print(f"Target field: {args.target_field}")
    print(f"Source size: {args.source_size}")
    print(f"Target size: {args.target_size}")

    # 配置文件
    class Config:
        in_channels = 1
        hidden_channels = args.hidden_channels
        num_modes = 1  # 对于NMR任务，可以设为1（不需要模态嵌入）
        learning_rate = args.learning_rate
        epochs = args.epochs
        batch_size = args.batch_size
        sigma_min = args.sigma_min

    config = Config()

    # 创建训练器
    trainer = FlowMatchingTrainer1D(config, device)

    # 检查数据路径
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_path}")
        return

    # 创建数据加载器
    print(f"\nLoading data from: {data_path}")
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

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    # 训练
    train_modality_pair(
        trainer, train_loader, val_loader,
        args.source_field, args.target_field,
        epochs=args.epochs,
        save_dir=args.save_dir,
        val_num_steps=args.val_num_steps,
        seed=args.seed
    )

    print("Training completed!")


if __name__ == '__main__':
    main()
