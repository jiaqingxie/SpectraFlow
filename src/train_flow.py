"""
训练Flow Matching模型，实现IR、UV、Raman之间的光谱转换
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader
import numpy as np
import argparse
import contextlib
import os
from pathlib import Path
from tqdm import tqdm

from model_flow import ConditionalFlowMatching
from train import PairedModalDataset, get_paired_loaders
from utils import get_device


def _flatten_spectrum_batch(x):
    """Flatten heatmap batches to per-sample 1D spectra for spectral losses."""
    return x.reshape(x.size(0), -1)


def first_derivative(x):
    dx = x[:, 1:] - x[:, :-1]
    return F.pad(dx, (1, 0), mode='constant', value=0.0)


def second_derivative(x):
    if x.shape[1] < 3:
        return torch.zeros_like(x)
    d2 = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    return F.pad(d2, (1, 1), mode='constant', value=0.0)


def normalize_by_sample_max(x, eps=1e-8):
    return x / (x.amax(dim=1, keepdim=True) + eps)


def spectral_importance_weight(target, amp_weight=1.0, grad_weight=0.5, curv_weight=0.5, eps=1e-8):
    """Weight high-amplitude, high-slope, and high-curvature spectral regions."""
    with torch.no_grad():
        amp = normalize_by_sample_max(target.abs(), eps)
        grad = normalize_by_sample_max(first_derivative(target).abs(), eps)
        curv = normalize_by_sample_max(second_derivative(target).abs(), eps)
        return 1.0 + amp_weight * amp + grad_weight * grad + curv_weight * curv


def weighted_elastic_loss(pred, target, weight=None, alpha_l1=0.6):
    err = pred - target
    loss = alpha_l1 * err.abs() + (1.0 - alpha_l1) * err.pow(2)
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def derivative_shape_loss(pred, target, curv_coef=0.5):
    return (first_derivative(pred) - first_derivative(target)).abs().mean() + curv_coef * (
        second_derivative(pred) - second_derivative(target)
    ).abs().mean()


def local_ot_loss(pred, target, window_size=64, eps=1e-8):
    """Windowed 1D Wasserstein-1 distance using cumulative distributions."""
    b, length = pred.shape
    pad = (window_size - length % window_size) % window_size
    if pad > 0:
        pred = F.pad(pred, (0, pad), mode='constant', value=0.0)
        target = F.pad(target, (0, pad), mode='constant', value=0.0)
    pred = F.relu(pred).reshape(b, -1, window_size) + eps
    target = F.relu(target).reshape(b, -1, window_size) + eps
    pred = pred / pred.sum(dim=-1, keepdim=True)
    target = target / target.sum(dim=-1, keepdim=True)
    return torch.cumsum(pred - target, dim=-1).abs().mean()


def nonnegative_penalty(x):
    return F.relu(-x).pow(2).mean()


class FlowMatchingTrainer:
    """Flow Matching训练器"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.train_steps = 0
        self.use_amp = bool(getattr(config, 'use_amp', False)) and device.type == 'cuda'
        self.scaler = amp.GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            print('[train] AMP (fp16) enabled for velocity + gen loss (torch.amp)')
        
        # 创建模型（支持 unet / dit / vibradit 三种 backbone）
        self.model = ConditionalFlowMatching(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            num_modes=config.num_modes,
            image_size=config.resize_shape,
            sigma_min=config.sigma_min,
            backbone=getattr(config, 'backbone', 'unet'),
            dit_hidden_dim=getattr(config, 'dit_hidden_dim', 256),
            dit_depth=getattr(config, 'dit_depth', 6),
            dit_num_heads=getattr(config, 'dit_num_heads', 4),
            dit_patch_size=getattr(config, 'dit_patch_size', 4),
        ).to(device)
        print(f"[model] requested backbone: {getattr(config, 'backbone', 'unet')}")
        print(f"[model] actual velocity_field: {self.model.velocity_field.__class__.__name__}")
        print(f"[model] actual model.backbone: {getattr(self.model, 'backbone', 'unknown')}")
        
        # 权重初始化（支持1D/2D卷积和Linear）
        # 注意：DiT/VibraDiT 里 adaLN / final projection 需要保持零初始化以保证训练稳定，
        # 不能被统一Xavier覆盖。
        zero_init_backbone = getattr(config, 'backbone', 'unet') in {'dit', 'vibradit'}
        skip_zero_init_names = (
            "velocity_field.final_adaLN",
            "velocity_field.out_proj",
            "velocity_field.final_layer.adaLN_modulation",
            "velocity_field.final_layer.linear",
        )
        for name, m in self.model.named_modules():
            if not isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                continue
            is_adaln_zero = ".adaLN" in name or ".adaLN_modulation" in name
            is_final_zero = any(name.startswith(prefix) for prefix in skip_zero_init_names)
            if zero_init_backbone and (is_adaln_zero or is_final_zero):
                continue
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        
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
    
    def _autocast_ctx(self):
        """CUDA 上用 torch.amp；CPU 上不用 autocast（避免无 CUDA 时 device_type='cuda' 出问题）。"""
        if self.device.type != 'cuda':
            return contextlib.nullcontext()
        return amp.autocast(device_type='cuda', enabled=self.use_amp)
    
    def train_step(self, source_batch, target_batch, target_mode, use_mixed_loss=False,
                   gen_loss_weight=0.1, compute_gen_prob=0.1, train_gen_steps=100,
                   use_rk4_in_train=False, gen_max_samples=0, use_vibradit_losses=False,
                   amp_weight=1.0, grad_weight=0.5, curv_weight=0.5,
                   lambda_shape=0.05, lambda_ot=0.02, lambda_pos=0.01,
                   ot_window_size=64):
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
            gen_max_samples: >0 时仅对前 K 个样本算 gen loss（velocity 仍用整批），高 gen_prob 时显著加速
        """
        self.model.train()
        self.train_steps += 1
        
        source_batch = source_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        target_mode_idx = self.mode_map[target_mode]
        
        # 前向传播
        with self._autocast_ctx():
            v_pred, v_true, x_t = self.model(
                x0=source_batch,
                x1=target_batch,
                t=None,  # 随机采样时间步
                target_mode=target_mode_idx
            )
        
        # Flow Matching损失：在 fp32 上算子损失，避免 fp16 loss 与 AMP 反传 dtype 冲突
        if use_vibradit_losses:
            target_flat = _flatten_spectrum_batch(target_batch.float())
            weight = spectral_importance_weight(target_flat, amp_weight, grad_weight, curv_weight)
            velocity_loss = weighted_elastic_loss(
                _flatten_spectrum_batch(v_pred.float()),
                _flatten_spectrum_batch(v_true.float()),
                weight=weight,
                alpha_l1=0.6,
            )
        else:
            l1_loss = F.l1_loss(v_pred.float(), v_true.float())
            l2_loss = F.mse_loss(v_pred.float(), v_true.float())
            velocity_loss = 0.6 * l1_loss + 0.4 * l2_loss
        
        # 如果使用混合损失，同时优化生成损失
        # 但只在部分batch计算（通过概率控制）以加速训练
        gen_loss = None
        total_loss = velocity_loss
        
        if use_mixed_loss and np.random.rand() < compute_gen_prob:
            if gen_max_samples and gen_max_samples > 0:
                k = min(gen_max_samples, source_batch.size(0))
                s_gen = source_batch[:k]
                t_gen = target_batch[:k]
            else:
                s_gen = source_batch
                t_gen = target_batch
            # ODE 多步展开在 AMP 下易出现 “Found dtype Float but expected Half”；训练时强制 fp32 推理链
            if self.device.type == 'cuda':
                with amp.autocast(device_type='cuda', enabled=False):
                    generated = self.model.sample(
                        s_gen,
                        target_mode=target_mode_idx,
                        num_steps=train_gen_steps,
                        use_rk4=use_rk4_in_train
                    )
            else:
                generated = self.model.sample(
                    s_gen,
                    target_mode=target_mode_idx,
                    num_steps=train_gen_steps,
                    use_rk4=use_rk4_in_train
                )
            if use_vibradit_losses:
                gen_flat = _flatten_spectrum_batch(generated.float())
                target_flat = _flatten_spectrum_batch(t_gen.float())
                gen_weight = spectral_importance_weight(target_flat, amp_weight, grad_weight, curv_weight)
                gen_loss = weighted_elastic_loss(gen_flat, target_flat, weight=gen_weight, alpha_l1=0.6)
                if lambda_shape > 0:
                    gen_loss = gen_loss + lambda_shape * derivative_shape_loss(gen_flat, target_flat)
                if lambda_ot > 0:
                    gen_loss = gen_loss + lambda_ot * local_ot_loss(gen_flat, target_flat, window_size=ot_window_size)
                if lambda_pos > 0:
                    gen_loss = gen_loss + lambda_pos * nonnegative_penalty(gen_flat)
            else:
                gen_l1 = F.l1_loss(generated.float(), t_gen.float())
                gen_l2 = F.mse_loss(generated.float(), t_gen.float())
                gen_loss = 0.6 * gen_l1 + 0.4 * gen_l2
            
            # 混合损失：velocity loss + generation loss
            # 注意：由于只在小部分batch计算，需要调整权重
            total_loss = velocity_loss + (gen_loss_weight / compute_gen_prob) * gen_loss
        
        # 反向传播（total_loss 已为 fp32；勿再对 loss 做 .float() 以免破坏计算图与 dtype）
        self.optimizer.zero_grad()
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
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
        with self._autocast_ctx():
            v_pred, v_true, x_t = self.model(
                x0=source_batch,
                x1=target_batch,
                t=None,
                target_mode=target_mode_idx
            )
        
        # 损失（与训练时一致，fp32 子损失）
        l1_loss = F.l1_loss(v_pred.float(), v_true.float())
        l2_loss = F.mse_loss(v_pred.float(), v_true.float())
        loss = 0.6 * l1_loss + 0.4 * l2_loss
        
        # 生成样本（与 train_step 一致：ODE 在 fp32 下展开）
        if self.device.type == 'cuda':
            with amp.autocast(device_type='cuda', enabled=False):
                generated = self.model.sample(
                    source_batch,
                    target_mode=target_mode_idx,
                    num_steps=num_steps,
                    use_rk4=use_rk4
                )
        else:
            generated = self.model.sample(
                source_batch,
                target_mode=target_mode_idx,
                num_steps=num_steps,
                use_rk4=use_rk4
            )
        
        gen_l1 = F.l1_loss(generated.float(), target_batch.float())
        gen_l2 = F.mse_loss(generated.float(), target_batch.float())
        gen_loss = 0.6 * gen_l1 + 0.4 * gen_l2
        
        return {
            'loss': loss.item(),
            'gen_loss': gen_loss.item(),
            'generated': generated.cpu(),
            'target': target_batch.cpu()
        }
    
    def save_checkpoint(self, path):
        """保存检查点"""
        payload = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_steps': self.train_steps,
        }
        if self.scaler is not None:
            payload['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(payload, path)
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.train_steps = checkpoint.get('train_steps', 0)


def train_modality_pair(trainer, train_loader, val_loader, source_mode, target_mode,
                       epochs=20, save_dir='checkpoints', use_mixed_loss=False,
                       gen_loss_weight=0.1, gen_loss_prob=0.1, val_num_steps=50,
                       train_gen_steps=100, use_rk4_in_train=False, gen_max_samples=0,
                       seed=42, backbone='unet', use_vibradit_losses=False,
                       amp_weight=1.0, grad_weight=0.5, curv_weight=0.5,
                       lambda_shape=0.05, lambda_ot=0.02, lambda_pos=0.01,
                       ot_window_size=64):
    """训练单个模态对"""
    os.makedirs(save_dir, exist_ok=True)
    # 使用val_gen_loss作为模型选择标准，因为它更直接反映最终生成质量
    best_val_gen_loss = float('inf')
    
    print(f"\nTraining Flow Matching: {source_mode} -> {target_mode}")
    if use_mixed_loss:
        print(f"Using mixed loss training (gen_loss_weight={gen_loss_weight}, gen_prob={gen_loss_prob})")
        if gen_max_samples and gen_max_samples > 0:
            print(f"  gen_max_samples={gen_max_samples} (gen loss only on first K samples per batch)")
    if use_vibradit_losses:
        print(
            "Using VibraDiT spectral losses "
            f"(amp={amp_weight}, grad={grad_weight}, curv={curv_weight}, "
            f"shape={lambda_shape}, ot={lambda_ot}, pos={lambda_pos})"
        )
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
                use_rk4_in_train=use_rk4_in_train,
                gen_max_samples=gen_max_samples,
                use_vibradit_losses=use_vibradit_losses,
                amp_weight=amp_weight,
                grad_weight=grad_weight,
                curv_weight=curv_weight,
                lambda_shape=lambda_shape,
                lambda_ot=lambda_ot,
                lambda_pos=lambda_pos,
                ot_window_size=ot_window_size,
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
            # 使用“实际模型backbone”命名，避免参数传递异常导致“文件名是dit、内容是unet”
            actual_backbone = getattr(trainer.model, 'backbone', backbone)
            backbone_tag = f'_{actual_backbone}' if actual_backbone != 'unet' else ''
            checkpoint_path = os.path.join(
                save_dir,
                f'flow_{source_mode}2{target_mode}{backbone_tag}_best_seed{seed}.pt'
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
    parser.add_argument('--val_source_csv', type=str, default=None,
                       help='Optional explicit validation source CSV/H5 stem')
    parser.add_argument('--val_target_csv', type=str, default=None,
                       help='Optional explicit validation target CSV/H5 stem')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                       help='Initialize model weights from a checkpoint without restoring optimizer state')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--train_fraction', type=float, default=0.7,
                       help='Fraction of the provided training file used for optimization')
    parser.add_argument('--val_fraction', type=float, default=0.15,
                       help='Fraction of the provided training file used for validation')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='Learning rate (aligned with VAE, more stable)')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--backbone', type=str, default='unet', choices=['unet', 'dit', 'vibradit'],
                       help='Backbone network: unet (default), dit (2D DiT), or vibradit (1D spectral DiT)')
    parser.add_argument('--dit_hidden_dim', type=int, default=256,
                       help='[DiT only] Transformer hidden dimension (default: 256)')
    parser.add_argument('--dit_depth', type=int, default=6,
                       help='[DiT only] Number of DiT blocks (default: 6)')
    parser.add_argument('--dit_num_heads', type=int, default=4,
                       help='[DiT only] Number of attention heads (default: 4)')
    parser.add_argument('--dit_patch_size', type=int, default=4,
                       help='[DiT only] Patch size for patchify (default: 4). image_size must be divisible by this.')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint directory (base)')
    parser.add_argument('--no_dataset_subdir', action='store_true',
                       help='Do not auto-create a dataset-specific subdirectory under save_dir (default: False). '
                            'By default, if --data_dir path contains "qm9s", checkpoints will be saved under save_dir/qm9s/.')
    parser.add_argument('--sigma_min', type=float, default=0.01, help='Minimum noise level')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--cpu_threads', type=int, default=0,
                       help='Limit PyTorch CPU and inter-op threads; 0 keeps the environment default')
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
    parser.add_argument('--preserve_spectral_order', action='store_true',
                       help='Reshape spectra directly without patch-wise index reordering')
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
    parser.add_argument('--no_amp', action='store_true',
                       help='Disable mixed precision (fp16) training on CUDA (default: AMP on when using GPU)')
    parser.add_argument('--gen_max_samples', type=int, default=0,
                       help='If >0, only first K samples in each batch get gen loss (velocity still full batch). '
                            'Use with high --gen_loss_prob (e.g. 0.5) to cut ODE cost. 0 = all samples (default).')
    parser.add_argument('--use_vibradit_losses', action='store_true',
                       help='Use VibraDiT spectral weighted/shape/OT losses. Enabled automatically for --backbone vibradit.')
    parser.add_argument('--amp_weight', type=float, default=1.0,
                       help='[VibraDiT loss] Weight for high-amplitude spectral regions')
    parser.add_argument('--grad_weight', type=float, default=0.5,
                       help='[VibraDiT loss] Weight for high-slope spectral regions')
    parser.add_argument('--curv_weight', type=float, default=0.5,
                       help='[VibraDiT loss] Weight for high-curvature spectral regions')
    parser.add_argument('--lambda_shape', type=float, default=0.05,
                       help='[VibraDiT loss] Endpoint derivative shape loss weight')
    parser.add_argument('--lambda_ot', type=float, default=0.02,
                       help='[VibraDiT loss] Endpoint local OT loss weight')
    parser.add_argument('--lambda_pos', type=float, default=0.01,
                       help='[VibraDiT loss] Endpoint nonnegative penalty weight')
    parser.add_argument('--ot_window_size', type=int, default=64,
                       help='[VibraDiT loss] Local OT window size')
    parser.add_argument('--cudnn_benchmark', action='store_true',
                       help='Enable cudnn.benchmark for faster convs (less strict reproducibility)')
    args = parser.parse_args()

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(args.cpu_threads)
        except RuntimeError:
            pass
        print(f"[train] CPU thread pools limited to {args.cpu_threads}")

    # If training on qm9s, save checkpoints into a separate subfolder to avoid collisions with other runs.
    # Users can disable this behavior via --no_dataset_subdir or fully override via --save_dir.
    # Check both data_dir and source/target CSV paths for 'qm9s'
    is_qm9s = False
    if not args.no_dataset_subdir:
        if 'qm9s' in str(args.data_dir).lower():
            is_qm9s = True
        elif args.source_csv and 'qm9' in str(args.source_csv).lower():
            is_qm9s = True
        elif args.target_csv and 'qm9' in str(args.target_csv).lower():
            is_qm9s = True
    
    if is_qm9s:
        args.save_dir = os.path.join(args.save_dir, 'qm9s')
        print(f"[save_dir] Detected qm9s dataset. Saving checkpoints to: {args.save_dir}")
    
    # 设置随机种子以确保结果可复现
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        if args.cudnn_benchmark:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            print('[train] cudnn.benchmark=True (faster convs, not fully deterministic)')
        else:
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
        backbone = args.backbone
        dit_hidden_dim = args.dit_hidden_dim
        dit_depth = args.dit_depth
        dit_num_heads = args.dit_num_heads
        dit_patch_size = args.dit_patch_size
        use_amp = (not args.no_amp) and (device.type == 'cuda')
    
    config = Config()
    
    # 创建训练器
    trainer = FlowMatchingTrainer(config, device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[fine-tune] Initialized model weights from: {args.init_checkpoint}")
    
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
    if bool(args.val_source_csv) != bool(args.val_target_csv):
        parser.error("--val_source_csv and --val_target_csv must be provided together")
    if args.val_source_csv:
        val_source_csv = (
            Path(args.val_source_csv)
            if os.path.isabs(args.val_source_csv)
            else data_dir / args.val_source_csv
        )
        val_target_csv = (
            Path(args.val_target_csv)
            if os.path.isabs(args.val_target_csv)
            else data_dir / args.val_target_csv
        )
        for path in (val_source_csv, val_target_csv):
            if not path.exists() and not path.with_suffix('.h5').exists():
                raise FileNotFoundError(path)
        dataset_kwargs = dict(
            source_size=args.source_size,
            target_size=args.target_size,
            heatmap_size=args.heatmap_size,
            resize_shape=config.resize_shape,
            out_channels=config.in_channels,
            use_h5=True,
            preserve_spectral_order=args.preserve_spectral_order,
        )
        train_dataset = PairedModalDataset(
            str(source_csv), str(target_csv), **dataset_kwargs
        )
        val_dataset = PairedModalDataset(
            str(val_source_csv), str(val_target_csv), **dataset_kwargs
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False
        )
        print("[split] Using explicit train and validation datasets.")
    else:
        train_loader, val_loader, _ = get_paired_loaders(
            str(source_csv), str(target_csv),
            batch_size=config.batch_size,
            source_size=args.source_size,
            target_size=args.target_size,
            heatmap_size=args.heatmap_size,
            resize_shape=config.resize_shape,
            out_channels=config.in_channels,
            use_h5=True,
            seed=args.seed,
            preserve_spectral_order=args.preserve_spectral_order,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction
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
        gen_max_samples=args.gen_max_samples,
        seed=args.seed,
        backbone=getattr(args, 'backbone', 'unet'),
        use_vibradit_losses=args.use_vibradit_losses or args.backbone == 'vibradit',
        amp_weight=args.amp_weight,
        grad_weight=args.grad_weight,
        curv_weight=args.curv_weight,
        lambda_shape=args.lambda_shape,
        lambda_ot=args.lambda_ot,
        lambda_pos=args.lambda_pos,
        ot_window_size=args.ot_window_size,
    )
    
    print("Training completed!")


if __name__ == '__main__':
    main()

