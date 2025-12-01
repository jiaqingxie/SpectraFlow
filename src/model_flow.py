"""
Flow Matching模型定义：用于跨模态光谱转换
基于Continuous Normalizing Flows (CNF) 和Conditional Flow Matching (CFM)
使用2D U-Net架构，在多个层级注入时间和模态嵌入（FiLM机制）
"""
import torch
import torch.nn as nn
import numpy as np


class SinusoidalPositionalEmbedding(nn.Module):
    """时间步嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM)
    用于在多个层级注入条件信息（时间和模态）
    支持2D特征图
    """
    def __init__(self, embedding_dim, feature_dim):
        super().__init__()
        self.scale = nn.Linear(embedding_dim, feature_dim)
        self.shift = nn.Linear(embedding_dim, feature_dim)
    
    def forward(self, x, embedding):
        """
        Args:
            x: 特征 (B, C, H, W) - 2D卷积特征
            embedding: 条件嵌入 (B, embedding_dim)
        Returns:
            modulated: 调制后的特征
        """
        scale = self.scale(embedding)  # (B, C)
        shift = self.shift(embedding)  # (B, C)
        
        # 扩展到 (B, C, 1, 1) 用于广播
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        
        return scale * x + shift


class VelocityFieldNetwork(nn.Module):
    """
    速度场网络：学习从源光谱到目标光谱的向量场
    基于2D U-Net架构，在多个层级注入时间和模态嵌入（FiLM机制）
    """
    def __init__(self, in_channels=1, hidden_channels=128, num_modes=3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_modes = num_modes
        
        # 时间嵌入维度
        time_dim = 128
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 模态嵌入（用于条件生成）
        self.mode_embed = nn.Embedding(num_modes, time_dim)
        
        # 条件编码器：将x0（源光谱）编码为条件向量
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 压缩空间维度到4x4
            nn.Flatten(),
            nn.Linear(hidden_channels * 16, time_dim),  # 4*4=16
            nn.SiLU()
        )
        
        # 条件嵌入（时间 + 模态 + x0内容）
        condition_dim = time_dim
        
        # 编码器（下采样）- 2D卷积
        # 注意：不再拼接condition，只使用x_t作为输入
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU()
        )
        self.film1 = FiLM(condition_dim, hidden_channels)
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.SiLU()
        )
        self.film2 = FiLM(condition_dim, hidden_channels * 2)
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.SiLU()
        )
        self.film3 = FiLM(condition_dim, hidden_channels * 4)
        
        # 中间层（融合条件嵌入）
        self.middle = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.SiLU()
        )
        self.film_middle = FiLM(condition_dim, hidden_channels * 4)
        
        # 解码器（上采样）- 2D转置卷积
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.SiLU()
        )
        self.film_dec3 = FiLM(condition_dim, hidden_channels * 2)
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU()
        )
        self.film_dec2 = FiLM(condition_dim, hidden_channels)
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, t, condition=None, target_mode=None):
        """
        前向传播
        
        Args:
            x: 当前状态 (B, C, H, W) - 2D热图
            t: 时间步 (B,) - 范围[0, 1]
            condition: 条件输入（源光谱）(B, C, H, W)
            target_mode: 目标模态索引 (0=IR, 1=UV, 2=Raman)
        
        Returns:
            velocity: 速度场预测 (B, C, H, W)
        """
        # 时间嵌入
        t_emb = self.time_embed(t)  # (B, time_dim)
        
        # 模态嵌入（如果提供）
        if target_mode is not None:
            # 将整数转换为Tensor，并扩展到批次维度
            if isinstance(target_mode, (int, torch.int32, torch.int64)):
                batch_size = x.size(0)
                target_mode_tensor = torch.full(
                    (batch_size,), target_mode, 
                    dtype=torch.long, device=x.device
                )
            elif isinstance(target_mode, torch.Tensor):
                target_mode_tensor = target_mode.long().to(x.device)
            else:
                target_mode_tensor = torch.tensor(target_mode, dtype=torch.long, device=x.device)
            
            mode_emb = self.mode_embed(target_mode_tensor)  # (B, time_dim)
            
            # 编码x0内容信息（如果提供）
            if condition is not None:
                x0_emb = self.condition_encoder(condition)  # (B, time_dim)
                condition_emb = t_emb + mode_emb + x0_emb  # 融合时间 + 模态 + x0内容
            else:
                condition_emb = t_emb + mode_emb
        else:
            if condition is not None:
                x0_emb = self.condition_encoder(condition)  # (B, time_dim)
                condition_emb = t_emb + x0_emb
            else:
                condition_emb = t_emb
        
        # 主输入：只用x_t（不再拼接condition）
        # CFM标准做法：条件信息通过FiLM注入，而不是拼接
        x_input = x
        
        # 编码器（带FiLM调制）
        h1 = self.encoder1(x_input)  # (B, hidden_channels, H, W)
        h1 = self.film1(h1, condition_emb)
        
        h2 = self.encoder2(h1)  # (B, hidden_channels*2, H/2, W/2)
        h2 = self.film2(h2, condition_emb)
        
        h3 = self.encoder3(h2)  # (B, hidden_channels*4, H/4, W/4)
        h3 = self.film3(h3, condition_emb)
        
        # 中间层（带FiLM调制）
        h_middle = self.middle(h3)
        h_middle = self.film_middle(h_middle, condition_emb)
        
        # 解码器（带跳跃连接和FiLM调制）
        h3_up = self.decoder3(h_middle)  # (B, hidden_channels*2, H/2, W/2)
        h3_up = self.film_dec3(h3_up, condition_emb)
        h3_up = torch.cat([h3_up, h2], dim=1)  # 跳跃连接
        
        h2_up = self.decoder2(h3_up)  # (B, hidden_channels, H, W)
        h2_up = self.film_dec2(h2_up, condition_emb)
        h2_up = torch.cat([h2_up, h1], dim=1)  # 跳跃连接
        
        velocity = self.decoder1(h2_up)  # (B, C, H, W)
        
        return velocity


class ConditionalFlowMatching(nn.Module):
    """
    条件Flow Matching模型
    用于从源光谱分布转换到目标光谱分布
    使用2D架构
    """
    def __init__(self, in_channels=1, hidden_channels=128, num_modes=3, 
                 image_size=(60, 60), sigma_min=0.01):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_modes = num_modes
        self.image_size = image_size
        self.sigma_min = sigma_min
        
        # 速度场网络（2D版本）
        self.velocity_field = VelocityFieldNetwork(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_modes=num_modes
        )
    
    def sample_t(self, batch_size, device):
        """随机采样时间步"""
        return torch.rand(batch_size, device=device)
    
    def compute_training_target(self, x0, x1, t):
        """
        计算训练目标（速度场）
        使用线性插值路径：x_t = (1-t)*x0 + t*x1
        速度场：v_t = x1 - x0
        
        Args:
            x0: 源分布样本 (B, C, H, W)
            x1: 目标分布样本 (B, C, H, W)
            t: 时间步 (B,)
        
        Returns:
            x_t: 插值点 (B, C, H, W)
            v_t: 真实速度场 (B, C, H, W)
        """
        # 确保t的形状为 (B, 1, 1, 1) 用于广播
        t = t.view(-1, 1, 1, 1)
        
        # 线性插值路径
        x_t = (1 - t) * x0 + t * x1
        
        # 速度场（路径的导数）
        v_t = x1 - x0
        
        return x_t, v_t
    
    def forward(self, x0, x1, t=None, target_mode=None):
        """
        前向传播（训练时）
        
        Args:
            x0: 源光谱 (B, C, H, W)
            x1: 目标光谱 (B, C, H, W)
            t: 时间步 (B,)，如果为None则随机采样
            target_mode: 目标模态索引
        
        Returns:
            v_pred: 预测的速度场 (B, C, H, W)
            v_true: 真实的速度场 (B, C, H, W)
            x_t: 插值点 (B, C, H, W)
        """
        if t is None:
            t = self.sample_t(x0.size(0), x0.device)
        
        # 计算训练目标
        x_t, v_true = self.compute_training_target(x0, x1, t)
        
        # 预测速度场
        v_pred = self.velocity_field(x_t, t, condition=x0, target_mode=target_mode)
        
        return v_pred, v_true, x_t
    
    def sample(self, x0, target_mode=None, num_steps=50, return_path=False, use_rk4=False):
        """
        从源光谱生成目标光谱（推理时）
        支持Euler方法和RK4方法
        
        Args:
            x0: 源光谱 (B, C, H, W)
            target_mode: 目标模态索引
            num_steps: ODE求解步数
            return_path: 是否返回完整路径
            use_rk4: 是否使用RK4方法（更准确但更慢）
        
        Returns:
            x1: 生成的目标光谱 (B, C, H, W)
            path: 完整路径（如果return_path=True）
        """
        if not self.training:
            self.eval()
        
        device = x0.device
        batch_size = x0.size(0)
        
        # 初始状态
        x = x0.clone()
        
        # 时间步
        dt = 1.0 / num_steps
        t = torch.zeros(batch_size, device=device)
        
        path = [x.clone()] if return_path else None
        
        if use_rk4:
            # RK4方法：更准确的ODE求解
            for i in range(num_steps):
                # k1 = v(x, t)
                k1 = self.velocity_field(x, t, condition=x0, target_mode=target_mode)
                k1 = torch.clamp(k1, -2.0, 2.0)
                
                # k2 = v(x + dt/2 * k1, t + dt/2)
                x2 = x + 0.5 * dt * k1
                t2 = t + 0.5 * dt
                k2 = self.velocity_field(x2, t2, condition=x0, target_mode=target_mode)
                k2 = torch.clamp(k2, -2.0, 2.0)
                
                # k3 = v(x + dt/2 * k2, t + dt/2)
                x3 = x + 0.5 * dt * k2
                k3 = self.velocity_field(x3, t2, condition=x0, target_mode=target_mode)
                k3 = torch.clamp(k3, -2.0, 2.0)
                
                # k4 = v(x + dt * k3, t + dt)
                x4 = x + dt * k3
                t4 = t + dt
                k4 = self.velocity_field(x4, t4, condition=x0, target_mode=target_mode)
                k4 = torch.clamp(k4, -2.0, 2.0)
                
                # 更新状态
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                t = t + dt
                
                # 中间步骤裁剪
                if i < num_steps - 1:
                    x = torch.clamp(x, -0.1, 1.1)
                
                if return_path:
                    path.append(x.clone())
        else:
            # Euler方法求解ODE: dx/dt = v_θ(x, t)
            for i in range(num_steps):
                # 预测速度场
                v = self.velocity_field(x, t, condition=x0, target_mode=target_mode)
                
                # 更新状态（使用更稳定的更新方式）
                # 限制速度场的大小，避免数值不稳定
                v = torch.clamp(v, -2.0, 2.0)
                x = x + dt * v
                
                # 更新时间
                t = t + dt
                
                # 中间步骤也进行裁剪，保持数值稳定
                if i < num_steps - 1:  # 最后一步不裁剪，让模型自由输出
                    x = torch.clamp(x, -0.1, 1.1)  # 允许稍微超出范围
                
                if return_path:
                    path.append(x.clone())
        
        # 确保输出在合理范围内
        x = torch.clamp(x, 0, 1)
        
        if return_path:
            return x, path
        return x
