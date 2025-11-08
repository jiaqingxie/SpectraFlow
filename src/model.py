"""
模型定义模块：跨模态VAE模型
"""
import torch
import torch.nn as nn
import numpy as np


class CrossModalVAE(nn.Module):
    """
    跨模态VAE模型，用于光谱模态之间的转换
    支持IR、UV、Raman之间的互转
    加入物理先验：将物理参数作为额外特征融合进编码器
    """
    def __init__(self, in_channels=1, hidden_channels=128, latent_dim=128, 
                 image_size=(60, 60), num_modes=3, use_physical_prior=False, 
                 physical_dim=7, clamp_output=True):
        super(CrossModalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.num_modes = num_modes
        self.use_physical_prior = use_physical_prior
        self.physical_dim = physical_dim
        self.clamp_output = clamp_output

        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, 2, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(),
        )

        self.feature_shape = (hidden_channels * 2, image_size[0] // 4, image_size[1] // 4)
        flat_dim = np.prod(self.feature_shape)
        
        # 物理先验处理网络（类似参考代码的fc1_seq）
        if use_physical_prior:
            # 物理参数维度 -> 特征维度（去掉BN，避免小batch导致NaN）
            self.fc_physical = nn.Linear(physical_dim, flat_dim)
        
        # 均值和对数方差
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        
        # 与SpectrumViT一致：使用默认初始化

        # 模态特定的解码器
        self.decoders = nn.ModuleList()
        for _ in range(num_modes):
            decoder = nn.ModuleList([
                nn.Linear(latent_dim, flat_dim),
                nn.Unflatten(1, self.feature_shape),
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, 2, 1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(),
                    nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 3, 1, 1),
                    nn.BatchNorm2d(hidden_channels // 2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(hidden_channels // 2, in_channels, 4, 2, 1),
                    # 注意：SpectrumViT使用了BatchNorm2d(hidden_channels // 2)，但这会导致维度不匹配
                    # 暂时移除，在decode中使用clamp代替
                )
            ])
            self.decoders.append(nn.ModuleList(decoder))

    def encode(self, x, physical_params=None):
        """
        编码输入到潜在空间
        Args:
            x: 输入图像 (B, C, H, W)
            physical_params: 物理参数 (B, physical_dim)
        """
        h = self.encoder(x)
        # 防御性处理，避免 BN/数值导致的 NaN/Inf 传播
        h = torch.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6)
        h_flat = h.view(h.size(0), -1)
        
        # 融合物理先验（如果启用）
        if self.use_physical_prior and physical_params is not None:
            # 处理物理参数：physical_dim -> flat_dim
            h_physical = self.fc_physical(physical_params)
            h_physical = torch.relu(h_physical)
            
            # 融合：图像特征 + 物理特征（相加）
            h_flat = h_flat + h_physical
            h_flat = torch.nan_to_num(h_flat, nan=0.0, posinf=1e6, neginf=-1e6)
        
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        # 防止 logvar 数值爆炸造成 NaN
        logvar = torch.clamp(logvar, -5.0, 5.0)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        logvar = torch.nan_to_num(logvar, nan=0.0, posinf=10.0, neginf=-10.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, target_mode):
        """解码潜在变量到指定模态"""
        decoder_modules = self.decoders[target_mode]
        h = decoder_modules[0](z)  # Linear
        h = decoder_modules[1](h)   # Unflatten
        h = decoder_modules[2](h)   # ConvTranspose layers (最后层无BatchNorm)
        # 可选：使用clamp确保输出在[0,1]范围，避免NaN
        if self.clamp_output:
            h = torch.clamp(h, 0, 1)
        return h

    def forward(self, x, target_mode=0, physical_params=None, return_latent=False):
        """
        前向传播
        
        Args:
            x: 输入光谱 (B, C, H, W)
            target_mode: 目标模态索引 (0=IR, 1=UV, 2=Raman)
            physical_params: 物理参数 (B, physical_dim)
            return_latent: 是否返回潜在变量
        
        Returns:
            recon: 重建的光谱
            mu: 潜在空间均值
            logvar: 潜在空间对数方差
            z: 潜在变量（如果return_latent=True）
        """
        mu, logvar = self.encode(x, physical_params)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, target_mode)
        
        if return_latent:
            return recon, mu, logvar, z
        return recon, mu, logvar



