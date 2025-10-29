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
    """
    def __init__(self, in_channels=1, hidden_channels=128, latent_dim=128, 
                 image_size=(60, 60), num_modes=3):
        super(CrossModalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.num_modes = num_modes

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
        
        # 均值和对数方差
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

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
                )
            ])
            self.decoders.append(nn.ModuleList(decoder))

    def encode(self, x):
        """编码输入到潜在空间"""
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
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
        h = decoder_modules[2](h)   # ConvTranspose layers
        return h

    def forward(self, x, target_mode=0, return_latent=False):
        """
        前向传播
        
        Args:
            x: 输入光谱 (B, C, H, W)
            target_mode: 目标模态索引 (0=IR, 1=UV, 2=Raman)
            return_latent: 是否返回潜在变量
        
        Returns:
            recon: 重建的光谱
            mu: 潜在空间均值
            logvar: 潜在空间对数方差
            z: 潜在变量（如果return_latent=True）
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, target_mode)
        
        if return_latent:
            return recon, mu, logvar, z
        return recon, mu, logvar


