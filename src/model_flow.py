"""
Flow Matching模型定义：用于跨模态光谱转换
基于Continuous Normalizing Flows (CNF) 和Conditional Flow Matching (CFM)
支持两种骨干网络：
  - backbone='unet': 2D U-Net架构，在多个层级注入FiLM条件
  - backbone='dit' : Diffusion Transformer (DiT)，使用adaptive LayerNorm (adaLN)条件
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ─────────────────────────────────────────────────────────────────────────────
# DiT backbone
# ─────────────────────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    DiT Transformer Block with adaptive Layer Norm (adaLN) conditioning.

    Condition vector c modulates each sub-layer via:
        scale, shift, gate  (predicted by a small MLP from c)
    which is equivalent to FiLM + gating, matching the original DiT paper.
    """
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )

        # 每个块预测 6 个 adaLN 参数：shift_sa, scale_sa, gate_sa,
        #                              shift_mlp, scale_mlp, gate_mlp
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        # 零初始化输出端（更稳定的训练起点）
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token序列 (B, N, D)
            c: 条件嵌入  (B, D)
        Returns:
            x: (B, N, D)
        """
        params = self.adaLN(c)                    # (B, 6*D)
        (shift_sa, scale_sa, gate_sa,
         shift_mlp, scale_mlp, gate_mlp) = params.chunk(6, dim=-1)  # each (B, D)

        # Self-attention sub-layer
        h = self.norm1(x)
        h = h * (1 + scale_sa.unsqueeze(1)) + shift_sa.unsqueeze(1)
        attn_out, _ = self.attn(h, h, h)
        x = x + gate_sa.unsqueeze(1) * attn_out

        # MLP sub-layer
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)

        return x


class DiTVelocityFieldNetwork(nn.Module):
    """
    Diffusion Transformer (DiT) 速度场网络。

    流程：
        1. Patchify：Conv2d with stride=patch_size → (B, N, hidden_dim) token序列
        2. 加可学习位置编码
        3. 计算条件嵌入 c = Emb_t(t) + Emb_m(m) + Enc(x0)
        4. 经过 depth 个 DiTBlock（每块以 c 做 adaLN 调制）
        5. Final norm + linear → 重组回 (B, C, H, W) 速度场

    参数与 VelocityFieldNetwork 完全兼容（共享相同的外部接口）。
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 128,   # 对应 U-Net 的 hidden_channels，DiT 用作 condition_dim
        num_modes: int = 3,
        image_size: tuple = (60, 60),
        patch_size: int = 4,
        dit_hidden_dim: int = 256,    # Transformer 隐藏维度
        depth: int = 6,               # DiT 块数量
        num_heads: int = 4,           # 注意力头数
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.in_channels   = in_channels
        self.patch_size    = patch_size
        self.image_size    = image_size
        self.dit_hidden_dim = dit_hidden_dim

        H, W = image_size
        assert H % patch_size == 0 and W % patch_size == 0, \
            f"image_size {image_size} must be divisible by patch_size {patch_size}"
        self.num_patches_h = H // patch_size
        self.num_patches_w = W // patch_size
        self.num_patches   = self.num_patches_h * self.num_patches_w
        patch_dim          = in_channels * patch_size * patch_size  # 输出 patch 的像素数

        # ── 条件嵌入（与 U-Net 版完全相同） ──────────────────────────────
        time_dim = dit_hidden_dim
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.mode_embed = nn.Embedding(num_modes, time_dim)
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(hidden_channels * 16, time_dim),
            nn.SiLU(),
        )

        # ── Patch Embedding ────────────────────────────────────────────────
        # x_t: (B, C, H, W) → (B, N, dit_hidden_dim)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dit_hidden_dim,
                      kernel_size=patch_size, stride=patch_size),  # (B, D, nh, nw)
        )
        # 可学习位置编码 (1, N, D)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, dit_hidden_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Transformer Blocks ─────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            DiTBlock(dit_hidden_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # ── 最终输出层 ────────────────────────────────────────────────────
        self.final_norm = nn.LayerNorm(dit_hidden_dim, elementwise_affine=False, eps=1e-6)
        # adaLN for final norm
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dit_hidden_dim, 2 * dit_hidden_dim),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        # 输出 projection: D → patch_dim，然后 unpatchify
        self.out_proj = nn.Linear(dit_hidden_dim, patch_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # ──────────────────────────────────────────────────────────────────────
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, D)"""
        x = self.patch_embed(x)               # (B, D, nh, nw)
        B, D, nh, nw = x.shape
        x = x.flatten(2).transpose(1, 2)      # (B, N, D)
        return x

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, patch_dim) → (B, C, H, W)"""
        B, N, _ = x.shape
        p = self.patch_size
        C = self.in_channels
        nh, nw = self.num_patches_h, self.num_patches_w

        # x: (B, N, C*p*p)
        x = x.reshape(B, nh, nw, C, p, p)
        # → (B, C, nh, p, nw, p) → (B, C, H, W)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, C, nh * p, nw * p)
        return x

    def _build_condition(self, x: torch.Tensor,
                         t: torch.Tensor,
                         condition: torch.Tensor | None,
                         target_mode) -> torch.Tensor:
        """与 VelocityFieldNetwork 完全相同的条件嵌入构建逻辑"""
        t_emb = self.time_embed(t)              # (B, D)

        if target_mode is not None:
            if isinstance(target_mode, (int,)):
                target_mode_tensor = torch.full(
                    (x.size(0),), target_mode, dtype=torch.long, device=x.device)
            elif isinstance(target_mode, torch.Tensor):
                target_mode_tensor = target_mode.long().to(x.device)
            else:
                target_mode_tensor = torch.tensor(
                    target_mode, dtype=torch.long, device=x.device)
            mode_emb = self.mode_embed(target_mode_tensor)   # (B, D)
        else:
            mode_emb = torch.zeros_like(t_emb)

        if condition is not None:
            x0_emb = self.condition_encoder(condition)       # (B, D)
            c = t_emb + mode_emb + x0_emb
        else:
            c = t_emb + mode_emb
        return c                                              # (B, D)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                condition=None, target_mode=None) -> torch.Tensor:
        """
        Args:
            x: 当前状态 (B, C, H, W)
            t: 时间步   (B,)
            condition: 源光谱热图 (B, C, H, W)，可 None
            target_mode: 目标模态索引
        Returns:
            velocity: 速度场 (B, C, H, W)
        """
        # 1. 条件嵌入
        c = self._build_condition(x, t, condition, target_mode)   # (B, D)

        # 2. Patchify + 位置编码
        tokens = self._patchify(x) + self.pos_embed               # (B, N, D)

        # 3. DiT Blocks
        for blk in self.blocks:
            tokens = blk(tokens, c)

        # 4. Final adaLN
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)       # (B, D) each
        tokens = self.final_norm(tokens)
        tokens = tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # 5. Output projection + Unpatchify
        tokens = self.out_proj(tokens)                             # (B, N, C*p*p)
        velocity = self._unpatchify(tokens)                        # (B, C, H, W)
        return velocity


def get_1d_sincos_pos_embed(num_patches: int, embed_dim: int) -> torch.Tensor:
    """Fixed 1D sin/cos position embedding, returned as (1, N, D)."""
    pos = torch.arange(num_patches, dtype=torch.float32)
    half = embed_dim // 2
    omega = torch.arange(half, dtype=torch.float32) / max(half, 1)
    omega = 1.0 / (10000 ** omega)
    out = torch.einsum("n,d->nd", pos, omega)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    if embed_dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb.unsqueeze(0)


class SourceEncoder1D(nn.Module):
    """Global source-spectrum encoder for VibraDiT conditioning."""
    def __init__(self, hidden_size: int):
        super().__init__()
        mid = max(hidden_size // 2, 64)
        self.net = nn.Sequential(
            nn.Conv1d(1, mid, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(mid, hidden_size, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x_source: torch.Tensor) -> torch.Tensor:
        h = self.net(x_source.unsqueeze(1))
        return self.proj(h)


class PatchEmbed1D(nn.Module):
    """Patchify a flattened 1D spectrum with Conv1d projection."""
    def __init__(self, seq_len: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = math.ceil(seq_len / patch_size)
        self.padded_len = self.num_patches * patch_size
        self.pad_len = self.padded_len - seq_len
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len), mode="constant", value=0.0)
        x = self.proj(x)
        return x.transpose(1, 2)


def modulate_tokens(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class VibraDiTBlock(nn.Module):
    """1D DiT block with adaLN-Zero conditioning."""
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        h = modulate_tokens(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa[:, None, :] * attn_out
        h = modulate_tokens(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :] * self.mlp(h)
        return x


class VibraDiTFinalLayer(nn.Module):
    """Final 1D DiT layer: tokens to velocity patches."""
    def __init__(self, hidden_size: int, patch_size: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.linear = nn.Linear(hidden_size, patch_size)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate_tokens(self.norm_final(x), shift, scale)
        return self.linear(x)


class VibraDiTVelocityFieldNetwork(nn.Module):
    """
    Wavenumber-aware 1D DiT velocity field adapted from vibradit_flow.py.

    The public interface stays compatible with the existing 2D Flow pipeline:
    inputs are heatmaps (B, C, H, W), flattened internally into spectra, then
    reshaped back to the original heatmap layout.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_modes: int = 3,
        image_size: tuple = (60, 60),
        patch_size: int = 16,
        hidden_size: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if in_channels != 1:
            raise ValueError("VibraDiT backbone currently expects in_channels=1.")

        self.in_channels = in_channels
        self.image_size = image_size
        self.seq_len = int(image_size[0] * image_size[1])
        self.patch_size = patch_size

        self.x_embedder = PatchEmbed1D(self.seq_len, patch_size, in_chans=2, embed_dim=hidden_size)
        self.num_patches = self.x_embedder.num_patches
        self.register_buffer(
            "pos_embed",
            get_1d_sincos_pos_embed(self.num_patches, hidden_size),
            persistent=False,
        )

        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.mode_embed = nn.Embedding(num_modes, hidden_size)
        self.source_encoder = SourceEncoder1D(hidden_size)
        self.cond_fuse = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size))

        self.blocks = nn.ModuleList([
            VibraDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.final_layer = VibraDiTFinalLayer(hidden_size, patch_size)

    def _target_mode_tensor(self, x: torch.Tensor, target_mode) -> torch.Tensor:
        if target_mode is None:
            return torch.zeros((x.size(0),), dtype=torch.long, device=x.device)
        if isinstance(target_mode, int):
            return torch.full((x.size(0),), target_mode, dtype=torch.long, device=x.device)
        if isinstance(target_mode, torch.Tensor):
            return target_mode.long().to(x.device)
        return torch.tensor(target_mode, dtype=torch.long, device=x.device)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition=None, target_mode=None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("VibraDiT backbone expects x with shape (B, C, H, W).")
        if condition is None:
            condition = x

        batch_size = x.size(0)
        original_shape = x.shape
        x_tau = x.reshape(batch_size, -1)
        x_source = condition.reshape(batch_size, -1)
        target_mode_tensor = self._target_mode_tensor(x, target_mode)

        x_in = torch.stack([x_tau, x_source], dim=1)
        h = self.x_embedder(x_in) + self.pos_embed.to(dtype=x.dtype, device=x.device)
        c = self.time_embed(t) + self.mode_embed(target_mode_tensor) + self.source_encoder(x_source)
        c = self.cond_fuse(c)

        for block in self.blocks:
            h = block(h, c)

        patches = self.final_layer(h, c)
        velocity = patches.reshape(batch_size, -1)[:, : self.seq_len]
        return velocity.reshape(original_shape)


class ConditionalFlowMatching(nn.Module):
    """
    条件Flow Matching模型
    用于从源光谱分布转换到目标光谱分布

    支持三种速度场骨干网络：
        backbone='unet'  (默认): 2D U-Net + FiLM
        backbone='dit'         : Diffusion Transformer + adaLN
        backbone='vibradit'    : 1D spectral DiT + adaLN-Zero

    DiT 额外参数（仅 backbone='dit' 时生效）：
        dit_hidden_dim  : Transformer 隐藏维度（默认 256）
        dit_depth       : DiT 块数量（默认 6）
        dit_num_heads   : 注意力头数（默认 4）
        dit_patch_size  : Patch 大小（默认 4）
    """
    def __init__(self, in_channels=1, hidden_channels=128, num_modes=3,
                 image_size=(60, 60), sigma_min=0.01,
                 backbone='unet',
                 dit_hidden_dim=256, dit_depth=6,
                 dit_num_heads=4, dit_patch_size=4):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_modes = num_modes
        self.image_size = image_size
        self.sigma_min = sigma_min
        self.backbone = backbone

        if backbone == 'dit':
            self.velocity_field = DiTVelocityFieldNetwork(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_modes=num_modes,
                image_size=image_size,
                patch_size=dit_patch_size,
                dit_hidden_dim=dit_hidden_dim,
                depth=dit_depth,
                num_heads=dit_num_heads,
            )
        elif backbone == 'vibradit':
            self.velocity_field = VibraDiTVelocityFieldNetwork(
                in_channels=in_channels,
                num_modes=num_modes,
                image_size=image_size,
                patch_size=dit_patch_size,
                hidden_size=dit_hidden_dim,
                depth=dit_depth,
                num_heads=dit_num_heads,
            )
        else:  # 'unet'（默认，向后兼容）
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
