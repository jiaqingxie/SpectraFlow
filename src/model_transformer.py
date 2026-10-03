"""
Pure Transformer baseline for cross-modal spectral translation.
"""
import torch
import torch.nn as nn


class ConditionalTransformerBlock(nn.Module):
    """Transformer block with target-mode conditioning."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cond):
        cond = cond.unsqueeze(1)
        h = self.norm1(x + cond)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        h = self.norm2(x + cond)
        x = x + self.mlp(h)
        return x


class TransformerSpectrumTranslator(nn.Module):
    """
    Pure Transformer baseline:
    source heatmap -> target heatmap.
    """

    def __init__(
        self,
        in_channels=1,
        image_size=(60, 60),
        patch_size=4,
        hidden_dim=256,
        depth=6,
        num_heads=4,
        num_modes=3,
        mlp_ratio=4.0,
        dropout=0.0,
        clamp_output=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.clamp_output = clamp_output

        height, width = image_size
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"image_size {image_size} must be divisible by patch_size {patch_size}"
            )

        self.num_patches_h = height // patch_size
        self.num_patches_w = width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.mode_embed = nn.Embedding(num_modes, hidden_dim)

        self.blocks = nn.ModuleList([
            ConditionalTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.patch_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _patchify(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def _unpatchify(self, patches):
        batch_size, _, _ = patches.shape
        patch_size = self.patch_size
        patches = patches.reshape(
            batch_size,
            self.num_patches_h,
            self.num_patches_w,
            self.in_channels,
            patch_size,
            patch_size,
        )
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        patches = patches.reshape(
            batch_size,
            self.in_channels,
            self.num_patches_h * patch_size,
            self.num_patches_w * patch_size,
        )
        return patches

    def _target_mode_tensor(self, x, target_mode):
        if isinstance(target_mode, int):
            return torch.full(
                (x.size(0),),
                target_mode,
                dtype=torch.long,
                device=x.device,
            )
        if isinstance(target_mode, torch.Tensor):
            return target_mode.long().to(x.device)
        raise TypeError("target_mode must be int or torch.Tensor")

    def forward(self, x, target_mode):
        target_mode = self._target_mode_tensor(x, target_mode)
        cond = self.mode_embed(target_mode)

        tokens = self._patchify(x) + cond.unsqueeze(1)

        for block in self.blocks:
            tokens = block(tokens, cond)

        tokens = self.norm(tokens)
        patches = self.out_proj(tokens)
        output = self._unpatchify(patches)

        if self.clamp_output:
            output = torch.sigmoid(output)
        return output
