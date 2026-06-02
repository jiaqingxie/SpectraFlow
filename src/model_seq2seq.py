"""
Encoder–Decoder (seq2seq) baseline: BiGRU encoder + Bahdanau attention + GRU decoder.

跨模態熱圖：patch 展平為序列，編碼端為雙向 GRU，解碼端逐步生成每個 patch，
每一步對 encoder 輸出做加性注意力；不使用額外 positional embedding（步序僅由遞歸隱狀態體現）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """加性注意力：query = 當前解碼狀態，key/value = encoder 各步輸出。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, dec_state: torch.Tensor, enc_h: torch.Tensor) -> torch.Tensor:
        # dec_state: (B, H), enc_h: (B, L, H)
        u = self.W_s(dec_state).unsqueeze(1) + self.W_h(enc_h)
        energy = self.v(torch.tanh(u)).squeeze(-1)
        alpha = F.softmax(energy, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), enc_h).squeeze(1)
        return context


class Seq2SeqSpectrumTranslator(nn.Module):
    """
    Source heatmap -> target heatmap：BiGRU encoder + attention + GRU decoder（按 patch 序逐步解碼）。
    """

    def __init__(
        self,
        in_channels=1,
        image_size=(60, 60),
        patch_size=4,
        hidden_dim=256,
        encoder_depth=6,
        decoder_depth=6,
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
        self.encoder_num_layers = encoder_depth
        # decoder_depth / num_heads / mlp_ratio：與舊 CLI 相容，本架構未使用

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

        self.gru_hidden = max(32, hidden_dim // 2)
        bi_dim = 2 * self.gru_hidden
        self.encoder_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=self.gru_hidden,
            num_layers=encoder_depth,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if encoder_depth > 1 else 0.0,
        )
        self.enc_proj = nn.Linear(bi_dim, hidden_dim)

        self.attention = BahdanauAttention(hidden_dim)

        self.dec_init = nn.Linear(bi_dim, hidden_dim)
        # 解碼每步僅以 attention context 為輸入（無 dec_pos / per-step query）
        self.decoder_gru = nn.GRUCell(hidden_dim, hidden_dim)

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

    def _init_decoder_state(self, h_n: torch.Tensor) -> torch.Tensor:
        # h_n: (num_layers * 2, B, gru_h) — 取最後一層前向與後向
        nl = self.encoder_num_layers
        idx_f = 2 * (nl - 1)
        idx_b = idx_f + 1
        h_cat = torch.cat([h_n[idx_f], h_n[idx_b]], dim=-1)
        return torch.tanh(self.dec_init(h_cat))

    def forward(self, x, target_mode):
        target_mode = self._target_mode_tensor(x, target_mode)
        cond = self.mode_embed(target_mode)

        enc_in = self._patchify(x) + cond.unsqueeze(1)
        enc_out, h_n = self.encoder_gru(enc_in)
        enc_h = self.enc_proj(enc_out)

        s = self._init_decoder_state(h_n)

        outs = []
        for _ in range(self.num_patches):
            ctx = self.attention(s, enc_h)
            s = self.decoder_gru(ctx, s)
            outs.append(self.out_proj(self.norm(s)))

        patches = torch.stack(outs, dim=1)
        output = self._unpatchify(patches)

        if self.clamp_output:
            output = torch.sigmoid(output)
        return output
