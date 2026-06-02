#!/usr/bin/env python3
"""
VibraDiT-Flow: Wavenumber-aware DiT Flow Matching for IR <-> Raman spectral translation.

This is a single-file PyTorch implementation for paired 1D spectral sequences.
It only requires paired arrays: IR spectra and Raman spectra.

Expected .npz format:
    data.npz with keys "ir" and "raman" by default
    ir.shape == raman.shape == [num_samples, spectral_length]

Example:
    python vibradit_flow.py \
        --data_npz data.npz \
        --ir_key ir --raman_key raman \
        --direction bidir \
        --epochs 100 --batch_size 64 \
        --seq_len 3600 --patch_size 20 \
        --hidden_size 384 --depth 8 --num_heads 6 \
        --out_dir runs/vibradit_flow

Notes:
  * If your spectra are already normalized, use --normalize none.
  * If they are raw nonnegative intensities, --normalize sample is usually a good first try.
  * For long spectra such as L=3600, patch_size 20 gives 180 tokens. Larger patch size is faster.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Utilities
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sample_minmax_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    xmin = x.min(axis=1, keepdims=True)
    xmax = x.max(axis=1, keepdims=True)
    return (x - xmin) / (xmax - xmin + eps)


def make_splits(n: int, seed: int = 42, train_frac: float = 0.70, val_frac: float = 0.15):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


# -----------------------------
# Dataset
# -----------------------------


class PairedSpectraDataset(Dataset):
    """Dataset for paired IR and Raman spectra.

    direction:
      - "ir2raman": source=IR, target=Raman, target_mod=1
      - "raman2ir": source=Raman, target=IR, target_mod=0
      - "bidir": length doubles; first half IR->Raman, second half Raman->IR
    """

    def __init__(
        self,
        ir: np.ndarray,
        raman: np.ndarray,
        indices: np.ndarray,
        direction: str = "bidir",
        normalize: str = "sample",
        seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert direction in {"ir2raman", "raman2ir", "bidir"}
        assert normalize in {"sample", "none"}
        assert ir.shape == raman.shape, f"IR and Raman shapes differ: {ir.shape} vs {raman.shape}"
        self.direction = direction
        self.indices = np.asarray(indices)

        ir = ir.astype(np.float32)
        raman = raman.astype(np.float32)
        if normalize == "sample":
            ir = sample_minmax_normalize(ir)
            raman = sample_minmax_normalize(raman)

        if seq_len is not None:
            if ir.shape[1] != seq_len:
                raise ValueError(f"--seq_len={seq_len}, but arrays have length {ir.shape[1]}")
        self.ir = ir
        self.raman = raman

    def __len__(self) -> int:
        if self.direction == "bidir":
            return 2 * len(self.indices)
        return len(self.indices)

    def __getitem__(self, j: int) -> Dict[str, torch.Tensor]:
        if self.direction == "bidir":
            base = j % len(self.indices)
            forward = j < len(self.indices)
        else:
            base = j
            forward = self.direction == "ir2raman"

        i = self.indices[base]
        ir = self.ir[i]
        ra = self.raman[i]
        if forward:
            source, target, target_mod = ir, ra, 1  # Raman target
        else:
            source, target, target_mod = ra, ir, 0  # IR target

        return {
            "source": torch.from_numpy(source.copy()),       # [L]
            "target": torch.from_numpy(target.copy()),       # [L]
            "target_mod": torch.tensor(target_mod).long(),   # scalar: 0=IR, 1=Raman
        }


# -----------------------------
# DiT modules for 1D spectra
# -----------------------------


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding. t is [B] in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / max(half, 1)
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def get_1d_sincos_pos_embed(num_patches: int, embed_dim: int) -> torch.Tensor:
    """Fixed 1D sin/cos position embedding, returned as [1, N, D]."""
    pos = torch.arange(num_patches, dtype=torch.float32)
    half = embed_dim // 2
    omega = torch.arange(half, dtype=torch.float32) / max(half, 1)
    omega = 1.0 / (10000 ** omega)
    out = torch.einsum("n,d->nd", pos, omega)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    if embed_dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb.unsqueeze(0)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.frequency_embedding_size))


class SourceEncoder1D(nn.Module):
    """Small global encoder for the source spectrum condition."""

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
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, x_source: torch.Tensor) -> torch.Tensor:
        # x_source: [B, L]
        h = self.net(x_source.unsqueeze(1))
        return self.proj(h)


class PatchEmbed1D(nn.Module):
    """Patchify 1D spectra with a Conv1d projection."""

    def __init__(self, seq_len: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = math.ceil(seq_len / patch_size)
        self.padded_len = self.num_patches * patch_size
        self.pad_len = self.padded_len - seq_len
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len), mode="constant", value=0.0)
        x = self.proj(x)               # [B, D, N]
        return x.transpose(1, 2)       # [B, N, D]


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: [B, N, D], shift/scale: [B, D]
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTBlock(nn.Module):
    """A DiT block with adaLN-Zero conditioning."""

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

        # adaLN-Zero initialization: stable for flow/diffusion transformers.
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa[:, None, :] * attn_out
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :] * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    """Final DiT layer: tokens -> velocity patches."""

    def __init__(self, hidden_size: int, patch_size: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.linear = nn.Linear(hidden_size, patch_size)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)  # [B, N, patch_size]


class SpectralDiTFlow(nn.Module):
    """Conditional 1D DiT velocity field for IR/Raman flow matching.

    Input channels are [x_tau, x_source]. The model outputs velocity v_theta with shape [B, L].
    target_mod: 0 means IR target, 1 means Raman target.
    """

    def __init__(
        self,
        seq_len: int,
        patch_size: int = 16,
        hidden_size: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.x_embedder = PatchEmbed1D(seq_len, patch_size, in_chans=2, embed_dim=hidden_size)
        self.num_patches = self.x_embedder.num_patches
        self.register_buffer("pos_embed", get_1d_sincos_pos_embed(self.num_patches, hidden_size), persistent=False)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.mod_embedder = nn.Embedding(2, hidden_size)  # 0=IR, 1=Raman
        self.source_encoder = SourceEncoder1D(hidden_size)
        self.cond_fuse = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size))

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size)

    def forward(self, x_tau: torch.Tensor, tau: torch.Tensor, x_source: torch.Tensor, target_mod: torch.Tensor) -> torch.Tensor:
        # x_tau, x_source: [B, L]; tau: [B]; target_mod: [B]
        if x_tau.ndim != 2 or x_source.ndim != 2:
            raise ValueError("x_tau and x_source must be [B, L]")
        x_in = torch.stack([x_tau, x_source], dim=1)  # [B, 2, L]
        h = self.x_embedder(x_in) + self.pos_embed.to(x_tau.dtype)
        c = self.t_embedder(tau) + self.mod_embedder(target_mod) + self.source_encoder(x_source)
        c = self.cond_fuse(c)
        for block in self.blocks:
            h = block(h, c)
        patches = self.final_layer(h, c)              # [B, N, P]
        velocity = patches.reshape(patches.shape[0], -1)[:, : self.seq_len]
        return velocity


# -----------------------------
# Physics-informed spectral losses
# -----------------------------


def first_derivative(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, 1:] - x[:, :-1]
    return F.pad(dx, (1, 0), mode="constant", value=0.0)


def second_derivative(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 3:
        return torch.zeros_like(x)
    d2 = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    return F.pad(d2, (1, 1), mode="constant", value=0.0)


def normalize_by_sample_max(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.amax(dim=1, keepdim=True) + eps)


def spectral_importance_weight(
    target: torch.Tensor,
    amp_weight: float = 1.0,
    grad_weight: float = 0.5,
    curv_weight: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weight high-amplitude, high-slope, high-curvature spectral regions.

    target is the supervised target spectrum x_t. This is used only in training.
    """
    with torch.no_grad():
        amp = normalize_by_sample_max(target.abs(), eps)
        grad = normalize_by_sample_max(first_derivative(target).abs(), eps)
        curv = normalize_by_sample_max(second_derivative(target).abs(), eps)
        w = 1.0 + amp_weight * amp + grad_weight * grad + curv_weight * curv
    return w


def weighted_elastic_loss(pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None, alpha_l1: float = 0.6) -> torch.Tensor:
    err = pred - target
    loss = alpha_l1 * err.abs() + (1.0 - alpha_l1) * err.pow(2)
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def derivative_shape_loss(pred: torch.Tensor, target: torch.Tensor, curv_coef: float = 0.5) -> torch.Tensor:
    return (first_derivative(pred) - first_derivative(target)).abs().mean() + curv_coef * (
        second_derivative(pred) - second_derivative(target)
    ).abs().mean()


def local_ot_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 64, eps: float = 1e-8) -> torch.Tensor:
    """Windowed 1D Wasserstein-1 distance using cumulative distributions.

    This is shift-tolerant inside each local window and is useful for peak localization.
    """
    b, l = pred.shape
    pad = (window_size - l % window_size) % window_size
    if pad > 0:
        pred = F.pad(pred, (0, pad), mode="constant", value=0.0)
        target = F.pad(target, (0, pad), mode="constant", value=0.0)
    pred = F.relu(pred).reshape(b, -1, window_size) + eps
    target = F.relu(target).reshape(b, -1, window_size) + eps
    pred = pred / pred.sum(dim=-1, keepdim=True)
    target = target / target.sum(dim=-1, keepdim=True)
    cdf_diff = torch.cumsum(pred - target, dim=-1).abs()
    return cdf_diff.mean()


def nonnegative_penalty(x: torch.Tensor) -> torch.Tensor:
    return F.relu(-x).pow(2).mean()


# -----------------------------
# ODE sampler and metrics
# -----------------------------


def ode_sample(
    model: SpectralDiTFlow,
    source: torch.Tensor,
    target_mod: torch.Tensor,
    steps: int = 16,
    method: str = "euler",
) -> torch.Tensor:
    """Integrate dx/dt = v_theta(x,t,source,target_mod), x(0)=source."""
    assert method in {"euler", "rk4"}
    x = source
    b = source.shape[0]
    dt = 1.0 / steps
    for i in range(steps):
        t0 = torch.full((b,), i * dt, device=source.device, dtype=source.dtype)
        if method == "euler":
            v = model(x, t0, source, target_mod)
            x = x + dt * v
        else:
            t1 = torch.full((b,), (i + 0.5) * dt, device=source.device, dtype=source.dtype)
            t2 = torch.full((b,), (i + 1.0) * dt, device=source.device, dtype=source.dtype)
            k1 = model(x, t0, source, target_mod)
            k2 = model(x + 0.5 * dt * k1, t1, source, target_mod)
            k3 = model(x + 0.5 * dt * k2, t1, source, target_mod)
            k4 = model(x + dt * k3, t2, source, target_mod)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = pred.float().reshape(-1)
    target = target.float().reshape(-1)
    mae = (pred - target).abs().mean().item()
    rmse = torch.sqrt((pred - target).pow(2).mean()).item()
    target_mean = target.mean()
    ss_res = (pred - target).pow(2).sum()
    ss_tot = (target - target_mean).pow(2).sum().clamp_min(1e-12)
    r2 = (1.0 - ss_res / ss_tot).item()
    vx = pred - pred.mean()
    vy = target - target.mean()
    pearson = (vx * vy).sum() / (torch.sqrt((vx.pow(2).sum() * vy.pow(2).sum()).clamp_min(1e-12)))
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson.item()}


@torch.no_grad()
def evaluate(
    model: SpectralDiTFlow,
    loader: DataLoader,
    device: torch.device,
    steps: int = 32,
    method: str = "euler",
) -> Dict[str, float]:
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        target_mod = batch["target_mod"].to(device)
        pred = ode_sample(model, source, target_mod, steps=steps, method=method)
        pred = pred.clamp_min(0.0)  # spectra are nonnegative after normalization
        preds.append(pred.cpu())
        targets.append(target.cpu())
    return compute_metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0))


# -----------------------------
# Training
# -----------------------------


@dataclass
class TrainConfig:
    data_npz: str
    ir_key: str = "ir"
    raman_key: str = "raman"
    direction: str = "bidir"
    normalize: str = "sample"
    seq_len: Optional[int] = None
    patch_size: int = 16
    hidden_size: int = 384
    depth: int = 8
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    epochs: int = 100
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 4
    amp_weight: float = 1.0
    grad_weight: float = 0.5
    curv_weight: float = 0.5
    lambda_end: float = 0.5
    lambda_shape: float = 0.05
    lambda_ot: float = 0.02
    lambda_pos: float = 0.01
    endpoint_prob: float = 0.25
    endpoint_steps: int = 8
    eval_steps: int = 32
    ode_method: str = "euler"
    out_dir: str = "runs/vibradit_flow"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(
    model: SpectralDiTFlow,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    meter = {"loss": 0.0, "fm": 0.0, "end": 0.0, "shape": 0.0, "ot": 0.0, "pos": 0.0}
    n_batches = 0
    for batch in loader:
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        target_mod = batch["target_mod"].to(device)
        b = source.shape[0]

        tau = torch.rand(b, device=device, dtype=source.dtype)
        x_tau = (1.0 - tau[:, None]) * source + tau[:, None] * target
        u_tau = target - source
        weight = spectral_importance_weight(target, cfg.amp_weight, cfg.grad_weight, cfg.curv_weight)

        pred_v = model(x_tau, tau, source, target_mod)
        loss_fm = weighted_elastic_loss(pred_v, u_tau, weight=weight, alpha_l1=0.6)

        loss_end = source.new_tensor(0.0)
        loss_shape = source.new_tensor(0.0)
        loss_ot = source.new_tensor(0.0)
        loss_pos = source.new_tensor(0.0)

        # Endpoint consistency is expensive because it backprops through ODE steps.
        # We apply it stochastically, as in the original VibraFlow spirit.
        if cfg.lambda_end > 0 and random.random() < cfg.endpoint_prob:
            pred_x = ode_sample(model, source, target_mod, steps=cfg.endpoint_steps, method=cfg.ode_method)
            loss_end = weighted_elastic_loss(pred_x, target, weight=weight, alpha_l1=0.6)
            if cfg.lambda_shape > 0:
                loss_shape = derivative_shape_loss(pred_x, target)
            if cfg.lambda_ot > 0:
                loss_ot = local_ot_loss(pred_x, target, window_size=max(cfg.patch_size * 4, 32))
            if cfg.lambda_pos > 0:
                loss_pos = nonnegative_penalty(pred_x)

        loss = (
            loss_fm
            + cfg.lambda_end * loss_end
            + cfg.lambda_shape * loss_shape
            + cfg.lambda_ot * loss_ot
            + cfg.lambda_pos * loss_pos
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        meter["loss"] += loss.item()
        meter["fm"] += loss_fm.item()
        meter["end"] += loss_end.item()
        meter["shape"] += loss_shape.item()
        meter["ot"] += loss_ot.item()
        meter["pos"] += loss_pos.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in meter.items()}


def build_loaders(cfg: TrainConfig):
    data = np.load(cfg.data_npz)
    if cfg.ir_key not in data or cfg.raman_key not in data:
        raise KeyError(f"NPZ must contain keys {cfg.ir_key!r} and {cfg.raman_key!r}. Found: {list(data.keys())}")
    ir = data[cfg.ir_key]
    raman = data[cfg.raman_key]
    if ir.ndim != 2 or raman.ndim != 2:
        raise ValueError(f"Expected [N,L] arrays. Got IR {ir.shape}, Raman {raman.shape}")
    if cfg.seq_len is None:
        cfg.seq_len = int(ir.shape[1])
    train_idx, val_idx, test_idx = make_splits(ir.shape[0], cfg.seed)

    train_ds = PairedSpectraDataset(ir, raman, train_idx, cfg.direction, cfg.normalize, cfg.seq_len)
    val_ds = PairedSpectraDataset(ir, raman, val_idx, cfg.direction, cfg.normalize, cfg.seq_len)
    test_ir2ra = PairedSpectraDataset(ir, raman, test_idx, "ir2raman", cfg.normalize, cfg.seq_len)
    test_ra2ir = PairedSpectraDataset(ir, raman, test_idx, "raman2ir", cfg.normalize, cfg.seq_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_ir2ra_loader = DataLoader(test_ir2ra, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_ra2ir_loader = DataLoader(test_ra2ir, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader, test_ir2ra_loader, test_ra2ir_loader


def run_training(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_loader, val_loader, test_ir2ra_loader, test_ra2ir_loader = build_loaders(cfg)
    device = torch.device(cfg.device)
    model = SpectralDiTFlow(
        seq_len=int(cfg.seq_len),
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
    ).to(device)

    print(f"Model parameters: {count_parameters(model)/1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val = float("inf")
    best_path = out_dir / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, cfg, epoch)
        scheduler.step()
        val_stats = evaluate(model, val_loader, device, steps=cfg.eval_steps, method=cfg.ode_method)
        is_best = val_stats["mae"] < best_val
        if is_best:
            best_val = val_stats["mae"]
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "val": val_stats}, best_path)

        print(
            f"Epoch {epoch:04d} | "
            f"train loss {train_stats['loss']:.5f} fm {train_stats['fm']:.5f} end {train_stats['end']:.5f} | "
            f"val MAE {val_stats['mae']:.5f} RMSE {val_stats['rmse']:.5f} R2 {val_stats['r2']:.4f} r {val_stats['pearson']:.4f}"
            + ("  *best" if is_best else "")
        )

    print(f"Loading best checkpoint: {best_path}")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    ir2ra = evaluate(model, test_ir2ra_loader, device, steps=cfg.eval_steps, method=cfg.ode_method)
    ra2ir = evaluate(model, test_ra2ir_loader, device, steps=cfg.eval_steps, method=cfg.ode_method)
    results = {"IR_to_Raman": ir2ra, "Raman_to_IR": ra2ir}
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Test metrics:")
    print(json.dumps(results, indent=2))


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train VibraDiT-Flow on paired IR/Raman spectra.")
    p.add_argument("--data_npz", type=str, required=True)
    p.add_argument("--ir_key", type=str, default="ir")
    p.add_argument("--raman_key", type=str, default="raman")
    p.add_argument("--direction", type=str, default="bidir", choices=["ir2raman", "raman2ir", "bidir"])
    p.add_argument("--normalize", type=str, default="sample", choices=["sample", "none"])
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--hidden_size", type=int, default=384)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp_weight", type=float, default=1.0)
    p.add_argument("--grad_weight", type=float, default=0.5)
    p.add_argument("--curv_weight", type=float, default=0.5)
    p.add_argument("--lambda_end", type=float, default=0.5)
    p.add_argument("--lambda_shape", type=float, default=0.05)
    p.add_argument("--lambda_ot", type=float, default=0.02)
    p.add_argument("--lambda_pos", type=float, default=0.01)
    p.add_argument("--endpoint_prob", type=float, default=0.25)
    p.add_argument("--endpoint_steps", type=int, default=8)
    p.add_argument("--eval_steps", type=int, default=32)
    p.add_argument("--ode_method", type=str, default="euler", choices=["euler", "rk4"])
    p.add_argument("--out_dir", type=str, default="runs/vibradit_flow")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    run_training(parse_args())
