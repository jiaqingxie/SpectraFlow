"""Train the parameter-matched SpectraDiT-Direct baseline.

The model uses exactly the same ConditionalFlowMatching/VibraDiT backbone as
SpectraFlow, but performs one network evaluation at a fixed time and predicts
the endpoint residual directly:

    target_hat = source + f_theta(source, t=direct_time, condition=source).

This isolates iterative flow integration from backbone capacity and spectral
loss design. Checkpoints are saved with a ``direct_`` prefix and therefore
cannot overwrite Flow checkpoints.
"""

import argparse
import contextlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_flow import ConditionalFlowMatching
from train import PairedModalDataset, get_paired_loaders
from train_flow import (
    _flatten_spectrum_batch,
    derivative_shape_loss,
    local_ot_loss,
    nonnegative_penalty,
    spectral_importance_weight,
    weighted_elastic_loss,
)
from utils import get_device


class SpectraDiTDirectTrainer:
    """Parameter-matched direct residual-regression baseline."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.direct_time = float(config.direct_time)
        self.use_amp = bool(config.use_amp) and device.type == "cuda"
        self.scaler = amp.GradScaler("cuda") if self.use_amp else None

        # This is intentionally the exact model class/configuration used by
        # FlowMatchingTrainer. Only the training/inference call is different.
        self.model = ConditionalFlowMatching(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            num_modes=config.num_modes,
            image_size=config.resize_shape,
            sigma_min=config.sigma_min,
            backbone=config.backbone,
            dit_hidden_dim=config.dit_hidden_dim,
            dit_depth=config.dit_depth,
            dit_num_heads=config.dit_num_heads,
            dit_patch_size=config.dit_patch_size,
        ).to(device)
        self._initialize_like_flow()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01,
        )
        self.mode_map = {"ir": 0, "uv": 1, "raman": 2}

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[model] backbone={self.model.backbone}")
        print(f"[model] network={self.model.velocity_field.__class__.__name__}")
        print(f"[model] trainable_parameters={n_params:,}")
        print(f"[direct] fixed_time={self.direct_time}; network_evaluations=1")
        if self.use_amp:
            print("[train] AMP (fp16) enabled")

    def _initialize_like_flow(self):
        zero_init_backbone = self.config.backbone in {"dit", "vibradit"}
        skip_zero_init_names = (
            "velocity_field.final_adaLN",
            "velocity_field.out_proj",
            "velocity_field.final_layer.adaLN_modulation",
            "velocity_field.final_layer.linear",
        )
        for name, module in self.model.named_modules():
            if not isinstance(
                module,
                (
                    nn.Conv1d,
                    nn.ConvTranspose1d,
                    nn.Conv2d,
                    nn.ConvTranspose2d,
                    nn.Linear,
                ),
            ):
                continue
            is_adaln_zero = ".adaLN" in name or ".adaLN_modulation" in name
            is_final_zero = any(name.startswith(prefix) for prefix in skip_zero_init_names)
            if zero_init_backbone and (is_adaln_zero or is_final_zero):
                continue
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def _autocast_ctx(self):
        if self.device.type != "cuda":
            return contextlib.nullcontext()
        return amp.autocast(device_type="cuda", enabled=self.use_amp)

    def predict(self, source, target_mode):
        """One-pass residual prediction with the full Flow backbone."""
        target_mode_idx = self.mode_map[target_mode]
        t = torch.full(
            (source.size(0),),
            self.direct_time,
            dtype=source.dtype,
            device=source.device,
        )
        residual = self.model.velocity_field(
            source, t, condition=source, target_mode=target_mode_idx
        )
        return source + residual, residual

    def _losses(
        self,
        prediction,
        residual,
        source,
        target,
        amp_weight,
        grad_weight,
        curv_weight,
        lambda_shape,
        lambda_ot,
        lambda_pos,
        ot_window_size,
        endpoint_loss_weight,
        endpoint_loss_prob,
        apply_endpoint,
        training,
    ):
        source_flat = _flatten_spectrum_batch(source.float())
        target_flat = _flatten_spectrum_batch(target.float())
        pred_flat = _flatten_spectrum_batch(prediction.float())
        residual_flat = _flatten_spectrum_batch(residual.float())
        true_residual_flat = target_flat - source_flat
        weight = spectral_importance_weight(
            target_flat, amp_weight, grad_weight, curv_weight
        )

        # Parameter-matched analogue of the Flow velocity objective.
        residual_loss = weighted_elastic_loss(
            residual_flat, true_residual_flat, weight=weight, alpha_l1=0.6
        )

        endpoint_loss = torch.zeros_like(residual_loss)
        if apply_endpoint:
            # Same spectroscopy-informed endpoint terms used by SpectraFlow.
            endpoint_loss = weighted_elastic_loss(
                pred_flat, target_flat, weight=weight, alpha_l1=0.6
            )
            if lambda_shape > 0:
                endpoint_loss = endpoint_loss + lambda_shape * derivative_shape_loss(
                    pred_flat, target_flat
                )
            if lambda_ot > 0:
                endpoint_loss = endpoint_loss + lambda_ot * local_ot_loss(
                    pred_flat, target_flat, window_size=ot_window_size
                )
            if lambda_pos > 0:
                endpoint_loss = endpoint_loss + lambda_pos * nonnegative_penalty(pred_flat)

        if training:
            endpoint_scale = (
                endpoint_loss_weight / endpoint_loss_prob if apply_endpoint else 0.0
            )
        else:
            endpoint_scale = endpoint_loss_weight
        total_loss = residual_loss + endpoint_scale * endpoint_loss
        # Match the Flow checkpoint-selection metric.
        val_gen_loss = 0.6 * F.l1_loss(pred_flat, target_flat) + 0.4 * F.mse_loss(
            pred_flat, target_flat
        )
        return total_loss, residual_loss, endpoint_loss, val_gen_loss

    def step(self, source, target, target_mode, train, loss_kwargs):
        source = source.to(self.device)
        target = target.to(self.device)
        if train:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.model.eval()

        endpoint_loss_prob = loss_kwargs["endpoint_loss_prob"]
        apply_endpoint = (not train) or (
            endpoint_loss_prob > 0 and np.random.rand() < endpoint_loss_prob
        )
        grad_ctx = contextlib.nullcontext() if train else torch.no_grad()
        with grad_ctx:
            with self._autocast_ctx():
                prediction, residual = self.predict(source, target_mode)
            losses = self._losses(
                prediction,
                residual,
                source,
                target,
                apply_endpoint=apply_endpoint,
                training=train,
                **loss_kwargs,
            )

        if train:
            total_loss = losses[0]
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

        names = ("total_loss", "residual_loss", "endpoint_loss", "val_gen_loss")
        result = {name: float(value.detach()) for name, value in zip(names, losses)}
        result["endpoint_applied"] = float(apply_endpoint)
        return result

    def save_checkpoint(self, path, epoch, best_val_gen_loss, args):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch,
                "best_val_gen_loss": best_val_gen_loss,
                "model_type": "spectradit_direct_residual",
                "direct_time": self.direct_time,
                "config": vars(args),
            },
            path,
        )


def make_loaders(args, config):
    data_dir = Path(args.data_dir)
    source_csv = (
        Path(args.source_csv)
        if args.source_csv and os.path.isabs(args.source_csv)
        else data_dir / (args.source_csv or f"{args.source_mode}_broaden_processed.csv")
    )
    target_csv = (
        Path(args.target_csv)
        if args.target_csv and os.path.isabs(args.target_csv)
        else data_dir / (args.target_csv or f"{args.target_mode}_broaden_processed.csv")
    )
    if not source_csv.exists() and not source_csv.with_suffix(".h5").exists():
        raise FileNotFoundError(source_csv)
    if not target_csv.exists() and not target_csv.with_suffix(".h5").exists():
        raise FileNotFoundError(target_csv)

    if bool(args.val_source_csv) != bool(args.val_target_csv):
        raise ValueError("--val_source_csv and --val_target_csv must be provided together")

    dataset_kwargs = dict(
        source_size=args.source_size,
        target_size=args.target_size,
        heatmap_size=args.heatmap_size,
        resize_shape=config.resize_shape,
        out_channels=config.in_channels,
        use_h5=True,
        preserve_spectral_order=args.preserve_spectral_order,
    )
    if args.val_source_csv:
        val_source = Path(args.val_source_csv)
        val_target = Path(args.val_target_csv)
        if not val_source.is_absolute():
            val_source = data_dir / val_source
        if not val_target.is_absolute():
            val_target = data_dir / val_target
        train_set = PairedModalDataset(str(source_csv), str(target_csv), **dataset_kwargs)
        val_set = PairedModalDataset(str(val_source), str(val_target), **dataset_kwargs)
        return (
            DataLoader(train_set, batch_size=args.batch_size, shuffle=True),
            DataLoader(val_set, batch_size=args.batch_size, shuffle=False),
        )

    train_loader, val_loader, _ = get_paired_loaders(
        str(source_csv),
        str(target_csv),
        batch_size=args.batch_size,
        source_size=args.source_size,
        target_size=args.target_size,
        heatmap_size=args.heatmap_size,
        resize_shape=config.resize_shape,
        out_channels=config.in_channels,
        use_h5=True,
        seed=args.seed,
        preserve_spectral_order=args.preserve_spectral_order,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    return train_loader, val_loader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the parameter-matched SpectraDiT-Direct baseline"
    )
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--source_csv")
    parser.add_argument("--target_csv")
    parser.add_argument("--val_source_csv")
    parser.add_argument("--val_target_csv")
    parser.add_argument("--source_mode", required=True, choices=["ir", "uv", "raman"])
    parser.add_argument("--target_mode", required=True, choices=["ir", "uv", "raman"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--heatmap_size", type=int, default=3600)
    parser.add_argument("--resize_shape", type=int, nargs=2, default=[60, 60])
    parser.add_argument("--source_size", type=int)
    parser.add_argument("--target_size", type=int)
    parser.add_argument("--preserve_spectral_order", action="store_true")
    parser.add_argument(
        "--backbone", default="vibradit", choices=["unet", "dit", "vibradit"]
    )
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--dit_hidden_dim", type=int, default=384)
    parser.add_argument("--dit_depth", type=int, default=8)
    parser.add_argument("--dit_num_heads", type=int, default=6)
    parser.add_argument("--dit_patch_size", type=int, default=20)
    parser.add_argument("--sigma_min", type=float, default=0.01)
    parser.add_argument("--direct_time", type=float, default=0.0)
    parser.add_argument("--endpoint_loss_weight", type=float, default=0.1)
    parser.add_argument(
        "--endpoint_loss_prob",
        type=float,
        default=0.1,
        help="Fraction of training batches that include endpoint reconstruction loss",
    )
    parser.add_argument("--amp_weight", type=float, default=1.0)
    parser.add_argument("--grad_weight", type=float, default=0.5)
    parser.add_argument("--curv_weight", type=float, default=0.5)
    parser.add_argument("--lambda_shape", type=float, default=0.05)
    parser.add_argument("--lambda_ot", type=float, default=0.02)
    parser.add_argument("--lambda_pos", type=float, default=0.01)
    parser.add_argument("--ot_window_size", type=int, default=64)
    parser.add_argument("--save_dir", default="checkpoints/qm9s_direct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=0,
        help="PyTorch CPU threads; 0 keeps the environment default",
    )
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--cudnn_benchmark", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.source_mode == args.target_mode:
        raise ValueError("source_mode and target_mode must differ")
    if not 0.0 <= args.direct_time <= 1.0:
        raise ValueError("--direct_time must be within [0, 1]")
    if not 0.0 <= args.endpoint_loss_prob <= 1.0:
        raise ValueError("--endpoint_loss_prob must be within [0, 1]")
    if np.prod(args.resize_shape) != args.heatmap_size:
        raise ValueError("--resize_shape product must equal --heatmap_size")
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(args.cpu_threads)
        except RuntimeError:
            pass
        print(f"[cpu] threads={args.cpu_threads}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = not args.cudnn_benchmark
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
    device = get_device(args.cpu)

    class Config:
        in_channels = 1
        hidden_channels = args.hidden_channels
        num_modes = 3
        resize_shape = tuple(args.resize_shape)
        sigma_min = args.sigma_min
        backbone = args.backbone
        dit_hidden_dim = args.dit_hidden_dim
        dit_depth = args.dit_depth
        dit_num_heads = args.dit_num_heads
        dit_patch_size = args.dit_patch_size
        learning_rate = args.learning_rate
        epochs = args.epochs
        direct_time = args.direct_time
        use_amp = not args.no_amp

    config = Config()
    trainer = SpectraDiTDirectTrainer(config, device)
    train_loader, val_loader = make_loaders(args, config)
    print(f"[split] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    loss_kwargs = dict(
        amp_weight=args.amp_weight,
        grad_weight=args.grad_weight,
        curv_weight=args.curv_weight,
        lambda_shape=args.lambda_shape,
        lambda_ot=args.lambda_ot,
        lambda_pos=args.lambda_pos,
        ot_window_size=args.ot_window_size,
        endpoint_loss_weight=args.endpoint_loss_weight,
        endpoint_loss_prob=args.endpoint_loss_prob,
    )
    print(
        f"[loss] endpoint_weight={args.endpoint_loss_weight} "
        f"endpoint_probability={args.endpoint_loss_prob}"
    )
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        args.save_dir,
        f"direct_{args.source_mode}2{args.target_mode}_{args.backbone}_best_seed{args.seed}.pt",
    )
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_rows = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            row = trainer.step(
                batch[0], batch[1], args.target_mode, train=True, loss_kwargs=loss_kwargs
            )
            train_rows.append(row)
            progress.set_postfix(loss=f"{row['total_loss']:.6f}")

        val_rows = [
            trainer.step(
                batch[0], batch[1], args.target_mode, train=False, loss_kwargs=loss_kwargs
            )
            for batch in val_loader
        ]
        trainer.scheduler.step()
        train_total = float(np.mean([row["total_loss"] for row in train_rows]))
        val_total = float(np.mean([row["total_loss"] for row in val_rows]))
        val_gen = float(np.mean([row["val_gen_loss"] for row in val_rows]))
        print(
            f"Epoch {epoch}: train_total={train_total:.6f} "
            f"val_total={val_total:.6f} val_gen_loss={val_gen:.6f}"
        )

        if val_gen < best_val:
            best_val = val_gen
            trainer.save_checkpoint(checkpoint_path, epoch, best_val, args)
            print(f"  -> saved {checkpoint_path}")

    print(f"Training completed. Best val_gen_loss={best_val:.6f}")


if __name__ == "__main__":
    main()
