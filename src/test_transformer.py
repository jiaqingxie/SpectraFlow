"""
Test the legacy patch Transformer baseline.

This loader targets checkpoints named like:
    transformer_{source_mode}2{target_mode}_best_seed{seed}.pt

The architecture is inferred from the checkpoint where possible, because older
Transformer checkpoints predate the current seq2seq baseline scripts.
"""
import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train import PairedModalDataset, get_paired_loaders
from utils import get_device, inverse_heatmap_to_spectrum, paired_dataset_use_random_split_by_default
from test_seq2seq import (
    calculate_metrics,
    compute_dtw_metric,
    peak_matching_metrics,
)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class LegacyPatchTransformer(nn.Module):
    """Patch-wise Transformer used by older transformer_* checkpoints."""

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
        final_norm_name="norm",
        output_name="out_proj",
        clamp_output=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = tuple(image_size)
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.clamp_output = clamp_output
        self.final_norm_name = final_norm_name
        self.output_name = output_name

        height, width = self.image_size
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(f"image_size {self.image_size} must be divisible by patch_size {patch_size}")

        self.num_patches_h = height // patch_size
        self.num_patches_w = width // patch_size
        self.patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.mode_embed = nn.Embedding(num_modes, hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        setattr(self, final_norm_name, nn.LayerNorm(hidden_dim))
        setattr(self, output_name, nn.Linear(hidden_dim, self.patch_dim))

    def _target_mode_tensor(self, x, target_mode):
        if isinstance(target_mode, int):
            return torch.full((x.size(0),), target_mode, dtype=torch.long, device=x.device)
        if isinstance(target_mode, torch.Tensor):
            return target_mode.long().to(x.device)
        raise TypeError("target_mode must be int or torch.Tensor")

    def _unpatchify(self, patches):
        batch_size = patches.size(0)
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
        return patches.reshape(
            batch_size,
            self.in_channels,
            self.num_patches_h * patch_size,
            self.num_patches_w * patch_size,
        )

    def forward(self, x, target_mode):
        target_mode = self._target_mode_tensor(x, target_mode)
        h = self.patch_embed(x).flatten(2).transpose(1, 2)
        h = h + self.mode_embed(target_mode).unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        h = getattr(self, self.final_norm_name)(h)
        patches = getattr(self, self.output_name)(h)
        output = self._unpatchify(patches)
        if self.clamp_output:
            output = torch.sigmoid(output)
        return output


def infer_transformer_config(state_dict, resize_shape, fallback_heads=4, fallback_dropout=0.0):
    patch_weight = state_dict["patch_embed.weight"]
    hidden_dim = int(patch_weight.shape[0])
    in_channels = int(patch_weight.shape[1])
    patch_size = int(patch_weight.shape[2])
    num_modes = int(state_dict["mode_embed.weight"].shape[0])

    block_ids = []
    for key in state_dict:
        match = re.match(r"blocks\.(\d+)\.", key)
        if match:
            block_ids.append(int(match.group(1)))
    depth = max(block_ids) + 1 if block_ids else 0

    mlp_key = "blocks.0.mlp.0.weight"
    mlp_ratio = 4.0
    if mlp_key in state_dict:
        mlp_ratio = float(state_dict[mlp_key].shape[0]) / float(hidden_dim)

    final_norm_name = "norm"
    for candidate in ("norm", "final_norm", "ln_f"):
        if f"{candidate}.weight" in state_dict:
            final_norm_name = candidate
            break

    output_name = "out_proj"
    for candidate in ("out_proj", "output_proj", "head", "patch_out"):
        if f"{candidate}.weight" in state_dict:
            output_name = candidate
            break

    return {
        "in_channels": in_channels,
        "image_size": tuple(resize_shape),
        "patch_size": patch_size,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "num_heads": fallback_heads,
        "num_modes": num_modes,
        "mlp_ratio": mlp_ratio,
        "dropout": fallback_dropout,
        "final_norm_name": final_norm_name,
        "output_name": output_name,
    }


def plot_comparison(original_list, reconstructed_list, save_dir, source_mode, target_mode, num_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    num_samples = min(num_samples, len(original_list))
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        orig = original_list[i]
        recon = reconstructed_list[i]
        x_axis = np.linspace(0, len(orig) - 1, len(orig))
        axes[i].plot(x_axis, orig, label="Original", alpha=0.8, linewidth=2)
        axes[i].plot(x_axis, recon, label="Transformer", alpha=0.8, linewidth=2, linestyle="--")
        axes[i].set_xlabel("Wavenumber Index")
        axes[i].set_ylabel("Intensity")
        axes[i].set_title(f"Sample {i + 1}: {source_mode} -> {target_mode} (Transformer)")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"transformer_{source_mode}2{target_mode}_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved transformer comparison plot to: {save_path}")


def test_modality_pair(
    model,
    test_loader,
    source_mode,
    target_mode,
    device,
    save_dir="results",
    extra_metrics=True,
    dtw_subsample=200,
    dtw_window_ratio=0.12,
    peak_prominence_rel=0.05,
    peak_distance=5,
    peak_top_k=30,
):
    model.eval()
    mode_map = {"ir": 0, "uv": 1, "raman": 2}
    target_mode_idx = mode_map[target_mode]

    all_metrics = {
        "mse": [], "rmse": [], "mae": [], "mape": [],
        "r2": [], "pearson": [], "psnr": [], "ssim": [], "js_div": [],
    }
    if extra_metrics:
        all_metrics["dtw"] = []
        all_metrics["peak_count_match"] = []
        all_metrics["peak_pos_mae"] = []
        all_metrics["peak_height_mae"] = []

    original_spectra = []
    predicted_spectra = []

    with torch.no_grad():
        for batch in test_loader:
            source_batch = batch[0].to(device)
            target_batch = batch[1]
            target_min_batch = batch[4]
            target_max_batch = batch[5]

            pred = model(source_batch, target_mode=target_mode_idx).cpu().numpy()
            target_np = target_batch.numpy()

            for i in range(pred.shape[0]):
                pred_spec = inverse_heatmap_to_spectrum(
                    pred[i, 0],
                    float(target_min_batch[i]),
                    float(target_max_batch[i]),
                )
                target_spec = inverse_heatmap_to_spectrum(
                    target_np[i, 0],
                    float(target_min_batch[i]),
                    float(target_max_batch[i]),
                )

                metrics = calculate_metrics(target_spec, pred_spec)
                if extra_metrics:
                    metrics["dtw"] = compute_dtw_metric(
                        target_spec, pred_spec,
                        subsample=dtw_subsample,
                        window_ratio=dtw_window_ratio,
                    )
                    metrics.update(
                        peak_matching_metrics(
                            target_spec, pred_spec,
                            prominence_rel=peak_prominence_rel,
                            distance=peak_distance,
                            top_k=peak_top_k,
                        )
                    )
                for key in all_metrics:
                    value = metrics.get(key)
                    if value is None:
                        continue
                    if isinstance(value, (float, np.floating)) and np.isnan(value):
                        continue
                    all_metrics[key].append(value)

                if len(original_spectra) < 5:
                    original_spectra.append(target_spec)
                    predicted_spectra.append(pred_spec)

    print(f"\n{'=' * 60}")
    print(f"Transformer Test Results: {source_mode} -> {target_mode}")
    print(f"{'=' * 60}")
    print(f"MSE:      {np.mean(all_metrics['mse']):.6e} +/- {np.std(all_metrics['mse']):.6e}")
    print(f"RMSE:     {np.mean(all_metrics['rmse']):.6e} +/- {np.std(all_metrics['rmse']):.6e}")
    print(f"MAE:      {np.mean(all_metrics['mae']):.6e} +/- {np.std(all_metrics['mae']):.6e}")
    print(f"MAPE:     {np.mean(all_metrics['mape']):.4f}% +/- {np.std(all_metrics['mape']):.4f}%")
    print(f"R2:       {np.mean(all_metrics['r2']):.6f} +/- {np.std(all_metrics['r2']):.6f}")
    print(f"Pearson:  {np.mean(all_metrics['pearson']):.6f} +/- {np.std(all_metrics['pearson']):.6f}")
    if all_metrics["psnr"]:
        print(f"PSNR:     {np.mean(all_metrics['psnr']):.6f} +/- {np.std(all_metrics['psnr']):.6f}")
    if all_metrics["ssim"]:
        print(f"SSIM:     {np.mean(all_metrics['ssim']):.6f} +/- {np.std(all_metrics['ssim']):.6f}")
    if all_metrics["js_div"]:
        print(f"JS Div:   {np.mean(all_metrics['js_div']):.6f} +/- {np.std(all_metrics['js_div']):.6f}")
    if extra_metrics and all_metrics.get("dtw"):
        dtw_vals = np.asarray(all_metrics["dtw"], dtype=np.float64)
        print(f"DTW:      {np.nanmean(dtw_vals):.6f} +/- {np.nanstd(dtw_vals):.6f}")
    if extra_metrics and all_metrics.get("peak_count_match"):
        pcm = np.asarray(all_metrics["peak_count_match"])
        print(f"Peak cnt acc: {np.mean(pcm):.4f}")
    if extra_metrics and all_metrics.get("peak_pos_mae"):
        pp = np.asarray(all_metrics["peak_pos_mae"], dtype=np.float64)
        print(f"Peak pos MAE (matched): {np.nanmean(pp):.4f} +/- {np.nanstd(pp):.4f}")
    if extra_metrics and all_metrics.get("peak_height_mae"):
        ph = np.asarray(all_metrics["peak_height_mae"], dtype=np.float64)
        print(f"Peak hgt MAE (matched): {np.nanmean(ph):.6e} +/- {np.nanstd(ph):.6e}")

    if original_spectra:
        plot_comparison(original_spectra, predicted_spectra, save_dir, source_mode, target_mode)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Test legacy patch Transformer baseline")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--source_mode", type=str, required=True, choices=["ir", "uv", "raman"])
    parser.add_argument("--target_mode", type=str, required=True, choices=["ir", "uv", "raman"])
    parser.add_argument("--source_csv", type=str, default=None)
    parser.add_argument("--target_csv", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4, help="Attention heads used by the checkpoint")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--heatmap_size", type=int, default=3600)
    parser.add_argument("--resize_shape", type=int, nargs=2, default=[60, 60])
    parser.add_argument("--source_size", type=int, default=None)
    parser.add_argument("--target_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--no_split", action="store_true",
                        help="Use full dataset as test. QM9S/QMe14S default is training split test subset.")
    parser.add_argument("--use_split_test", action="store_true",
                        help="Force random_split test subset, even with explicit CSVs.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_extra_metrics", action="store_true")
    parser.add_argument("--dtw_subsample", type=int, default=200)
    parser.add_argument("--dtw_window_ratio", type=float, default=0.12)
    parser.add_argument("--peak_prominence_rel", type=float, default=0.05)
    parser.add_argument("--peak_distance", type=int, default=5)
    parser.add_argument("--peak_top_k", type=int, default=30)
    args = parser.parse_args()

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

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"transformer_{args.source_mode}2{args.target_mode}_best_seed{args.seed}.pt",
    )
    if not os.path.exists(checkpoint_path):
        checkpoint_path_old = os.path.join(
            args.checkpoint_dir,
            f"transformer_{args.source_mode}2{args.target_mode}_best.pt",
        )
        if os.path.exists(checkpoint_path_old):
            checkpoint_path = checkpoint_path_old
            print(f"Warning: using old checkpoint without seed: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Error: transformer checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model_kwargs = infer_transformer_config(
        state_dict,
        resize_shape=args.resize_shape,
        fallback_heads=args.num_heads,
        fallback_dropout=args.dropout,
    )
    print(
        "Inferred Transformer config: "
        f"hidden_dim={model_kwargs['hidden_dim']}, depth={model_kwargs['depth']}, "
        f"patch_size={model_kwargs['patch_size']}, num_heads={model_kwargs['num_heads']}, "
        f"mlp_ratio={model_kwargs['mlp_ratio']:.2f}, "
        f"final_norm='{model_kwargs['final_norm_name']}', output='{model_kwargs['output_name']}'"
    )

    model = LegacyPatchTransformer(**model_kwargs).to(device)
    load_result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)
    if missing_keys or unexpected_keys:
        print("Error: checkpoint/model mismatch detected.")
        if missing_keys:
            print("Missing keys:")
            for key in missing_keys[:30]:
                print(f"  - {key}")
        if unexpected_keys:
            print("Unexpected keys:")
            for key in unexpected_keys[:30]:
                print(f"  - {key}")
        raise RuntimeError("Transformer checkpoint does not match the inferred legacy model.")
    print(f"Loaded transformer checkpoint from: {checkpoint_path}")

    data_dir = Path(args.data_dir)
    if args.source_csv:
        source_csv = Path(args.source_csv) if os.path.isabs(args.source_csv) else data_dir / args.source_csv
    else:
        source_csv = data_dir / f"{args.source_mode}_broaden_processed.csv"

    if args.target_csv:
        target_csv = Path(args.target_csv) if os.path.isabs(args.target_csv) else data_dir / args.target_csv
    else:
        target_csv = data_dir / f"{args.target_mode}_broaden_processed.csv"

    if not source_csv.exists() or not target_csv.exists():
        print("Error: Data files not found")
        print(f"Source: {source_csv}")
        print(f"Target: {target_csv}")
        return

    if args.use_split_test and args.no_split:
        raise ValueError("--use_split_test and --no_split cannot be used together.")

    if args.use_split_test:
        use_full_as_test = False
    elif paired_dataset_use_random_split_by_default(args.data_dir, args.source_csv, args.target_csv):
        use_full_as_test = args.no_split
    else:
        use_full_as_test = args.no_split or (args.source_csv is not None) or (args.target_csv is not None)

    if use_full_as_test:
        dataset = PairedModalDataset(
            str(source_csv), str(target_csv),
            source_size=args.source_size,
            target_size=args.target_size,
            heatmap_size=args.heatmap_size,
            resize_shape=tuple(args.resize_shape),
            out_channels=1,
            use_h5=True,
        )
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        print("Data split: using FULL dataset as test (no random_split).")
    else:
        _, _, test_loader = get_paired_loaders(
            str(source_csv), str(target_csv),
            batch_size=args.batch_size,
            source_size=args.source_size,
            target_size=args.target_size,
            heatmap_size=args.heatmap_size,
            resize_shape=tuple(args.resize_shape),
            out_channels=1,
            seed=args.seed,
        )
        print("Data split: using random_split(test subset) to match training split.")

    print(f"Test dataset size: {len(test_loader.dataset)}")
    test_modality_pair(
        model, test_loader,
        args.source_mode, args.target_mode,
        device, args.save_dir,
        extra_metrics=not args.no_extra_metrics,
        dtw_subsample=args.dtw_subsample,
        dtw_window_ratio=args.dtw_window_ratio,
        peak_prominence_rel=args.peak_prominence_rel,
        peak_distance=args.peak_distance,
        peak_top_k=args.peak_top_k,
    )
    print("\nTransformer testing completed!")


if __name__ == "__main__":
    main()
