"""
从已训练的 Flow Matching 模型中抽取嵌入，并用 RDKit 计算分子性质与分子指纹，
构建下游任务用的数据集：

- IR raw 光谱特征
- Raman raw 光谱特征
- Flow embedding 特征（条件嵌入 c_t）
- 分子指纹（如 Morgan fingerprint）
- 分子性质标签（RDKit 计算）
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, AllChem

from model_flow import ConditionalFlowMatching
from utils import get_device
from train import PairedModalDataset


class SavedSpectraMatrixDataset(torch.utils.data.Dataset):
    """
    Dataset for saved result matrices such as flow_ir2raman_sources.csv and
    flow_ir2raman_targets.csv. These files are plain sample x spectrum matrices
    with a header row, unlike the processed CSV format used by PairedModalDataset.
    """

    def __init__(
        self,
        source_csv,
        target_csv,
        source_size=None,
        target_size=None,
        heatmap_size=3600,
        resize_shape=(60, 60),
    ):
        source_df = pd.read_csv(source_csv).drop(columns=["index"], errors="ignore")
        target_df = pd.read_csv(target_csv).drop(columns=["index"], errors="ignore")
        self.source_data = source_df.to_numpy(dtype=np.float32)
        self.target_data = target_df.to_numpy(dtype=np.float32)
        if len(self.source_data) != len(self.target_data):
            raise ValueError(
                f"Source and target row counts differ: {len(self.source_data)} vs {len(self.target_data)}"
            )

        self.source_size = source_size if source_size is not None else heatmap_size
        self.target_size = target_size if target_size is not None else heatmap_size
        self.heatmap_size = heatmap_size
        self.resize_shape = resize_shape

        self.source_interpolated_original = [
            self._interpolate(spec, self.source_size) for spec in self.source_data
        ]
        self.target_interpolated_original = [
            self._interpolate(spec, self.target_size) for spec in self.target_data
        ]
        self.source_interpolated_for_heatmap = [
            self._interpolate(spec, self.heatmap_size) for spec in self.source_interpolated_original
        ]
        self.target_interpolated_for_heatmap = [
            self._interpolate(spec, self.heatmap_size) for spec in self.target_interpolated_original
        ]

        self.source_min_vals = [float(spec.min()) for spec in self.source_interpolated_for_heatmap]
        self.source_max_vals = [float(spec.max()) for spec in self.source_interpolated_for_heatmap]
        self.target_min_vals = [float(spec.min()) for spec in self.target_interpolated_for_heatmap]
        self.target_max_vals = [float(spec.max()) for spec in self.target_interpolated_for_heatmap]

        self.source_heatmaps = [
            self._spectrum_to_heatmap(spec, minv, maxv)
            for spec, minv, maxv in zip(
                self.source_interpolated_for_heatmap, self.source_min_vals, self.source_max_vals
            )
        ]
        self.target_heatmaps = [
            self._spectrum_to_heatmap(spec, minv, maxv)
            for spec, minv, maxv in zip(
                self.target_interpolated_for_heatmap, self.target_min_vals, self.target_max_vals
            )
        ]

    def _interpolate(self, spectrum, target_length):
        if len(spectrum) == target_length:
            return spectrum.copy()
        x_old = np.linspace(0, 1, len(spectrum))
        x_new = np.linspace(0, 1, target_length)
        return np.interp(x_new, x_old, spectrum).astype(np.float32)

    def _spectrum_to_heatmap(self, spectrum, min_val, max_val):
        norm = (spectrum - min_val) / (max_val - min_val + 1e-8)
        side = int(np.sqrt(len(norm)))
        if side * side != len(norm):
            raise ValueError(f"heatmap_size must be a perfect square, got {len(norm)}")
        patch_size = max(p for p in [10, 8, 5, 4, 2, 1] if side % p == 0)
        num_patches_per_row = side // patch_size
        patches = norm.reshape(-1, patch_size, patch_size)
        rows = [
            np.concatenate(
                patches[i * num_patches_per_row : (i + 1) * num_patches_per_row],
                axis=1,
            )
            for i in range(num_patches_per_row)
        ]
        heatmap = np.concatenate(rows, axis=0)
        return np.expand_dims(heatmap, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.source_heatmaps)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.source_heatmaps[idx]).float(),
            torch.tensor(self.target_heatmaps[idx]).float(),
            torch.tensor(self.source_min_vals[idx]).float(),
            torch.tensor(self.source_max_vals[idx]).float(),
            torch.tensor(self.target_min_vals[idx]).float(),
            torch.tensor(self.target_max_vals[idx]).float(),
            torch.tensor(self.source_interpolated_original[idx]).float(),
            torch.tensor(self.target_interpolated_original[idx]).float(),
            torch.zeros(7).float(),
        )


def compute_rdkit_properties(smiles: str):
    """
    使用 RDKit 计算一些常见的分子性质。
    如果 SMILES 无法解析，返回 NaN。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "MolWt": np.nan,
            "LogP": np.nan,
            "TPSA": np.nan,
            "NumHDonors": np.nan,
            "NumHAcceptors": np.nan,
            "NumRotatableBonds": np.nan,
            "RingCount": np.nan,
            "HeavyAtomCount": np.nan,
            "FractionCSP3": np.nan,
            "NumAromaticRings": np.nan,
            "MolMR": np.nan,
            "LabuteASA": np.nan,
        }

    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "MolMR": Crippen.MolMR(mol),
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
    }


def compute_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """
    计算 Morgan 指纹（ECFP），返回 float32 向量 (n_bits,)
    如果 SMILES 解析失败，返回全 0。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    # RDKit 将 bitvect 转为 numpy
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def load_smiles_list(smiles_path: Path):
    """
    Load SMILES in the exact row order used by the paired spectra dataset.

    Supported formats:
    - .txt: one SMILES per line
    - .csv: a column named "smiles" or "SMILES"; if absent, a single-column CSV
    """
    if smiles_path.suffix.lower() == ".csv":
        df = pd.read_csv(smiles_path)
        if "smiles" in df.columns:
            values = df["smiles"]
        elif "SMILES" in df.columns:
            values = df["SMILES"]
        elif df.shape[1] == 1:
            values = df.iloc[:, 0]
        else:
            raise ValueError(
                f"Cannot find a SMILES column in {smiles_path}. "
                f"Available columns: {df.columns.tolist()}"
            )
        return [str(x).strip() for x in values.tolist() if str(x).strip()]

    with open(smiles_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _flow_state_at_t(
    model: ConditionalFlowMatching,
    source_heatmap: torch.Tensor,
    target_mode_idx: int,
    t_scalar: float,
    num_steps: int,
    use_rk4: bool,
) -> torch.Tensor:
    """
    Return x_t from the same ODE trajectory used by model.sample().

    num_steps is the full 0->1 sampling step count, matching test_flow. For
    example, with num_steps=8 and t_scalar=0.5, this returns path[4].
    """
    if float(t_scalar) <= 0:
        return source_heatmap

    total_steps = max(1, int(num_steps))
    _, path = model.sample(
        source_heatmap,
        target_mode=target_mode_idx,
        num_steps=total_steps,
        use_rk4=use_rk4,
        return_path=True,
    )
    path_idx = int(round(float(t_scalar) * total_steps))
    path_idx = max(0, min(path_idx, len(path) - 1))
    return path[path_idx]


def compute_flow_embedding(
    model: ConditionalFlowMatching,
    source_heatmap: torch.Tensor,
    target_mode_idx: int,
    t_scalar: float = 0.5,
    num_steps: int = 75,
    use_rk4: bool = False,
    embedding_type: str = "hidden",
):
    """
    Extract a Flow embedding without retraining.

    embedding_type="conditional" reproduces the original conditioning vector:
    time embedding + target-mode embedding + source encoder.

    embedding_type="hidden" first moves the source to x_t and mean-pools the
    final DiT/VibraDiT token sequence. For legacy U-Net checkpoints, hidden
    falls back to the conditional embedding because there is no token state.
    """
    vf = model.velocity_field
    device = source_heatmap.device
    B = source_heatmap.size(0)
    t = torch.full((B,), float(t_scalar), device=device)
    target_mode_tensor = torch.full(
        (B,), int(target_mode_idx), dtype=torch.long, device=device
    )

    if embedding_type == "conditional":
        t_emb = vf.time_embed(t)
        mode_emb = vf.mode_embed(target_mode_tensor)
        if hasattr(vf, "source_encoder"):
            x_source = source_heatmap.reshape(B, -1)
            c = t_emb + mode_emb + vf.source_encoder(x_source)
            if hasattr(vf, "cond_fuse"):
                c = vf.cond_fuse(c)
            return c
        if hasattr(vf, "condition_encoder"):
            return t_emb + mode_emb + vf.condition_encoder(source_heatmap)
        raise AttributeError(f"Unsupported velocity_field: {vf.__class__.__name__}")

    if embedding_type != "hidden":
        raise ValueError(f"Unknown embedding_type: {embedding_type}")

    x_t = _flow_state_at_t(
        model,
        source_heatmap,
        target_mode_idx,
        t_scalar=t_scalar,
        num_steps=num_steps,
        use_rk4=use_rk4,
    )

    if model.backbone == "vibradit" and hasattr(vf, "x_embedder"):
        x_tau = x_t.reshape(B, -1)
        x_source = source_heatmap.reshape(B, -1)
        h = vf.x_embedder(torch.stack([x_tau, x_source], dim=1))
        h = h + vf.pos_embed.to(dtype=x_t.dtype, device=device)
        c = vf.time_embed(t) + vf.mode_embed(target_mode_tensor) + vf.source_encoder(x_source)
        c = vf.cond_fuse(c)
        for block in vf.blocks:
            h = block(h, c)
        return h.mean(dim=1)

    if model.backbone == "dit" and hasattr(vf, "_patchify"):
        c = vf._build_condition(x_t, t, source_heatmap, target_mode_idx)
        tokens = vf._patchify(x_t) + vf.pos_embed
        for block in vf.blocks:
            tokens = block(tokens, c)
        return tokens.mean(dim=1)

    if hasattr(vf, "condition_encoder"):
        t_emb = vf.time_embed(t)
        mode_emb = vf.mode_embed(target_mode_tensor)
        x0_emb = vf.condition_encoder(source_heatmap)
        return t_emb + mode_emb + x0_emb

    raise AttributeError(f"Unsupported velocity_field: {vf.__class__.__name__}")


def extract_embeddings_and_properties(args):
    device = get_device(args.cpu)
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")
    print(
        f"Flow embedding type: {args.embedding_type} at "
        f"t={args.t_for_embedding} after {args.embedding_num_steps} ODE steps "
        f"({'RK4' if args.embedding_use_rk4 else 'Euler'})"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 创建模型
    model = ConditionalFlowMatching(
        in_channels=1,
        hidden_channels=args.hidden_channels,
        num_modes=3,
        image_size=tuple(args.resize_shape),
        sigma_min=0.01,
        backbone=args.backbone,
        dit_hidden_dim=args.dit_hidden_dim,
        dit_depth=args.dit_depth,
        dit_num_heads=args.dit_num_heads,
        dit_patch_size=args.dit_patch_size,
    ).to(device)
    print(f"Backbone: {args.backbone}")
    print(f"Velocity field: {model.velocity_field.__class__.__name__}")

    # 加载 checkpoint
    backbone_tag = f"_{args.backbone}" if args.backbone != "unet" else ""
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"flow_{args.source_mode}2{args.target_mode}{backbone_tag}_best_seed{args.seed}.pt",
    )
    if args.backbone == "unet" and not os.path.exists(checkpoint_path):
        checkpoint_path_old = os.path.join(
            args.checkpoint_dir,
            f"flow_{args.source_mode}2{args.target_mode}_best.pt",
        )
        if os.path.exists(checkpoint_path_old):
            checkpoint_path = checkpoint_path_old
            print(
                f"Warning: Using old checkpoint format (without seed). "
                f"Consider retraining with --seed {args.seed}"
            )

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print("Warning: checkpoint loaded with non-strict key mismatch.")
        print(f"  missing keys: {len(load_result.missing_keys)}")
        print(f"  unexpected keys: {len(load_result.unexpected_keys)}")
        print("  First missing:", load_result.missing_keys[:5])
        print("  First unexpected:", load_result.unexpected_keys[:5])
    model.eval()
    print(f"Loaded checkpoint from: {checkpoint_path}")

    # 加载数据集（与 test_flow 一致）
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

    if args.saved_matrix_format:
        dataset = SavedSpectraMatrixDataset(
            str(source_csv),
            str(target_csv),
            source_size=args.source_size,
            target_size=args.target_size,
            heatmap_size=args.heatmap_size,
            resize_shape=tuple(args.resize_shape),
        )
        print("Data format: saved spectra matrix CSV (sample x spectrum).")
    else:
        dataset = PairedModalDataset(
            str(source_csv),
            str(target_csv),
            source_size=args.source_size,
            target_size=args.target_size,
            heatmap_size=args.heatmap_size,
            resize_shape=tuple(args.resize_shape),
            out_channels=1,
            use_h5=True,
        )
        print("Data format: processed paired modal dataset.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Dataset size: {len(dataset)}")

    # 读取 SMILES：必须与 dataset 的行顺序一一对应。
    smiles_path = Path(args.smiles_path)
    if not smiles_path.exists():
        print(f"Error: SMILES file not found at {smiles_path}")
        return
    smiles_list = load_smiles_list(smiles_path)
    if len(smiles_list) != len(dataset):
        msg = (
            f"SMILES count ({len(smiles_list)}) != dataset size ({len(dataset)}). "
            "Use a SMILES file aligned to the exact source/target spectra row order."
        )
        if not args.allow_truncate:
            raise ValueError(msg + " Pass --allow_truncate only if you intentionally want the overlap.")
        print("Warning: " + msg + " Truncating to the minimum because --allow_truncate was set.")
    N = min(len(smiles_list), len(dataset))

    # 准备存储
    ir_raw_list = []
    raman_raw_list = []
    flow_emb_list = []
    fp_list = []
    prop_rows = []

    mode_map = {"ir": 0, "uv": 1, "raman": 2}
    target_mode_idx = mode_map[args.target_mode]

    global_idx = 0
    for batch in tqdm(loader, desc="Extracting embeddings and properties"):
        if global_idx >= N:
            break

        source_heatmap = batch[0].to(device)          # (B, 1, H, W)
        # batch[6], batch[7] 是原始长度的源/目标光谱
        source_orig = batch[6].numpy()                # (B, L_src)
        target_orig = batch[7].numpy()                # (B, L_tgt)

        B = source_heatmap.size(0)
        # 只处理到 N
        if global_idx + B > N:
            valid_B = N - global_idx
            source_heatmap = source_heatmap[:valid_B]
            source_orig = source_orig[:valid_B]
            target_orig = target_orig[:valid_B]
            B = valid_B

        # 计算 Flow embedding
        with torch.no_grad():
            c_t = compute_flow_embedding(
                model,
                source_heatmap,
                target_mode_idx,
                t_scalar=args.t_for_embedding,
                num_steps=args.embedding_num_steps,
                use_rk4=args.embedding_use_rk4,
                embedding_type=args.embedding_type,
            )  # (B, D)
        c_t_np = c_t.cpu().numpy()

        for i in range(B):
            idx = global_idx + i
            smiles = smiles_list[idx]
            props = compute_rdkit_properties(smiles)
            fp_vec = compute_morgan_fingerprint(
                smiles, radius=args.fp_radius, n_bits=args.fp_n_bits
            )

            ir_raw_list.append(source_orig[i])
            raman_raw_list.append(target_orig[i])
            flow_emb_list.append(c_t_np[i])
            fp_list.append(fp_vec)

            row = {"index": idx, "smiles": smiles}
            row.update(props)
            prop_rows.append(row)

        global_idx += B

    ir_raw = np.vstack(ir_raw_list)
    raman_raw = np.vstack(raman_raw_list)
    flow_emb = np.vstack(flow_emb_list)
    fp_arr = np.vstack(fp_list)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存特征为 .npy
    np.save(out_dir / f"ir_raw_{args.source_mode}2{args.target_mode}.npy", ir_raw)
    np.save(out_dir / f"raman_raw_{args.source_mode}2{args.target_mode}.npy", raman_raw)
    np.save(out_dir / f"flow_embedding_{args.source_mode}2{args.target_mode}.npy", flow_emb)
    np.save(
        out_dir / f"fingerprint_morgan_{args.source_mode}2{args.target_mode}.npy",
        fp_arr,
    )

    # 保存分子性质为 CSV
    df_props = pd.DataFrame(prop_rows)
    df_props.to_csv(
        out_dir / f"molecular_properties_{args.source_mode}2{args.target_mode}.csv",
        index=False,
    )

    print("\nSaved downstream dataset:")
    print(f"  IR raw features:        {out_dir / f'ir_raw_{args.source_mode}2{args.target_mode}.npy'}  shape={ir_raw.shape}")
    print(f"  Raman raw features:     {out_dir / f'raman_raw_{args.source_mode}2{args.target_mode}.npy'}  shape={raman_raw.shape}")
    print(f"  Flow embeddings:        {out_dir / f'flow_embedding_{args.source_mode}2{args.target_mode}.npy'}  shape={flow_emb.shape}")
    print(f"  Morgan fingerprints:    {out_dir / f'fingerprint_morgan_{args.source_mode}2{args.target_mode}.npy'}  shape={fp_arr.shape}")
    print(f"  Molecular properties:   {out_dir / f'molecular_properties_{args.source_mode}2{args.target_mode}.csv'}  rows={len(df_props)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Flow embeddings and RDKit properties for downstream tasks"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed CSV files",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--source_mode",
        type=str,
        required=True,
        choices=["ir", "uv", "raman"],
        help="Source modality",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        required=True,
        choices=["ir", "uv", "raman"],
        help="Target modality",
    )
    parser.add_argument(
        "--source_csv",
        type=str,
        default=None,
        help="Custom source CSV filename (optional)",
    )
    parser.add_argument(
        "--target_csv",
        type=str,
        default=None,
        help="Custom target CSV filename (optional)",
    )
    parser.add_argument(
        "--saved_matrix_format",
        action="store_true",
        help="Read --source_csv/--target_csv as saved result matrices "
        "(e.g. flow_ir2raman_sources.csv / flow_ir2raman_targets.csv) "
        "instead of processed spectral CSV/H5 files.",
    )
    parser.add_argument(
        "--smiles_path",
        type=str,
        required=True,
        help="Path to SMILES txt/csv aligned with the exact dataset row order. "
        "CSV files may contain a 'smiles' or 'SMILES' column.",
    )
    parser.add_argument(
        "--allow_truncate",
        action="store_true",
        help="Allow truncating when SMILES count and dataset size differ. "
        "Default is to fail fast to avoid misaligned property labels.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for extraction"
    )
    parser.add_argument(
        "--hidden_channels", type=int, default=128, help="Hidden channels"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="unet",
        choices=["unet", "dit", "vibradit"],
        help="Flow backbone. Must match the checkpoint. Use 'vibradit' for flow_*_vibradit checkpoints.",
    )
    parser.add_argument(
        "--dit_hidden_dim",
        type=int,
        default=256,
        help="[DiT/VibraDiT] Transformer hidden dimension",
    )
    parser.add_argument(
        "--dit_depth",
        type=int,
        default=6,
        help="[DiT/VibraDiT] Number of transformer blocks",
    )
    parser.add_argument(
        "--dit_num_heads",
        type=int,
        default=4,
        help="[DiT/VibraDiT] Number of attention heads",
    )
    parser.add_argument(
        "--dit_patch_size",
        type=int,
        default=4,
        help="[DiT/VibraDiT] Patch size",
    )
    parser.add_argument(
        "--heatmap_size",
        type=int,
        default=3600,
        help="Heatmap size (must be a perfect square)",
    )
    parser.add_argument(
        "--resize_shape",
        type=int,
        nargs=2,
        default=[60, 60],
        help="Heatmap reshape size, e.g., 60 60 or 32 32",
    )
    parser.add_argument(
        "--source_size",
        type=int,
        default=None,
        help="Optional source spectrum length before heatmap",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=None,
        help="Optional target spectrum length before heatmap",
    )
    parser.add_argument(
        "--t_for_embedding",
        type=float,
        default=0.5,
        help="Time scalar t used for hidden-state Flow embedding extraction (0-1)",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="hidden",
        choices=["hidden", "conditional"],
        help="Flow embedding to save: hidden pools DiT/VibraDiT token states; "
        "conditional saves the original conditioning vector.",
    )
    parser.add_argument(
        "--embedding_num_steps",
        type=int,
        default=75,
        help="ODE steps used to move the source spectrum to --t_for_embedding before pooling hidden states",
    )
    parser.add_argument(
        "--embedding_use_rk4",
        action="store_true",
        help="Use RK4 instead of Euler when moving to --t_for_embedding for hidden-state extraction",
    )
    parser.add_argument(
        "--fp_radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius",
    )
    parser.add_argument(
        "--fp_n_bits",
        type=int,
        default=2048,
        help="Morgan fingerprint length (number of bits)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="downstream",
        help="Output directory for features and properties",
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    extract_embeddings_and_properties(args)


if __name__ == "__main__":
    main()

