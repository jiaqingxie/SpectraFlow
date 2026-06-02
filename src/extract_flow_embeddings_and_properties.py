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
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, AllChem

from model_flow import ConditionalFlowMatching
from utils import get_device
from train import PairedModalDataset


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
        }

    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
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


def compute_flow_embedding(model: ConditionalFlowMatching,
                           source_heatmap: torch.Tensor,
                           target_mode_idx: int,
                           t_scalar: float = 0.5):
    """
    计算 Flow 模型中的条件嵌入 c_t = Emb_t(t) + Emb_m(m) + Enc(x0)
    不需要改模型结构，直接复用内部模块。

    Args:
        model: 已加载权重的 ConditionalFlowMatching
        source_heatmap: 源模态热图 (B, 1, H, W)
        target_mode_idx: 目标模态索引 (0=IR,1=UV,2=Raman)
        t_scalar: 使用的时间标量，例如 0.5

    Returns:
        c_t: 条件嵌入 (B, D)
    """
    vf = model.velocity_field
    device = source_heatmap.device
    B = source_heatmap.size(0)

    # 时间嵌入
    t = torch.full((B,), float(t_scalar), device=device)
    t_emb = vf.time_embed(t)  # (B, D)

    # 模态嵌入
    target_mode_tensor = torch.full(
        (B,), int(target_mode_idx), dtype=torch.long, device=device
    )
    mode_emb = vf.mode_embed(target_mode_tensor)  # (B, D)

    # 源内容编码 Enc(x0)
    x0_emb = vf.condition_encoder(source_heatmap)  # (B, D)

    c_t = t_emb + mode_emb + x0_emb
    return c_t


def extract_embeddings_and_properties(args):
    device = get_device(args.cpu)
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")

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
    ).to(device)

    # 加载 checkpoint
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"flow_{args.source_mode}2{args.target_mode}_best_seed{args.seed}.pt",
    )
    if not os.path.exists(checkpoint_path):
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
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Dataset size: {len(dataset)}")

    # 读取 SMILES（假设行数与样本数一致）
    smiles_path = Path(args.smiles_path)
    if not smiles_path.exists():
        print(f"Error: SMILES file not found at {smiles_path}")
        return
    with open(smiles_path, "r", encoding="utf-8") as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    if len(smiles_list) != len(dataset):
        print(
            f"Warning: SMILES count ({len(smiles_list)}) != dataset size ({len(dataset)}). "
            "Will truncate to the minimum."
        )
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
                model, source_heatmap, target_mode_idx, t_scalar=args.t_for_embedding
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
    import pandas as pd

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
        "--smiles_path",
        type=str,
        required=True,
        help="Path to SMILES txt file aligned with the test set "
        "(e.g., *_test_raman_smiles.txt)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for extraction"
    )
    parser.add_argument(
        "--hidden_channels", type=int, default=128, help="Hidden channels"
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
        help="Time scalar t used for computing c_t embedding (0-1)",
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

