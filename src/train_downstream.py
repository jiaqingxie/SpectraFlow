"""
端到端下游任务脚本：

- 读取 extract_flow_embeddings_and_properties.py 生成的特征：
  - IR raw 光谱特征 (.npy)
  - Raman raw 光谱特征 (.npy)
  - Flow embedding 特征 (.npy)
  - 分子性质标签 (.csv)

- 使用简单的 sklearn 模型做性质预测（默认回归）
  - 支持对比：IR raw / Raman raw / Flow embedding
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


def load_features_and_labels(base_dir: Path, source_mode: str, target_mode: str):
    """
从指定目录加载特征和标签文件。
    """
    ir_path = base_dir / f"ir_raw_{source_mode}2{target_mode}.npy"
    raman_path = base_dir / f"raman_raw_{source_mode}2{target_mode}.npy"
    flow_path = base_dir / f"flow_embedding_{source_mode}2{target_mode}.npy"
    props_path = base_dir / f"molecular_properties_{source_mode}2{target_mode}.csv"
    fp_path = base_dir / f"fingerprint_morgan_{source_mode}2{target_mode}.npy"

    if not ir_path.exists() or not raman_path.exists() or not flow_path.exists() or not props_path.exists():
        raise FileNotFoundError(
            f"Required files not found in {base_dir}. "
            f"Make sure you have run extract_flow_embeddings_and_properties.py first."
        )

    ir = np.load(ir_path)
    raman = np.load(raman_path)
    flow = np.load(flow_path)
    fp = np.load(fp_path) if fp_path.exists() else None
    props_df = pd.read_csv(props_path)

    # 对齐长度（以最小 N 为准）
    N_list = [ir.shape[0], raman.shape[0], flow.shape[0], len(props_df)]
    if fp is not None:
        N_list.append(fp.shape[0])
    N = min(N_list)
    ir = ir[:N]
    raman = raman[:N]
    flow = flow[:N]
    if fp is not None:
        fp = fp[:N]
    props_df = props_df.iloc[:N].reset_index(drop=True)

    return ir, raman, flow, fp, props_df


def train_and_eval_one_feature(
    X: np.ndarray,
    y: np.ndarray,
    feature_name: str,
    target_name: str,
    model_type: str = "ridge",
    ridge_alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    对单一种特征训练一个简单回归模型，并打印指标。
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if model_type == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha, random_state=random_state))
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    # 一些较旧版本的sklearn不支持 squared 参数，这里手动计算 RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))

    print(f"\n=== Downstream: {target_name} | Features: {feature_name} | Model: {model_type} ===")
    print(f"R^2   : {r2:.4f}")
    print(f"RMSE  : {rmse:.4f}")

    return {"r2": r2, "rmse": rmse}


def cross_validate_feature(
    X: np.ndarray,
    y: np.ndarray,
    feature_name: str,
    target_name: str,
    model_type: str = "ridge",
    ridge_alpha: float = 1.0,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    使用 K 折交叉验证评估单一种特征。
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    r2_list = []
    rmse_list = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model_type == "ridge":
            model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha, random_state=random_state))
        elif model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2_list.append(r2)
        rmse_list.append(rmse)

        print(
            f"[Fold {fold}/{n_splits}] {feature_name} | {target_name} | "
            f"R^2={r2:.4f}, RMSE={rmse:.4f}"
        )

    r2_mean = float(np.mean(r2_list))
    r2_std = float(np.std(r2_list))
    rmse_mean = float(np.mean(rmse_list))
    rmse_std = float(np.std(rmse_list))

    print(
        f"\n=== CV Summary: {target_name} | Features: {feature_name} | "
        f"Model: {model_type} | Folds: {n_splits} ==="
    )
    print(f"R^2   : {r2_mean:.4f} ± {r2_std:.4f}")
    print(f"RMSE  : {rmse_mean:.4f} ± {rmse_std:.4f}")

    return {"r2_mean": r2_mean, "r2_std": r2_std, "rmse_mean": rmse_mean, "rmse_std": rmse_std}


def main():
    parser = argparse.ArgumentParser(
        description="Train simple downstream models on IR/Raman/Flow embeddings."
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help="Directory containing *.npy and properties CSV from extract_flow_embeddings_and_properties.py",
    )
    parser.add_argument(
        "--source_mode",
        type=str,
        required=True,
        choices=["ir", "uv", "raman"],
        help="Source modality used when extracting features",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        required=True,
        choices=["ir", "uv", "raman"],
        help="Target modality used when extracting features",
    )
    parser.add_argument(
        "--target_property",
        type=str,
        default="LogP",
        help="Which molecular property column to predict (e.g., MolWt, LogP, TPSA)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ridge",
        choices=["ridge", "rf"],
        help="Downstream model type: ridge (linear) or rf (Random Forest)",
    )
    parser.add_argument(
        "--ridge_alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength. Increase this (e.g. 10 or 100) "
        "if ridge reports ill-conditioned matrix warnings.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split ratio",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=1,
        help="If >1, use K-fold cross validation instead of a single train/test split",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=1,
        help="If cv_folds==1, run multiple seeds and report variability",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    base_dir = Path(args.feature_dir)
    ir, raman, flow, fp, props_df = load_features_and_labels(
        base_dir, args.source_mode, args.target_mode
    )

    if args.target_property not in props_df.columns:
        raise ValueError(
            f"Property {args.target_property} not found in {props_df.columns.tolist()}"
        )

    y = props_df[args.target_property].values.astype(np.float32)
    mask = ~np.isnan(y)
    ir = ir[mask]
    raman = raman[mask]
    flow = flow[mask]
    if fp is not None:
        fp = fp[mask]
    y = y[mask]

    print(
        f"Using {len(y)} samples with valid {args.target_property} "
        f"for downstream regression."
    )

    # 如果有指纹，并且维度匹配，构造 Flow+FP 拼接特征
    flow_fp = None
    if fp is not None and fp.shape[0] == flow.shape[0]:
        try:
            flow_fp = np.concatenate([flow, fp], axis=1)
            print(f"Using concatenated features Flow+FP with shape {flow_fp.shape}")
        except Exception:
            flow_fp = None

    if args.cv_folds > 1:
        # K 折交叉验证
        print(f"\n>>> Using {args.cv_folds}-fold cross validation")
        cross_validate_feature(
            ir,
            y,
            feature_name="IR raw",
            target_name=args.target_property,
            model_type=args.model_type,
            ridge_alpha=args.ridge_alpha,
            n_splits=args.cv_folds,
            random_state=args.seed,
        )
        cross_validate_feature(
            raman,
            y,
            feature_name="Raman raw",
            target_name=args.target_property,
            model_type=args.model_type,
            ridge_alpha=args.ridge_alpha,
            n_splits=args.cv_folds,
            random_state=args.seed,
        )
        cross_validate_feature(
            flow,
            y,
            feature_name="Flow embedding",
            target_name=args.target_property,
            model_type=args.model_type,
            ridge_alpha=args.ridge_alpha,
            n_splits=args.cv_folds,
            random_state=args.seed,
        )
        if fp is not None:
            cross_validate_feature(
                fp,
                y,
                feature_name="Morgan fingerprint",
                target_name=args.target_property,
                model_type=args.model_type,
                ridge_alpha=args.ridge_alpha,
                n_splits=args.cv_folds,
                random_state=args.seed,
            )
        if flow_fp is not None:
            cross_validate_feature(
                flow_fp,
                y,
                feature_name="Flow + Morgan FP",
                target_name=args.target_property,
                model_type=args.model_type,
                ridge_alpha=args.ridge_alpha,
                n_splits=args.cv_folds,
                random_state=args.seed,
            )
    else:
        # 多 seed 简单重复实验，并统计 mean/std
        print(f"\n>>> Using {args.num_seeds} random seeds (single train/test split each)")
        stats_ir = {"r2": [], "rmse": []}
        stats_raman = {"r2": [], "rmse": []}
        stats_flow = {"r2": [], "rmse": []}
        stats_fp = {"r2": [], "rmse": []} if fp is not None else None
        stats_flow_fp = {"r2": [], "rmse": []} if flow_fp is not None else None

        for seed in range(args.seed, args.seed + args.num_seeds):
            print(f"\n--- Seed {seed} ---")
            np.random.seed(seed)
            res_ir = train_and_eval_one_feature(
                ir,
                y,
                feature_name="IR raw",
                target_name=args.target_property,
                model_type=args.model_type,
                ridge_alpha=args.ridge_alpha,
                test_size=args.test_size,
                random_state=seed,
            )
            res_raman = train_and_eval_one_feature(
                raman,
                y,
                feature_name="Raman raw",
                target_name=args.target_property,
                model_type=args.model_type,
                ridge_alpha=args.ridge_alpha,
                test_size=args.test_size,
                random_state=seed,
            )
            res_flow = train_and_eval_one_feature(
                flow,
                y,
                feature_name="Flow embedding",
                target_name=args.target_property,
                model_type=args.model_type,
                ridge_alpha=args.ridge_alpha,
                test_size=args.test_size,
                random_state=seed,
            )
            if fp is not None:
                res_fp = train_and_eval_one_feature(
                    fp,
                    y,
                    feature_name="Morgan fingerprint",
                    target_name=args.target_property,
                    model_type=args.model_type,
                    ridge_alpha=args.ridge_alpha,
                    test_size=args.test_size,
                    random_state=seed,
                )

            stats_ir["r2"].append(res_ir["r2"])
            stats_ir["rmse"].append(res_ir["rmse"])
            stats_raman["r2"].append(res_raman["r2"])
            stats_raman["rmse"].append(res_raman["rmse"])
            stats_flow["r2"].append(res_flow["r2"])
            stats_flow["rmse"].append(res_flow["rmse"])
            if fp is not None and stats_fp is not None:
                stats_fp["r2"].append(res_fp["r2"])
                stats_fp["rmse"].append(res_fp["rmse"])

            if flow_fp is not None and stats_flow_fp is not None:
                res_flow_fp = train_and_eval_one_feature(
                    flow_fp,
                    y,
                    feature_name="Flow + Morgan FP",
                    target_name=args.target_property,
                    model_type=args.model_type,
                    ridge_alpha=args.ridge_alpha,
                    test_size=args.test_size,
                    random_state=seed,
                )
                stats_flow_fp["r2"].append(res_flow_fp["r2"])
                stats_flow_fp["rmse"].append(res_flow_fp["rmse"])

        def _print_seed_summary(name, stats):
            r2_arr = np.array(stats["r2"], dtype=float)
            rmse_arr = np.array(stats["rmse"], dtype=float)
            print(
                f"\n=== Seed Summary: {args.target_property} | Features: {name} | "
                f"Model: {args.model_type} | Seeds: {args.num_seeds} ==="
            )
            print(f"R^2   : {np.mean(r2_arr):.4f} ± {np.std(r2_arr):.4f}")
            print(f"RMSE  : {np.mean(rmse_arr):.4f} ± {np.std(rmse_arr):.4f}")

        _print_seed_summary("IR raw", stats_ir)
        _print_seed_summary("Raman raw", stats_raman)
        _print_seed_summary("Flow embedding", stats_flow)
        if stats_fp is not None:
            _print_seed_summary("Morgan fingerprint", stats_fp)
        if stats_flow_fp is not None:
            _print_seed_summary("Flow + Morgan FP", stats_flow_fp)


if __name__ == "__main__":
    main()

