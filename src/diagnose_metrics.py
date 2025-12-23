"""
诊断评估指标异常的问题
分析为什么MSE/MAE较小但R²为负数、MAPE很高
"""
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def diagnose_metrics(original, reconstructed, title="Diagnostic Analysis"):
    """
    详细诊断预测结果
    
    Args:
        original: 真实值数组 (N, length) 或 (length,)
        reconstructed: 预测值数组 (N, length) 或 (length,)
    """
    original = np.array(original).flatten()
    reconstructed = np.array(reconstructed).flatten()
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # 1. 基本统计
    print(f"\n1. Data Statistics:")
    print(f"   Original:      shape={original.shape}, mean={original.mean():.6f}, std={original.std():.6f}")
    print(f"                  min={original.min():.6f}, max={original.max():.6f}")
    print(f"   Reconstructed: shape={reconstructed.shape}, mean={reconstructed.mean():.6f}, std={reconstructed.std():.6f}")
    print(f"                  min={reconstructed.min():.6f}, max={reconstructed.max():.6f}")
    
    # 2. 误差分析
    errors = original - reconstructed
    abs_errors = np.abs(errors)
    print(f"\n2. Error Statistics:")
    print(f"   Mean error:           {errors.mean():.6f}")
    print(f"   Mean absolute error:  {abs_errors.mean():.6f}")
    print(f"   Max absolute error:   {abs_errors.max():.6f}")
    print(f"   Error std:            {errors.std():.6f}")
    
    # 3. 评估指标
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_errors)
    
    # MAPE (注意接近0的值)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.abs(errors / (original + 1e-8)) * 100
    mape_valid = mape[np.isfinite(mape)]
    print(f"\n3. Metrics:")
    print(f"   MSE:    {mse:.6e}")
    print(f"   RMSE:   {rmse:.6e}")
    print(f"   MAE:    {mae:.6e}")
    print(f"   MAPE:   {np.mean(mape_valid):.4f}% (valid: {len(mape_valid)}/{len(original)})")
    print(f"           min={np.min(mape_valid):.4f}%, max={np.max(mape_valid):.4f}%, std={np.std(mape_valid):.4f}%")
    
    # 4. R² 详细分析
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((original - original.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    print(f"\n4. R² Analysis:")
    print(f"   SS_res (residual sum of squares): {ss_res:.6e}")
    print(f"   SS_tot (total sum of squares):    {ss_tot:.6e}")
    print(f"   R²:                                {r2:.6f}")
    if r2 < 0:
        print(f"   ⚠️  Negative R² means model is worse than predicting the mean!")
        print(f"      This happens when SS_res > SS_tot")
        print(f"      Ratio (SS_res/SS_tot): {ss_res/ss_tot:.6f}")
    
    # 5. Pearson 相关性
    try:
        pearson, p_value = pearsonr(original, reconstructed)
        print(f"\n5. Pearson Correlation:")
        print(f"   Pearson: {pearson:.6f}")
        print(f"   p-value: {p_value:.6e}")
    except:
        print(f"\n5. Pearson Correlation: Cannot compute")
    
    # 6. 检查接近0的值（可能导致MAPE爆炸）
    zero_threshold = 0.01
    near_zero_original = np.abs(original) < zero_threshold
    near_zero_pred = np.abs(reconstructed) < zero_threshold
    
    print(f"\n6. Near-zero Values Analysis:")
    print(f"   Original values near zero (<{zero_threshold}): {near_zero_original.sum()}/{len(original)} ({near_zero_original.sum()/len(original)*100:.2f}%)")
    print(f"   Predicted values near zero (<{zero_threshold}): {near_zero_pred.sum()}/{len(reconstructed)} ({near_zero_pred.sum()/len(reconstructed)*100:.2f}%)")
    
    if near_zero_original.sum() > 0:
        print(f"   ⚠️  Many near-zero values can cause MAPE to explode!")
        mape_near_zero = mape_valid[near_zero_original]
        mape_far_zero = mape_valid[~near_zero_original]
        if len(mape_near_zero) > 0:
            print(f"      MAPE for near-zero: {np.mean(mape_near_zero):.2f}% (n={len(mape_near_zero)})")
        if len(mape_far_zero) > 0:
            print(f"      MAPE for far-from-zero: {np.mean(mape_far_zero):.2f}% (n={len(mape_far_zero)})")
    
    # 7. 检查极端误差
    print(f"\n7. Extreme Errors:")
    top_10_errors_idx = np.argsort(abs_errors)[-10:][::-1]
    print(f"   Top 10 largest absolute errors:")
    for i, idx in enumerate(top_10_errors_idx[:5]):
        print(f"      {i+1}. idx={idx}: orig={original[idx]:.6f}, pred={reconstructed[idx]:.6f}, error={errors[idx]:.6f}")
    
    # 8. 分布分析
    print(f"\n8. Distribution Analysis:")
    print(f"   Original variance:      {original.var():.6e}")
    print(f"   Reconstructed variance: {reconstructed.var():.6e}")
    print(f"   Variance ratio:         {reconstructed.var()/(original.var()+1e-8):.6f}")
    
    if reconstructed.var() < original.var() * 0.5:
        print(f"   ⚠️  Model is predicting less variance than original (possible underfitting)")
    elif reconstructed.var() > original.var() * 1.5:
        print(f"   ⚠️  Model is predicting more variance than original (possible overfitting)")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': np.mean(mape_valid),
        'r2': r2,
        'pearson': pearson if 'pearson' in locals() else np.nan,
        'ss_res': ss_res,
        'ss_tot': ss_tot,
        'variance_ratio': reconstructed.var()/(original.var()+1e-8),
    }


def plot_diagnosis(original, reconstructed, save_path=None):
    """绘制诊断图"""
    original = np.array(original).flatten()
    reconstructed = np.array(reconstructed).flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 散点图：真实值 vs 预测值
    ax1 = axes[0, 0]
    ax1.scatter(original, reconstructed, alpha=0.3, s=1)
    min_val = min(original.min(), reconstructed.min())
    max_val = max(original.max(), reconstructed.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    ax1.set_xlabel('Original')
    ax1.set_ylabel('Reconstructed')
    ax1.set_title('Scatter Plot: Original vs Reconstructed')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 误差分布
    ax2 = axes[0, 1]
    errors = original - reconstructed
    ax2.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(errors.mean(), color='r', linestyle='--', label=f'Mean={errors.mean():.6f}')
    ax2.set_xlabel('Error (Original - Reconstructed)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 样本对比（前几个样本）
    ax3 = axes[1, 0]
    n_samples = min(5, len(original))
    indices = np.linspace(0, len(original)-1, n_samples, dtype=int)
    x = np.arange(n_samples)
    ax3.plot(x, original[indices], 'o-', label='Original', alpha=0.7)
    ax3.plot(x, reconstructed[indices], 's-', label='Reconstructed', alpha=0.7)
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Value')
    ax3.set_title('Sample Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 误差 vs 真实值
    ax4 = axes[1, 1]
    abs_errors = np.abs(errors)
    ax4.scatter(original, abs_errors, alpha=0.3, s=1)
    ax4.set_xlabel('Original Value')
    ax4.set_ylabel('Absolute Error')
    ax4.set_title('Absolute Error vs Original Value')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nDiagnostic plot saved to: {save_path}")
    plt.close()


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Diagnose evaluation metrics')
    parser.add_argument('--original', type=str, required=True, help='Path to original data (numpy array or CSV)')
    parser.add_argument('--reconstructed', type=str, required=True, help='Path to reconstructed data (numpy array or CSV)')
    parser.add_argument('--save_plot', type=str, default=None, help='Path to save diagnostic plot')
    
    args = parser.parse_args()
    
    # Load data
    if args.original.endswith('.npy'):
        original = np.load(args.original)
    else:
        original = np.loadtxt(args.original, delimiter=',')
    
    if args.reconstructed.endswith('.npy'):
        reconstructed = np.load(args.reconstructed)
    else:
        reconstructed = np.loadtxt(args.reconstructed, delimiter=',')
    
    # Run diagnosis
    metrics = diagnose_metrics(original, reconstructed)
    
    # Plot
    if args.save_plot:
        plot_diagnosis(original, reconstructed, args.save_plot)



