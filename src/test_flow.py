"""
测试Flow Matching模型在不同模态转换上的性能
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

from model_flow import ConditionalFlowMatching
from utils import (
    inverse_heatmap_to_spectrum,
    get_device,
    paired_dataset_use_random_split_by_default,
)
from train import PairedModalDataset, get_paired_loaders


def calculate_psnr(original, reconstructed):
    """计算PSNR (Peak Signal-to-Noise Ratio)"""
    original = original.flatten()
    reconstructed = reconstructed.flatten()
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_val = np.max(original)
    if max_val == 0:
        return np.nan
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr


def calculate_ssim_1d(original, reconstructed):
    """计算1D数据的SSIM (Structural Similarity Index)"""
    original = original.flatten()
    reconstructed = reconstructed.flatten()
    
    # 确保长度一致
    if len(original) != len(reconstructed):
        min_len = min(len(original), len(reconstructed))
        original = original[:min_len]
        reconstructed = reconstructed[:min_len]
    
    # 计算均值和方差
    mu1 = np.mean(original)
    mu2 = np.mean(reconstructed)
    sigma1_sq = np.var(original)
    sigma2_sq = np.var(reconstructed)
    sigma12 = np.mean((original - mu1) * (reconstructed - mu2))
    
    # SSIM参数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 计算SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    if denominator == 0:
        return np.nan
    
    ssim = numerator / denominator
    return ssim


def calculate_js_divergence(original, reconstructed):
    """计算Jensen-Shannon Divergence"""
    original = original.flatten()
    reconstructed = reconstructed.flatten()
    
    # 归一化为概率分布（确保非负）
    original_norm = original - np.min(original) + 1e-10
    reconstructed_norm = reconstructed - np.min(reconstructed) + 1e-10
    
    original_prob = original_norm / np.sum(original_norm)
    reconstructed_prob = reconstructed_norm / np.sum(reconstructed_norm)
    
    # 计算JS Divergence
    js_div = jensenshannon(original_prob, reconstructed_prob)
    return js_div


def calculate_metrics(original, reconstructed):
    """
    计算评估指标
    
    Returns:
        dict: 包含MSE, R2, Pearson相关系数等指标
    """
    original = original.flatten()
    reconstructed = reconstructed.flatten()
    
    # MSE
    mse = np.mean((original - reconstructed) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # R²
    r2 = r2_score(original, reconstructed)
    
    # Pearson相关系数
    try:
        pearson, _ = pearsonr(original, reconstructed)
    except:
        pearson = np.nan
    
    # MAE
    mae = np.mean(np.abs(original - reconstructed))
    
    # 相对误差百分比
    mape = np.mean(np.abs((original - reconstructed) / (original + 1e-8))) * 100
    
    # PSNR
    psnr = calculate_psnr(original, reconstructed)
    
    # SSIM
    ssim = calculate_ssim_1d(original, reconstructed)
    
    # JS Divergence
    js_div = calculate_js_divergence(original, reconstructed)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'pearson': pearson,
        'psnr': psnr,
        'ssim': ssim,
        'js_div': js_div
    }


def _resample_1d(y, new_len):
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) == new_len:
        return y
    x_old = np.linspace(0.0, 1.0, len(y))
    x_new = np.linspace(0.0, 1.0, new_len)
    return np.interp(x_new, x_old, y).astype(np.float64)


def dtw_distance_sakoe_chiba(a, b, window_ratio=0.12):
    """Euclidean DTW with Sakoe-Chiba band; a, b same length."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = len(a)
    if n == 0 or len(b) != n:
        return np.nan
    r = max(1, int(window_ratio * n))
    inf = 1e30
    dtw = np.full((n + 1, n + 1), inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        j_min = max(1, i - r)
        j_max = min(n, i + r)
        for j in range(j_min, j_max + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    if dtw[n, n] >= inf * 0.5:
        return np.nan
    return float(np.sqrt(dtw[n, n]))


def compute_dtw_metric(target, pred, subsample=200, window_ratio=0.12):
    """DTW on uniformly resampled sequences (length capped for speed)."""
    target = np.asarray(target, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    L = int(min(subsample, len(target), len(pred)))
    if L < 4:
        return np.nan
    a = _resample_1d(target, L)
    b = _resample_1d(pred, L)
    return dtw_distance_sakoe_chiba(a, b, window_ratio=window_ratio)


def peak_matching_metrics(target, pred, prominence_rel=0.05, distance=5, top_k=30):
    """
    find_peaks on target/pred (same relative prominence), then Hungarian match.
    peak_count_match: 1 if raw peak counts equal else 0 (before top_k truncation).
    """
    target = np.asarray(target, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    scale = float(np.max(target) - np.min(target) + 1e-8)

    prom = prominence_rel * scale
    peaks_gt, _ = find_peaks(target, prominence=prom, distance=distance)
    peaks_pr, _ = find_peaks(pred, prominence=prom, distance=distance)

    n_gt_raw = len(peaks_gt)
    n_pr_raw = len(peaks_pr)
    count_match = 1.0 if n_gt_raw == n_pr_raw else 0.0

    if n_gt_raw == 0 and n_pr_raw == 0:
        return {
            'peak_count_match': count_match,
            'peak_pos_mae': 0.0,
            'peak_height_mae': 0.0,
        }

    if n_gt_raw == 0 or n_pr_raw == 0:
        return {
            'peak_count_match': count_match,
            'peak_pos_mae': np.nan,
            'peak_height_mae': np.nan,
        }

    h_gt = target[peaks_gt]
    h_pr = pred[peaks_pr]
    if len(peaks_gt) > top_k:
        order = np.argsort(h_gt)[::-1][:top_k]
        peaks_gt = peaks_gt[order]
        h_gt = h_gt[order]
    if len(peaks_pr) > top_k:
        order = np.argsort(h_pr)[::-1][:top_k]
        peaks_pr = peaks_pr[order]
        h_pr = h_pr[order]

    n_gt = len(peaks_gt)
    n_pr = len(peaks_pr)
    len_idx = max(len(target) - 1, 1)
    cost = np.zeros((n_gt, n_pr), dtype=np.float64)
    for i in range(n_gt):
        for j in range(n_pr):
            cost[i, j] = (
                abs(float(peaks_gt[i]) - float(peaks_pr[j])) / len_idx
                + abs(float(h_gt[i]) - float(h_pr[j])) / scale
            )

    row_ind, col_ind = linear_sum_assignment(cost)
    pos_errs = [abs(float(peaks_gt[i]) - float(peaks_pr[j])) for i, j in zip(row_ind, col_ind)]
    h_errs = [abs(float(h_gt[i]) - float(h_pr[j])) for i, j in zip(row_ind, col_ind)]

    return {
        'peak_count_match': count_match,
        'peak_pos_mae': float(np.mean(pos_errs)) if pos_errs else np.nan,
        'peak_height_mae': float(np.mean(h_errs)) if h_errs else np.nan,
    }


def plot_flow_process(source_heatmaps, target_heatmaps, path_heatmaps_list, save_dir,
                     source_mode, target_mode, num_samples=6, num_steps_show=5, 
                     target_min_list=None, target_max_list=None, cmap='viridis'):
    """
    可视化Flow Matching的生成过程
    展示从源模态到目标模态的转换过程
    
    Args:
        source_heatmaps: 源热图列表 (list of (H, W) arrays) - 归一化后的[0,1]
        target_heatmaps: 目标热图列表 (list of (H, W) arrays) - 归一化后的[0,1]
        path_heatmaps_list: 每个样本的转换路径列表 (list of lists, each inner list contains (H, W) arrays)
        save_dir: 保存目录
        source_mode: 源模态
        target_mode: 目标模态
        num_samples: 展示的样本数量（行数）
        num_steps_show: 展示的步骤数（列数，包括起点和终点）
        target_min_list: 目标光谱最小值列表（用于denormalize）
        target_max_list: 目标光谱最大值列表（用于denormalize）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples = min(num_samples, len(source_heatmaps))
    
    # 创建6行5列的子图
    fig, axes = plt.subplots(num_samples, num_steps_show, figsize=(15, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        path = path_heatmaps_list[i]
        source_hm = source_heatmaps[i]
        target_hm = target_heatmaps[i]
        
        # 选择要展示的步骤（包括起点和终点）
        total_steps = len(path)
        step_indices = [0]  # 起点
        if total_steps > 1:
            # 中间步骤均匀分布
            for j in range(1, num_steps_show - 1):
                idx = int(j * (total_steps - 1) / (num_steps_show - 1))
                step_indices.append(min(idx, total_steps - 1))
        if total_steps > 1:
            step_indices.append(total_steps - 1)  # 终点
        
        # 确保正好5个步骤
        step_indices = step_indices[:num_steps_show]
        while len(step_indices) < num_steps_show and total_steps > 1:
            step_indices.append(total_steps - 1)
        
        # Denormalize热图（如果提供了min/max值）
        if target_min_list is not None and target_max_list is not None:
            spec_min = target_min_list[i]
            spec_max = target_max_list[i]
            denormalize_fn = lambda hm: hm * (spec_max - spec_min + 1e-8) + spec_min
        else:
            denormalize_fn = lambda hm: hm  # 不denormalize
        
        # 计算该样本所有热图的范围用于统一显示
        all_heatmaps = [denormalize_fn(source_hm)] + [denormalize_fn(target_hm)]
        if len(path) > 0:
            all_heatmaps.extend([denormalize_fn(p) for p in path])
        vmin = min([hm.min() for hm in all_heatmaps])
        vmax = max([hm.max() for hm in all_heatmaps])
        
        # 绘制每个步骤
        for j, step_idx in enumerate(step_indices):
            if j == 0:
                # 第一列：源光谱（IR）
                heatmap = denormalize_fn(source_hm)
                title = f'{source_mode.upper()}\n(t=0)'
            elif j == num_steps_show - 1:
                # 最后一列：目标光谱（Raman）
                heatmap = denormalize_fn(target_hm)
                title = f'{target_mode.upper()}\n(t=1.0)'
            else:
                # 中间步骤：转换过程
                if step_idx < len(path):
                    heatmap = denormalize_fn(path[step_idx])
                    t_val = step_idx / (total_steps - 1) if total_steps > 1 else 0.0
                    title = f't={t_val:.2f}'
                else:
                    heatmap = denormalize_fn(target_hm)
                    title = f't=1.0'
            
            ax = axes[i, j]
            # 显示60x60热图（patch），使用'imshow'直接显示2D图像（denormalize后的）
            im = ax.imshow(heatmap, cmap=cmap, aspect='equal', interpolation='nearest', 
                          vmin=vmin, vmax=vmax)  # 使用denormalize后的范围
            ax.set_title(title, fontsize=10, fontweight='bold' if j in [0, num_steps_show-1] else 'normal')
            ax.axis('off')
            ax.set_aspect('equal')  # 保持60x60的方形比例
            
            # 只在第一行最后一列添加颜色条
            if i == 0 and j == num_steps_show - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')
        
        # 在左侧添加样本编号
        axes[i, 0].text(-0.15, 0.5, f'Sample {i+1}', 
                       transform=axes[i, 0].transAxes, 
                       fontsize=12, fontweight='bold',
                       va='center', ha='right')
    
    plt.suptitle(f'Flow Matching Process: {source_mode.upper()} → {target_mode.upper()}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = os.path.join(save_dir, f'flow_{source_mode}2{target_mode}_process.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved flow process visualization to: {save_path}")


def plot_source_xt_pair(source_heatmap, xt_heatmap, save_dir, source_mode, target_mode, t_value, cmap='viridis'):
    """
    绘制单个样本的 source 与 x_t 热图（1x2）
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # 使用统一色彩范围，便于直接比较
    vmin = min(source_heatmap.min(), xt_heatmap.min())
    vmax = max(source_heatmap.max(), xt_heatmap.max())

    im0 = axes[0].imshow(
        source_heatmap, cmap=cmap, aspect='equal', interpolation='nearest',
        vmin=vmin, vmax=vmax
    )
    axes[0].set_title(f'Source {source_mode.upper()} (t=0)', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    im1 = axes[1].imshow(
        xt_heatmap, cmap=cmap, aspect='equal', interpolation='nearest',
        vmin=vmin, vmax=vmax
    )
    axes[1].set_title(f'x_t ({target_mode.upper()} flow, t={t_value:.2f})', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Normalized intensity')
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'flow_{source_mode}2{target_mode}_source_xt_t{t_value:.2f}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved source + x_t visualization to: {save_path}")


def plot_comparison(original_list, reconstructed_list, save_dir, prefix, 
                   source_mode, target_mode, num_samples=5):
    """
    绘制原始和重建光谱的对比图
    
    Args:
        original_list: 原始光谱列表
        reconstructed_list: 重建光谱列表
        save_dir: 保存目录
        prefix: 文件名前缀
        source_mode: 源模态
        target_mode: 目标模态
        num_samples: 绘制的样本数量
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples = min(num_samples, len(original_list))
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        orig = original_list[i]
        recon = reconstructed_list[i]
        
        x_axis = np.linspace(0, len(orig) - 1, len(orig))
        axes[i].plot(x_axis, orig, label='Original', alpha=0.7, linewidth=2)
        axes[i].plot(x_axis, recon, label='Generated (Flow Matching)', alpha=0.7, 
                    linewidth=2, linestyle='--')
        axes[i].set_xlabel('Wavenumber Index')
        axes[i].set_ylabel('Intensity')
        axes[i].set_title(f'Sample {i+1}: {source_mode} -> {target_mode}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{prefix}_{source_mode}2{target_mode}_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to: {save_path}")


def test_modality_pair(model, test_loader, source_mode, target_mode, device, 
                      save_dir='results', num_steps=100, use_rk4=True, dump_index=None,
                      extra_metrics=True,
                      dtw_subsample=200,
                      dtw_window_ratio=0.12,
                      peak_prominence_rel=0.05,
                      peak_distance=5,
                      peak_top_k=30):
    """
    测试单个模态对的转换性能
    
    Args:
        model: 训练好的Flow Matching模型
        test_loader: 测试数据加载器
        source_mode: 源模态
        target_mode: 目标模态
        device: 计算设备
        save_dir: 结果保存目录
        num_steps: ODE求解步数
        use_rk4: 是否使用RK4方法（更准确但更慢）
        extra_metrics: 是否计算 DTW 与峰相关指标（更慢）
        dtw_subsample: DTW 前将谱统一重采样到该长度（上限）
        dtw_window_ratio: Sakoe-Chiba 带宽比例
        peak_prominence_rel: find_peaks prominence = rel * (max-min)
        peak_distance: find_peaks 最小峰间距（索引）
        peak_top_k: 匹配时每侧最多保留的峰数（避免过多小峰）
    """
    model.eval()
    
    mode_map = {'ir': 0, 'uv': 1, 'raman': 2}
    target_mode_idx = mode_map[target_mode]
    
    all_metrics = {
        'mse': [], 'rmse': [], 'mae': [], 'mape': [],
        'r2': [], 'pearson': [], 'psnr': [], 'ssim': [], 'js_div': []
    }
    if extra_metrics:
        all_metrics['dtw'] = []
        all_metrics['peak_count_match'] = []
        all_metrics['peak_pos_mae'] = []
        all_metrics['peak_height_mae'] = []

    def has_invalid_core_metrics(metrics):
        core_metric_keys = (
            'mse', 'rmse', 'mae', 'mape',
            'r2', 'pearson', 'psnr', 'ssim', 'js_div',
        )
        for key in core_metric_keys:
            value = metrics.get(key)
            if value is None or np.isnan(value):
                return True
        return False
    
    original_spectra = []
    generated_spectra = []
    all_targets = []
    all_preds = []
    all_sources = []
    sample_indices = []
    total_samples = 0
    skipped_invalid_samples = 0
    
    # 用于Flow过程可视化的数据
    source_heatmaps_for_flow = []
    target_heatmaps_for_flow = []
    path_heatmaps_list = []
    target_min_list_for_flow = []
    target_max_list_for_flow = []
    
    # 使用tqdm显示测试进度
    pbar = tqdm(test_loader, desc=f"Testing {source_mode}->{target_mode}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            source_batch, target_batch = batch[0], batch[1]
            source_min_batch = batch[2]
            source_max_batch = batch[3]
            target_min_batch = batch[4]
            target_max_batch = batch[5]
            
            source_batch = source_batch.to(device)
            
            # 收集前6个样本用于Flow过程可视化
            collect_flow_data = len(source_heatmaps_for_flow) < 6
            
            if collect_flow_data:
                # 使用Flow Matching生成，并获取完整路径
                generated, path = model.sample(
                    source_batch[:6-len(source_heatmaps_for_flow)] if len(source_heatmaps_for_flow) < 6 else source_batch,
                    target_mode=target_mode_idx,
                    num_steps=num_steps,
                    use_rk4=use_rk4,
                    return_path=True
                )
                
                # 处理路径数据
                batch_size_collect = min(6 - len(source_heatmaps_for_flow), source_batch.size(0))
                path_np = [p.cpu().numpy() for p in path]
                
                for i in range(batch_size_collect):
                    source_hm = source_batch[i, 0].cpu().numpy()  # (H, W)
                    target_hm = target_batch[i, 0].numpy()  # (H, W)
                    path_hms = [p[i, 0] for p in path_np]  # List of (H, W)
                    spec_min = float(target_min_batch[i])
                    spec_max = float(target_max_batch[i])
                    
                    source_heatmaps_for_flow.append(source_hm)
                    target_heatmaps_for_flow.append(target_hm)
                    path_heatmaps_list.append(path_hms)
                    target_min_list_for_flow.append(spec_min)
                    target_max_list_for_flow.append(spec_max)
                
                # 继续处理完整batch用于指标计算
                generated = model.sample(
                    source_batch,
                    target_mode=target_mode_idx,
                    num_steps=num_steps,
                    use_rk4=use_rk4,
                    return_path=False
                )
            else:
                # 使用Flow Matching生成（不返回路径）
                generated = model.sample(
                    source_batch,
                    target_mode=target_mode_idx,
                    num_steps=num_steps,
                    use_rk4=use_rk4,
                    return_path=False
                )
            
            # 转换为numpy
            gen_np = generated.cpu().numpy()
            target_np = target_batch.numpy()
            
            # 将热图转换回光谱
            base_idx = total_samples
            for i in range(gen_np.shape[0]):
                total_samples += 1
                gen_heatmap = gen_np[i, 0]  # (H, W)
                target_heatmap = target_np[i, 0]
                source_heatmap_np = source_batch[i, 0].cpu().numpy()
                
                spec_min = float(target_min_batch[i])
                spec_max = float(target_max_batch[i])
                src_min = float(source_min_batch[i])
                src_max = float(source_max_batch[i])
                
                # 转换回1D光谱
                gen_spec = inverse_heatmap_to_spectrum(gen_heatmap, spec_min, spec_max)
                target_spec = inverse_heatmap_to_spectrum(target_heatmap, spec_min, spec_max)
                source_spec = inverse_heatmap_to_spectrum(source_heatmap_np, src_min, src_max)
                
                # 计算指标
                metrics = calculate_metrics(target_spec, gen_spec)
                if extra_metrics:
                    metrics['dtw'] = compute_dtw_metric(
                        target_spec, gen_spec,
                        subsample=dtw_subsample,
                        window_ratio=dtw_window_ratio,
                    )
                    metrics.update(
                        peak_matching_metrics(
                            target_spec, gen_spec,
                            prominence_rel=peak_prominence_rel,
                            distance=peak_distance,
                            top_k=peak_top_k,
                        )
                    )
                if has_invalid_core_metrics(metrics):
                    skipped_invalid_samples += 1
                    continue
                for key in all_metrics:
                    if key not in metrics:
                        continue
                    v = metrics[key]
                    if isinstance(v, (float, np.floating)) and np.isnan(v):
                        continue
                    all_metrics[key].append(v)
                all_targets.append(target_spec)
                all_preds.append(gen_spec)
                all_sources.append(source_spec)
                sample_indices.append(base_idx + i)
                
                # 保存前几个样本用于可视化
                if len(original_spectra) < 5:
                    original_spectra.append(target_spec)
                    generated_spectra.append(gen_spec)
            
            # 更新进度条，显示当前R²分数（如果有的话）
            if len(all_metrics['r2']) > 0:
                current_r2 = np.mean(all_metrics['r2'])
                pbar.set_postfix({
                    'R²': f"{current_r2:.4f}",
                    'valid': len(all_metrics['r2']),
                    'skipped': skipped_invalid_samples,
                })
            else:
                pbar.set_postfix({
                    'valid': len(all_metrics['mse']),
                    'skipped': skipped_invalid_samples,
                })
    
    # 打印平均指标
    print(f"\n{'='*60}")
    print(f"Test Results (Flow Matching): {source_mode} -> {target_mode}")
    print(f"{'='*60}")
    print(f"Valid samples: {len(all_metrics['mse'])}/{total_samples} "
          f"(skipped invalid core-metric samples: {skipped_invalid_samples})")
    if len(all_metrics['mse']) == 0:
        print("Error: no valid samples remain after filtering invalid metrics.")
        return all_metrics
    print(f"MSE:      {np.mean(all_metrics['mse']):.6e} ± {np.std(all_metrics['mse']):.6e}")
    print(f"RMSE:     {np.mean(all_metrics['rmse']):.6e} ± {np.std(all_metrics['rmse']):.6e}")
    print(f"MAE:      {np.mean(all_metrics['mae']):.6e} ± {np.std(all_metrics['mae']):.6e}")
    print(f"MAPE:     {np.mean(all_metrics['mape']):.4f}% ± {np.std(all_metrics['mape']):.4f}%")
    print(f"R²:       {np.mean(all_metrics['r2']):.6f} ± {np.std(all_metrics['r2']):.6f}")
    print(f"Pearson:  {np.mean(all_metrics['pearson']):.6f} ± {np.std(all_metrics['pearson']):.6f}")
    if len(all_metrics['psnr']) > 0:
        print(f"PSNR:     {np.mean(all_metrics['psnr']):.6f} ± {np.std(all_metrics['psnr']):.6f}")
    if len(all_metrics['ssim']) > 0:
        print(f"SSIM:     {np.mean(all_metrics['ssim']):.6f} ± {np.std(all_metrics['ssim']):.6f}")
    if len(all_metrics['js_div']) > 0:
        print(f"JS Div:   {np.mean(all_metrics['js_div']):.6f} ± {np.std(all_metrics['js_div']):.6f}")
    if extra_metrics and all_metrics.get('dtw'):
        dtw_vals = np.asarray(all_metrics['dtw'], dtype=np.float64)
        print(f"DTW:      {np.nanmean(dtw_vals):.6f} ± {np.nanstd(dtw_vals):.6f}  "
              f"(subsample<={dtw_subsample}, Sakoe-Chiba w={dtw_window_ratio:.3f})")
    if extra_metrics and all_metrics.get('peak_count_match'):
        pcm = np.asarray(all_metrics['peak_count_match'])
        print(f"Peak cnt acc: {np.mean(pcm):.4f}  "
              f"(1 if raw #peaks equal on target vs pred; prominence_rel={peak_prominence_rel}, dist={peak_distance})")
    if extra_metrics and all_metrics.get('peak_pos_mae'):
        pp = np.asarray(all_metrics['peak_pos_mae'], dtype=np.float64)
        print(f"Peak pos MAE (matched): {np.nanmean(pp):.4f} ± {np.nanstd(pp):.4f}  (index units, top_k={peak_top_k})")
    if extra_metrics and all_metrics.get('peak_height_mae'):
        ph = np.asarray(all_metrics['peak_height_mae'], dtype=np.float64)
        print(f"Peak hgt MAE (matched): {np.nanmean(ph):.6e} ± {np.nanstd(ph):.6e}")
    
    # 绘制对比图
    if len(original_spectra) > 0:
        plot_comparison(
            original_spectra, generated_spectra,
            save_dir, 'flow',
            source_mode, target_mode,
            num_samples=min(5, len(original_spectra))
        )
    
    # 保存全量预测/目标/输入，及指定索引的对照
    if sample_indices and all_targets and all_preds and all_sources:
        os.makedirs(save_dir, exist_ok=True)
        preds_arr = np.vstack(all_preds)
        targets_arr = np.vstack(all_targets)
        sources_arr = np.vstack(all_sources)
        idx_arr = np.array(sample_indices, dtype=int)

        def _save_matrix(path, idx, matrix):
            with open(path, "w", encoding="utf-8") as f:
                header = "index," + ",".join([f"v{i}" for i in range(matrix.shape[1])])
                f.write(header + "\n")
                for i_row, row in zip(idx, matrix):
                    f.write(f"{i_row}," + ",".join(f"{v:.6e}" for v in row) + "\n")

        preds_path = Path(save_dir) / f"flow_{source_mode}2{target_mode}_preds.csv"
        targets_path = Path(save_dir) / f"flow_{source_mode}2{target_mode}_targets.csv"
        sources_path = Path(save_dir) / f"flow_{source_mode}2{target_mode}_sources.csv"
        _save_matrix(preds_path, idx_arr, preds_arr)
        _save_matrix(targets_path, idx_arr, targets_arr)
        _save_matrix(sources_path, idx_arr, sources_arr)
        print(f"\nSaved predictions to: {preds_path}")
        print(f"Saved targets to:      {targets_path}")
        print(f"Saved sources to:      {sources_path}")

        # 保存每个样本的R2列表（顺序与index一致）
        r2_list = np.array(all_metrics['r2'], dtype=float)
        r2_path = Path(save_dir) / f"flow_{source_mode}2{target_mode}_r2_per_sample.csv"
        with open(r2_path, "w", encoding="utf-8") as f:
            f.write("index,r2\n")
            for idx_val, r2_val in zip(idx_arr, r2_list):
                if not np.isnan(r2_val):
                    f.write(f"{idx_val},{r2_val:.6f}\n")
        print(f"Saved per-sample R2 to: {r2_path}")

        # 保存per-sample PSNR, SSIM, JS Divergence
        psnr_list = np.array(all_metrics['psnr'], dtype=float)
        ssim_list = np.array(all_metrics['ssim'], dtype=float)
        js_div_list = np.array(all_metrics['js_div'], dtype=float)
        metrics_path = Path(save_dir) / f"flow_{source_mode}2{target_mode}_metrics_per_sample.csv"
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("index,psnr,ssim,js_div\n")
            for idx_val, psnr_val, ssim_val, js_val in zip(idx_arr, psnr_list, ssim_list, js_div_list):
                f.write(f"{idx_val},{psnr_val:.6f},{ssim_val:.6f},{js_val:.6f}\n")
        print(f"Saved per-sample metrics (PSNR, SSIM, JS Div) to: {metrics_path}")

        if dump_index:
            for want_idx in dump_index:
                if want_idx in idx_arr:
                    pos = np.where(idx_arr == want_idx)[0][0]
                    tgt_row = targets_arr[pos]
                    pred_row = preds_arr[pos]
                    src_row = sources_arr[pos]
                    out_path = Path(save_dir) / f"flow_{source_mode}2{target_mode}_idx{want_idx}.csv"
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write("target,pred,source\n")
                        for tv, pv, sv in zip(tgt_row, pred_row, src_row):
                            f.write(f"{tv:.6e},{pv:.6e},{sv:.6e}\n")
                    print(f"Dumped target/pred for index {want_idx} -> {out_path}")
                else:
                    print(f"Warning: requested index {want_idx} not found in this test run.")
    
    # 绘制Flow Matching过程可视化
    if len(source_heatmaps_for_flow) > 0:
        plot_flow_process(
            source_heatmaps_for_flow,
            target_heatmaps_for_flow,
            path_heatmaps_list,
            save_dir,
            source_mode,
            target_mode,
            num_samples=min(6, len(source_heatmaps_for_flow)),
            num_steps_show=5,
            target_min_list=target_min_list_for_flow if len(target_min_list_for_flow) > 0 else None,
            target_max_list=target_max_list_for_flow if len(target_max_list_for_flow) > 0 else None
        )
    
    return all_metrics


def generate_process_only(model, test_loader, source_mode, target_mode, device, 
                          save_dir='results', num_steps=150, use_rk4=True, cmap='viridis'):
    """
    快速生成Flow Matching过程可视化（只处理6个样本，不运行完整测试）
    """
    model.eval()
    
    mode_map = {'ir': 0, 'uv': 1, 'raman': 2}
    target_mode_idx = mode_map[target_mode]
    
    source_heatmaps_for_flow = []
    target_heatmaps_for_flow = []
    path_heatmaps_list = []
    target_min_list_for_flow = []
    target_max_list_for_flow = []
    
    print(f"Generating flow process visualization: {source_mode} -> {target_mode}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(source_heatmaps_for_flow) >= 6:
                break
                
            source_batch, target_batch = batch[0], batch[1]
            target_min_batch = batch[4]
            target_max_batch = batch[5]
            
            source_batch = source_batch.to(device)
            
            # 只处理需要的样本数
            batch_size_collect = min(6 - len(source_heatmaps_for_flow), source_batch.size(0))
            source_batch_collect = source_batch[:batch_size_collect]
            
            # 使用Flow Matching生成，并获取完整路径
            generated, path = model.sample(
                source_batch_collect,
                target_mode=target_mode_idx,
                num_steps=num_steps,
                use_rk4=use_rk4,
                return_path=True
            )
            
            # 处理路径数据
            path_np = [p.cpu().numpy() for p in path]
            
            for i in range(batch_size_collect):
                source_hm = source_batch_collect[i, 0].cpu().numpy()  # (H, W)
                target_hm = target_batch[i, 0].numpy()  # (H, W)
                path_hms = [p[i, 0] for p in path_np]  # List of (H, W)
                spec_min = float(target_min_batch[i])
                spec_max = float(target_max_batch[i])
                
                source_heatmaps_for_flow.append(source_hm)
                target_heatmaps_for_flow.append(target_hm)
                path_heatmaps_list.append(path_hms)
                target_min_list_for_flow.append(spec_min)
                target_max_list_for_flow.append(spec_max)
    
    # 绘制Flow Matching过程可视化
    if len(source_heatmaps_for_flow) > 0:
        plot_flow_process(
            source_heatmaps_for_flow,
            target_heatmaps_for_flow,
            path_heatmaps_list,
            save_dir,
            source_mode,
            target_mode,
            num_samples=min(6, len(source_heatmaps_for_flow)),
            num_steps_show=5,
            target_min_list=target_min_list_for_flow,
            target_max_list=target_max_list_for_flow,
            cmap=cmap
        )
        print(f"Process visualization generated successfully!")
    else:
        print("Error: No samples collected for visualization")


def generate_single_xt_example(model, test_loader, source_mode, target_mode, device,
                               save_dir='results', num_steps=150, use_rk4=True, xt_t=0.5, cmap='viridis'):
    """
    仅生成单个样本的 source 与指定时刻 x_t 热图（1x2）
    """
    model.eval()

    mode_map = {'ir': 0, 'uv': 1, 'raman': 2}
    target_mode_idx = mode_map[target_mode]

    # 限制到合法时间范围
    xt_t = max(0.0, min(1.0, float(xt_t)))
    print(f"Generating single source + x_t visualization: {source_mode} -> {target_mode}, t={xt_t:.2f}")

    with torch.no_grad():
        for batch in test_loader:
            source_batch = batch[0].to(device)

            # 只取第一个样本
            source_sample = source_batch[:1]

            # 获取完整路径
            _, path = model.sample(
                source_sample,
                target_mode=target_mode_idx,
                num_steps=num_steps,
                use_rk4=use_rk4,
                return_path=True
            )

            if len(path) == 0:
                print("Error: Empty path returned by model.sample")
                return

            source_hm = source_sample[0, 0].cpu().numpy()
            path_np = [p[0, 0].cpu().numpy() for p in path]

            step_idx = int(round(xt_t * (len(path_np) - 1)))
            step_idx = max(0, min(step_idx, len(path_np) - 1))
            xt_hm = path_np[step_idx]
            t_actual = step_idx / (len(path_np) - 1) if len(path_np) > 1 else 0.0

            plot_source_xt_pair(
                source_hm, xt_hm, save_dir,
                source_mode, target_mode, t_actual, cmap=cmap
            )
            print("Single source + x_t visualization generated successfully!")
            return

    print("Error: No samples found in test_loader")


def main():
    parser = argparse.ArgumentParser(description='Test Flow Matching Model')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing processed CSV files')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--source_mode', type=str, required=True,
                       choices=['ir', 'uv', 'raman'],
                       help='Source modality')
    parser.add_argument('--target_mode', type=str, required=True,
                       choices=['ir', 'uv', 'raman'],
                       help='Target modality')
    parser.add_argument('--source_csv', type=str, default=None,
                       help='Custom source CSV filename (optional, overrides default naming)')
    parser.add_argument('--target_csv', type=str, default=None,
                       help='Custom target CSV filename (optional, overrides default naming)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--backbone', type=str, default='unet', choices=['unet', 'dit', 'vibradit'],
                       help='Backbone: unet (default), dit (2D DiT), or vibradit (1D spectral DiT). Must match training.')
    parser.add_argument('--dit_hidden_dim', type=int, default=256, help='[DiT only] Transformer hidden dim')
    parser.add_argument('--dit_depth', type=int, default=6, help='[DiT only] Number of DiT blocks')
    parser.add_argument('--dit_num_heads', type=int, default=4, help='[DiT only] Number of attention heads')
    parser.add_argument('--dit_patch_size', type=int, default=4, help='[DiT only] Patch size')
    parser.add_argument('--heatmap_size', type=int, default=3600,
                       help='Heatmap size (must be a perfect square, e.g., 3600=60x60, 1024=32x32)')
    parser.add_argument('--resize_shape', type=int, nargs=2, default=[60, 60],
                       help='Heatmap reshape size, e.g., 60 60 or 32 32')
    parser.add_argument('--source_size', type=int, default=None,
                       help='Optional source spectrum length before heatmap (if None, uses heatmap_size)')
    parser.add_argument('--target_size', type=int, default=None,
                       help='Optional target spectrum length before heatmap (if None, uses heatmap_size)')
    parser.add_argument('--no_split', action='store_true',
                       help='Do not split dataset; treat the whole provided CSV/H5 as the test set. '
                            'Note: For QM9S / QMe14S, the same random_split test subset as training is used unless this flag is set. '
                            'For other datasets, if --source_csv/--target_csv is provided, full dataset is used by default.')
    parser.add_argument('--use_split_test', action='store_true',
                       help='Force random_split and evaluate only the test subset, even when --source_csv/--target_csv is provided.')
    parser.add_argument('--save_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--num_steps', type=int, default=150, help='Number of ODE steps (default: 150, more steps = better quality)')
    parser.add_argument('--use_rk4', action='store_true', help='Use RK4 ODE solver (more accurate but slower, default: True)')
    parser.add_argument('--no_rk4', action='store_true', help='Disable RK4 and use Euler method (faster but less accurate)')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--process_only', action='store_true',
                       help='Only generate process visualization (fast, no full test)')
    parser.add_argument('--xt_only', action='store_true',
                       help='Only generate a single 1x2 heatmap figure: source and selected x_t')
    parser.add_argument('--xt_t', type=float, default=0.5,
                       help='The t value for x_t visualization in [0,1] (default: 0.5)')
    parser.add_argument('--cmap', type=str, default='viridis',
                       help='Matplotlib colormap for heatmaps (e.g., coolwarm, RdBu_r)')
    parser.add_argument('--dump_index', type=int, nargs='+', default=None,
                       help='List of sample indices (within this test run) to dump target/pred spectra to CSV')
    parser.add_argument('--no_extra_metrics', action='store_true',
                       help='Skip DTW and peak-based metrics (faster full test)')
    parser.add_argument('--dtw_subsample', type=int, default=200,
                       help='Resample length cap before DTW (default: 200)')
    parser.add_argument('--dtw_window_ratio', type=float, default=0.12,
                       help='Sakoe-Chiba band width as fraction of length (default: 0.12)')
    parser.add_argument('--peak_prominence_rel', type=float, default=0.05,
                       help='find_peaks prominence = rel * (max-min) of target (default: 0.05)')
    parser.add_argument('--peak_distance', type=int, default=5,
                       help='find_peaks minimum distance between peaks in index (default: 5)')
    parser.add_argument('--peak_top_k', type=int, default=30,
                       help='Max peaks per side for Hungarian match (default: 30)')
    args = parser.parse_args()
    
    # 设置随机种子以确保结果可复现
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
    
    # 创建模型（支持 unet / dit backbone）
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
    
    # 加载检查点：
    # - unet: 兼容旧命名（无 backbone 后缀 / 无 seed）
    # - dit : 必须加载 dit 对应 checkpoint，禁止回退到 unet checkpoint
    backbone_tag = f'_{args.backbone}' if args.backbone != 'unet' else ''
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f'flow_{args.source_mode}2{args.target_mode}{backbone_tag}_best_seed{args.seed}.pt'
    )
    
    # 仅对 unet 保持旧 checkpoint 命名兼容
    if args.backbone == 'unet' and not os.path.exists(checkpoint_path):
        checkpoint_path_old = os.path.join(
            args.checkpoint_dir,
            f'flow_{args.source_mode}2{args.target_mode}_best.pt'
        )
        if os.path.exists(checkpoint_path_old):
            checkpoint_path = checkpoint_path_old
            print(f"Warning: Using old checkpoint format (without seed). Consider retraining with --seed {args.seed}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)
    if missing_keys or unexpected_keys:
        print("Error: Checkpoint/model mismatch detected.")
        if missing_keys:
            print("Missing keys:")
            for key in missing_keys[:20]:
                print(f"  - {key}")
            if len(missing_keys) > 20:
                print(f"  ... and {len(missing_keys) - 20} more")
        if unexpected_keys:
            print("Unexpected keys:")
            for key in unexpected_keys[:20]:
                print(f"  - {key}")
            if len(unexpected_keys) > 20:
                print(f"  ... and {len(unexpected_keys) - 20} more")
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} does not match backbone='{args.backbone}'. "
            "Please load the correct checkpoint."
        )
    print(f"Loaded checkpoint from: {checkpoint_path}")
    
    # 加载测试数据
    data_dir = Path(args.data_dir)
    # Use custom filenames if provided, otherwise use default naming
    if args.source_csv:
        source_csv = Path(args.source_csv) if os.path.isabs(args.source_csv) else data_dir / args.source_csv
    else:
        source_csv = data_dir / f'{args.source_mode}_broaden_processed.csv'
    
    if args.target_csv:
        target_csv = Path(args.target_csv) if os.path.isabs(args.target_csv) else data_dir / args.target_csv
    else:
        target_csv = data_dir / f'{args.target_mode}_broaden_processed.csv'
    
    if not source_csv.exists() or not target_csv.exists():
        print(f"Error: Data files not found")
        print(f"Source: {source_csv}")
        print(f"Target: {target_csv}")
        return
    
    # 创建测试数据加载器
    # QM9S / QMe14S：与训练一致，默认 random_split 的 test 子集（即使显式给了默认 CSV 名）
    # 其他数据：若显式提供 source/target CSV，则默认全量作 test，除非 --no_split 未提供… 见下
    if args.use_split_test and args.no_split:
        raise ValueError("--use_split_test and --no_split cannot be used together.")

    if args.use_split_test:
        use_full_as_test = False
    elif paired_dataset_use_random_split_by_default(
        args.data_dir, args.source_csv, args.target_csv
    ):
        use_full_as_test = args.no_split
    else:
        use_full_as_test = args.no_split or (args.source_csv is not None) or (args.target_csv is not None)
    
    if use_full_as_test:
        # When user provides explicit CSVs (often already "test split"), do NOT random-split again.
        dataset = PairedModalDataset(
            str(source_csv), str(target_csv),
            source_size=args.source_size,
            target_size=args.target_size,
            heatmap_size=args.heatmap_size,
            resize_shape=tuple(args.resize_shape),
            out_channels=1,
            use_h5=True
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
            seed=args.seed
        )
        print("Data split: using random_split(test subset) to match training split.")
    
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # 确定是否使用RK4（默认True，使用RK4获得更准确的结果）
    # 如果指定了--no_rk4，则使用Euler；否则默认使用RK4（也可以通过--use_rk4显式指定）
    use_rk4 = not args.no_rk4  # 默认True，除非指定--no_rk4
    
    # 如果只生成source+x_t图，跳过完整测试
    if args.xt_only:
        generate_single_xt_example(
            model, test_loader,
            args.source_mode, args.target_mode,
            device, args.save_dir, args.num_steps, use_rk4, args.xt_t, args.cmap
        )
        print("\nSingle x_t visualization completed!")
    # 如果只生成process图，跳过完整测试
    elif args.process_only:
        generate_process_only(
            model, test_loader,
            args.source_mode, args.target_mode,
            device, args.save_dir, args.num_steps, use_rk4, args.cmap
        )
        print("\nProcess visualization completed!")
    else:
        # 完整测试
        metrics = test_modality_pair(
            model, test_loader,
            args.source_mode, args.target_mode,
            device, args.save_dir, args.num_steps, use_rk4,
            dump_index=args.dump_index,
            extra_metrics=not args.no_extra_metrics,
            dtw_subsample=args.dtw_subsample,
            dtw_window_ratio=args.dtw_window_ratio,
            peak_prominence_rel=args.peak_prominence_rel,
            peak_distance=args.peak_distance,
            peak_top_k=args.peak_top_k,
        )
        print("\nTesting completed!")


if __name__ == '__main__':
    main()

