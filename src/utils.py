"""
工具函数模块：数据处理和训练工具函数
"""
import torch
import torch.nn.functional as F
import numpy as np
import math
from scipy.signal import find_peaks


def compute_params(spectrum, x_axis=None):
    """
    计算物理先验参数（光谱展宽分布参数）
    参考: https://github.com/ymzhu19eee/Raman-generation/blob/main/train_paras.py
    
    Args:
        spectrum: 光谱数据 (3600,) 或 (B, 3600)
        x_axis: x轴数据（波数/波长），可选
    
    Returns:
        params: 包含物理先验参数字典
            - mean: 均值
            - std: 标准差
            - bandwidth: 展宽带宽
            - peak_positions: 峰值位置
            - peak_intensities: 峰值强度
    """
    spectrum = np.asarray(spectrum)
    
    # 如果是批量数据，处理第一个样本
    if spectrum.ndim > 1:
        spectrum = spectrum[0] if spectrum.shape[0] > 0 else spectrum.flatten()
    
    # 计算基本统计量
    mean_val = np.mean(spectrum)
    std_val = np.std(spectrum)
    
    # 计算展宽带宽（基于半峰全宽FWHM）
    max_intensity = np.max(spectrum)
    half_max = max_intensity / 2.0
    
    # 找到峰值位置
    peak_idx = np.argmax(spectrum)
    
    # 计算半峰全宽
    indices_above_half = np.where(spectrum >= half_max)[0]
    if len(indices_above_half) > 0:
        fwhm = indices_above_half[-1] - indices_above_half[0]
        bandwidth = fwhm / len(spectrum) if len(spectrum) > 0 else 0.1
    else:
        bandwidth = 0.1  # 默认值
    
    # 找到所有峰值（局部最大值）
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(spectrum, height=np.percentile(spectrum, 50))
    peak_positions = peaks.tolist() if len(peaks) > 0 else [peak_idx]
    peak_intensities = spectrum[peaks].tolist() if len(peaks) > 0 else [max_intensity]
    
    # 限制峰值数量
    if len(peak_positions) > 20:
        # 选择强度最高的20个峰值
        peak_indices_sorted = np.argsort(peak_intensities)[::-1][:20]
        peak_positions = [peak_positions[i] for i in peak_indices_sorted]
        peak_intensities = [peak_intensities[i] for i in peak_indices_sorted]
    
    params = {
        'mean': mean_val,
        'std': std_val,
        'bandwidth': bandwidth,
        'peak_positions': peak_positions,
        'peak_intensities': peak_intensities,
        'max_intensity': max_intensity,
        'energy_range': (np.min(spectrum), np.max(spectrum))
    }
    
    return params


def compute_prior_distribution(source_params, target_params):
    """
    计算物理先验分布参数（用于KL散度损失）
    基于源光谱和目标光谱的物理参数
    
    注意：这里的prior_mu和prior_logvar是**先验分布p(z)的参数**，
    不是VAE编码器输出的mu和logvar（那些是后验分布q(z|x)的参数）。
    
    标准VAE: p(z) = N(0, 1)  (prior_mu=0, prior_logvar=0)
    物理先验VAE: p(z) = N(prior_mu, prior_logvar)  (基于物理参数计算)
    
    Args:
        source_params: 源光谱参数（从compute_params得到）
        target_params: 目标光谱参数（从compute_params得到）
    
    Returns:
        prior_mu: 先验分布p(z)的均值参数
        prior_logvar: 先验分布p(z)的对数方差参数
    """
    # 基于展宽参数计算先验
    source_bandwidth = source_params['bandwidth']
    target_bandwidth = target_params['bandwidth']
    
    # 先验均值：考虑带宽差异
    prior_mu = np.log(target_bandwidth / (source_bandwidth + 1e-8) + 1e-8)
    
    # 先验方差：基于强度分布的方差
    prior_std = np.sqrt(source_params['std'] ** 2 + target_params['std'] ** 2)
    prior_logvar = np.log(prior_std ** 2 + 1e-8)
    
    return prior_mu, prior_logvar


def compute_kl_loss_with_prior(mu, logvar, prior_mu, prior_logvar):
    """
    计算带物理先验的KL散度损失
    参考: https://github.com/ymzhu19eee/Raman-generation/blob/main/train_paras.py
    
    计算 KL(q(z|x) || p(z))，其中：
    - q(z|x) = N(mu, exp(logvar))：编码器输出的后验分布
    - p(z) = N(prior_mu, exp(prior_logvar))：基于物理参数计算的先验分布
    
    Args:
        mu: VAE编码器输出的均值 (B, latent_dim)，表示q(z|x)的均值
        logvar: VAE编码器输出的对数方差 (B, latent_dim)，表示q(z|x)的对数方差
        prior_mu: 物理先验分布p(z)的均值参数（scalar or tensor）
        prior_logvar: 物理先验分布p(z)的对数方差参数（scalar or tensor）
    
    Returns:
        kl_loss: KL散度损失 KL(q(z|x) || p(z))
    """
    # 将先验参数转换为tensor（如果需要）
    if isinstance(prior_mu, (int, float)):
        prior_mu = torch.tensor(prior_mu, device=mu.device, dtype=mu.dtype)
    if isinstance(prior_logvar, (int, float)):
        prior_logvar = torch.tensor(prior_logvar, device=logvar.device, dtype=logvar.dtype)
    
    # 确保维度匹配
    if prior_mu.dim() == 0:
        prior_mu = prior_mu.expand_as(mu)
    if prior_logvar.dim() == 0:
        prior_logvar = prior_logvar.expand_as(logvar)
    
    # KL散度: KL(q(z|x) || p(z))
    # KL = -0.5 * sum(1 + logvar - prior_logvar - (mu - prior_mu)^2 / exp(prior_logvar) - exp(logvar - prior_logvar))
    kl_loss = -0.5 * torch.sum(
        1 + logvar - prior_logvar - 
        (mu - prior_mu).pow(2) / (torch.exp(prior_logvar) + 1e-8) -
        torch.exp(logvar - prior_logvar),
        dim=1
    ).mean()
    
    return kl_loss


def spectrum_to_heatmap(spectrum_1d, min_val=None, max_val=None):
    """
    将1D光谱转换为2D热图
    
    Args:
        spectrum_1d: 1D光谱数组 (3600,)
        min_val: 最小值（用于归一化）
        max_val: 最大值（用于归一化）
    
    Returns:
        heatmap: 2D热图 (60, 60)
        min_val: 实际最小值
        max_val: 实际最大值
    """
    spectrum_1d = np.asarray(spectrum_1d)
    
    if min_val is None:
        min_val = spectrum_1d.min()
    if max_val is None:
        max_val = spectrum_1d.max()
    
    # 归一化
    norm = (spectrum_1d - min_val) / (max_val - min_val + 1e-8)
    
    # 重塑为60x60
    side = int(math.sqrt(len(norm)))
    if side * side != len(norm):
        raise ValueError(f"Spectrum length {len(norm)} is not a perfect square")
    
    # 寻找合适的patch size
    patch_size = 1
    for candidate in [10, 8, 5, 4, 2, 1]:
        if side % candidate == 0:
            patch_size = candidate
            break
    
    num_patches_per_row = side // patch_size
    patches = norm.reshape(-1, patch_size, patch_size)
    rows = [
        np.concatenate(patches[i * num_patches_per_row:(i + 1) * num_patches_per_row], axis=1)
        for i in range(num_patches_per_row)
    ]
    heatmap = np.concatenate(rows, axis=0)
    
    return heatmap.astype(np.float32), min_val, max_val


def inverse_heatmap_to_spectrum(heatmap, spec_min, spec_max):
    """
    将2D热图转换回1D光谱
    
    Args:
        heatmap: 2D热图 (60, 60) 或 (C, 60, 60)
        spec_min: 原始光谱最小值
        spec_max: 原始光谱最大值
    
    Returns:
        spectrum: 1D光谱 (3600,)
    """
    if heatmap.ndim == 3:
        heatmap = heatmap[0] if heatmap.shape[0] == 1 else np.mean(heatmap, axis=0)
    
    side = heatmap.shape[0]
    patch_size = max(p for p in [10, 8, 5, 4, 2, 1] if side % p == 0)
    num_patches = side // patch_size
    
    # 反向重组
    spectra_chunks = []
    for row_idx in range(num_patches):
        row_block = heatmap[row_idx * patch_size:(row_idx + 1) * patch_size]
        patches = np.split(row_block, num_patches, axis=1)
        for patch in patches:
            spectra_chunks.append(patch.flatten())
    
    spec_norm = np.concatenate(spectra_chunks)
    spectrum = spec_norm * (spec_max - spec_min + 1e-8) + spec_min
    
    return spectrum


def get_device(use_cpu=False):
    """获取计算设备"""
    if use_cpu:
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def kl_divergence_loss(mu, logvar):
    """
    计算KL散度损失（与SpectrumViT对齐，使用sum）
    
    Args:
        mu: 潜在空间均值 (B, latent_dim)
        logvar: 潜在空间对数方差 (B, latent_dim)
    
    Returns:
        kl_loss: KL散度损失
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def reconstruction_loss(recon, target):
    """
    计算重建损失（与SpectrumViT对齐，使用sum）
    
    Args:
        recon: 重建的光谱
        target: 目标光谱
    
    Returns:
        recon_loss: 重建损失
    """
    return F.mse_loss(recon, target, reduction='sum')


def kl_annealing(step, beta_max=1.0, k=0.001, x0=500):
    """
    KL散度权重退火
    
    Args:
        step: 当前训练步数
        beta_max: 最大beta值
        k: 退火速率
        x0: 退火中点
    
    Returns:
        beta: 当前beta值
    """
    return float(beta_max / (1 + math.exp(-k * (step - x0))))


def paired_dataset_use_random_split_by_default(
    data_dir,
    source_csv=None,
    target_csv=None,
) -> bool:
    """
    QM9S / QMe14S 等大配对光谱表：与 train 一致，默认使用 random_split 得到的 test 子集。
    若路径中出现这些关键字，则即使显式传入默认命名的 source/target CSV，也不自动「全量当 test」。
    """
    blob = " ".join(
        [
            str(data_dir),
            str(source_csv or ""),
            str(target_csv or ""),
        ]
    ).lower()
    if "qm9s" in blob or "qme14s" in blob:
        return True
    if source_csv and "qm9" in str(source_csv).lower():
        return True
    if target_csv and "qm9" in str(target_csv).lower():
        return True
    return False


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """保存模型检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)


def load_checkpoint(model, optimizer, filepath):
    """加载模型检查点"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
