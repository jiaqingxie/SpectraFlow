"""
从Parquet文件加载多模态光谱数据集
支持从IR光谱生成H-NMR和C-NMR光谱
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from scipy import interpolate
from pathlib import Path
from tqdm import tqdm
import os


class ParquetSpectrumDataset(Dataset):
    """
    从Parquet文件加载光谱数据集
    支持从IR光谱生成NMR光谱（H-NMR和C-NMR）
    """
    def __init__(self, parquet_path, source_field='ir_spectra', 
                 target_field='h_nmr_spectra', source_size=None, 
                 target_size=None, normalize=True, filter_valid=True):
        """
        Args:
            parquet_path: Parquet文件路径或包含parquet文件的目录
            source_field: 源光谱字段名（默认'ir_spectra'）
            target_field: 目标光谱字段名（'h_nmr_spectra'或'c_nmr_spectra'）
            source_size: 源光谱长度（如果指定，会将源光谱插值到该长度）
            target_size: 目标光谱长度（如果指定，会将目标光谱插值到该长度）
            normalize: 是否归一化到[0, 1]
            filter_valid: 是否过滤掉无效数据（None、空列表等）
        """
        self.source_field = source_field
        self.target_field = target_field
        self.source_size = source_size
        self.target_size = target_size
        self.normalize = normalize
        
        # 加载parquet文件
        print(f"Loading parquet data from: {parquet_path}")
        if os.path.isdir(parquet_path):
            # 如果是目录，查找所有parquet文件
            parquet_files = list(Path(parquet_path).glob("*.parquet"))
            if len(parquet_files) == 0:
                raise ValueError(f"No parquet files found in {parquet_path}")
            print(f"Found {len(parquet_files)} parquet files")
            # 读取所有文件并合并
            dfs = []
            for f in tqdm(parquet_files, desc="Loading parquet files"):
                df = pd.read_parquet(f)
                dfs.append(df)
            self.df = pd.concat(dfs, ignore_index=True)
        else:
            # 单个文件
            self.df = pd.read_parquet(parquet_path)
        
        print(f"Total rows: {len(self.df)}")
        
        # 过滤有效数据
        if filter_valid:
            print(f"Filtering valid data...")
            valid_mask = (
                self.df[source_field].notna() & 
                self.df[target_field].notna()
            )
            # 检查不是空列表
            def is_valid_spectrum(x):
                if pd.isna(x):
                    return False
                if isinstance(x, list):
                    return len(x) > 0
                if isinstance(x, np.ndarray):
                    return len(x) > 0
                return True
            
            valid_mask = valid_mask & self.df[source_field].apply(is_valid_spectrum)
            valid_mask = valid_mask & self.df[target_field].apply(is_valid_spectrum)
            
            self.df = self.df[valid_mask].reset_index(drop=True)
            print(f"Valid rows after filtering: {len(self.df)}")
        
        # 处理光谱数据
        print(f"Processing spectra (source: {source_field}, target: {target_field})...")
        self.source_spectra = []
        self.target_spectra = []
        self.source_min_vals = []
        self.source_max_vals = []
        self.target_min_vals = []
        self.target_max_vals = []
        
        invalid_count = 0
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing spectra"):
            try:
                # 提取源光谱
                source_spec = row[source_field]
                if isinstance(source_spec, list):
                    source_spec = np.array(source_spec, dtype=np.float32)
                elif isinstance(source_spec, np.ndarray):
                    source_spec = source_spec.astype(np.float32)
                else:
                    invalid_count += 1
                    continue
                
                # 处理多维数组（如果是列表的列表，取第一个或展平）
                if source_spec.ndim > 1:
                    if source_spec.shape[0] == 1:
                        source_spec = source_spec[0]
                    else:
                        # 如果是2D，可能需要展平或取平均
                        source_spec = source_spec.flatten()
                
                # 提取目标光谱
                target_spec = row[target_field]
                if isinstance(target_spec, list):
                    target_spec = np.array(target_spec, dtype=np.float32)
                elif isinstance(target_spec, np.ndarray):
                    target_spec = target_spec.astype(np.float32)
                else:
                    invalid_count += 1
                    continue
                
                # 处理多维数组
                if target_spec.ndim > 1:
                    if target_spec.shape[0] == 1:
                        target_spec = target_spec[0]
                    else:
                        target_spec = target_spec.flatten()
                
                # 确保是1D数组
                source_spec = source_spec.flatten()
                target_spec = target_spec.flatten()
                
                # 检查长度
                if len(source_spec) == 0 or len(target_spec) == 0:
                    invalid_count += 1
                    continue
                
                # 如果指定了长度，进行插值
                if source_size is not None and len(source_spec) != source_size:
                    source_spec = self._interpolate(source_spec, source_size)
                
                if target_size is not None and len(target_spec) != target_size:
                    target_spec = self._interpolate(target_spec, target_size)
                
                # 归一化
                source_min = source_spec.min()
                source_max = source_spec.max()
                target_min = target_spec.min()
                target_max = target_spec.max()
                
                if normalize:
                    source_norm = (source_spec - source_min) / (source_max - source_min + 1e-8)
                    target_norm = (target_spec - target_min) / (target_max - target_min + 1e-8)
                else:
                    source_norm = source_spec
                    target_norm = target_spec
                
                self.source_spectra.append(source_norm)
                self.target_spectra.append(target_norm)
                self.source_min_vals.append(source_min)
                self.source_max_vals.append(source_max)
                self.target_min_vals.append(target_min)
                self.target_max_vals.append(target_max)
                
            except Exception as e:
                invalid_count += 1
                if invalid_count <= 5:  # 只打印前5个错误
                    print(f"Error processing row {idx}: {e}")
                continue
        
        if invalid_count > 0:
            print(f"Warning: {invalid_count} rows were skipped due to processing errors")
        
        print(f"Successfully processed {len(self.source_spectra)} spectra")
        if len(self.source_spectra) > 0:
            print(f"Source spectrum shape: {self.source_spectra[0].shape}")
            print(f"Target spectrum shape: {self.target_spectra[0].shape}")
    
    def _interpolate(self, spectrum, target_length):
        """插值到指定长度"""
        if len(spectrum) == target_length:
            return spectrum.copy()
        x_old = np.linspace(0, 1, len(spectrum))
        x_new = np.linspace(0, 1, target_length)
        return interpolate.interp1d(x_old, spectrum, kind='linear', 
                                   fill_value='extrapolate', bounds_error=False)(x_new)
    
    def __len__(self):
        return len(self.source_spectra)
    
    def __getitem__(self, idx):
        source_spec = self.source_spectra[idx]  # (L_source,)
        target_spec = self.target_spectra[idx]  # (L_target,)
        
        # 转换为tensor并添加channel维度: (1, L)
        source_tensor = torch.tensor(source_spec, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor(target_spec, dtype=torch.float32).unsqueeze(0)
        
        return source_tensor, target_tensor


def get_parquet_loaders(parquet_path, source_field='ir_spectra', 
                        target_field='h_nmr_spectra', batch_size=32,
                        source_size=None, target_size=None, 
                        normalize=True, seed=42):
    """
    获取从Parquet文件加载的数据加载器
    
    Args:
        parquet_path: Parquet文件路径或目录
        source_field: 源光谱字段名
        target_field: 目标光谱字段名
        batch_size: 批次大小
        source_size: 源光谱长度
        target_size: 目标光谱长度
        normalize: 是否归一化
        seed: 随机种子
    """
    dataset = ParquetSpectrumDataset(
        parquet_path=parquet_path,
        source_field=source_field,
        target_field=target_field,
        source_size=source_size,
        target_size=target_size,
        normalize=normalize,
        filter_valid=True
    )
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # 使用固定的随机种子确保训练和测试时的分割一致
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

