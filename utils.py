import random
import numpy as np
import torch
import rasterio
from rasterio.warp import reproject, Resampling

def set_seed(seed: int):
    """Pin all random scales for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def align_array(src_data, src_transform, src_crs, target_shape, target_transform, target_crs):
    """Spatially aligns via bilinear interpolation to the target grid."""
    dst_data = np.zeros(target_shape, dtype=np.float32)
    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear
    )
    return dst_data

def normalize_percentile(x):
    """Percentile scaling to ignore massive outliers (especially in optical S2 data)."""
    x = np.nan_to_num(x, nan=float(np.nanmean(x)) if not np.isnan(x).all() else 0.0)
    p2, p98 = np.percentile(x, (2, 98))
    if p98 > p2:
        x_clipped = np.clip(x, p2, p98)
        return (x_clipped - p2) / (p98 - p2 + 1e-8)
    return np.zeros_like(x)

def normalize_channel(data):
    """Standard Min-Max for behaviorally sound layers."""
    data = np.nan_to_num(data, nan=float(np.nanmean(data)) if not np.isnan(data).all() else 0.0)
    c_min, c_max = np.min(data), np.max(data)
    if c_max > c_min:
        return (data - c_min) / (c_max - c_min)
    return np.zeros_like(data)
