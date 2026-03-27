import numpy as np
import xarray as xr
import rasterio
import warnings
from rasterio.transform import from_origin

from config import *
from utils import align_array, normalize_channel, normalize_percentile

def build_tensor():
    """Builds and properly aligns the multi-modal geospatial 8-channel tensor."""
    print("Building aligned tensor from raw data (Sentinel-2 Master Grid)...")
    
    # 1. Master Grid Metadata (Sentinel-2)
    with rasterio.open(S2_DIR / S2_BANDS[0]) as src:
        target_shape = (src.height, src.width)
        target_transform = src.transform
        target_crs = src.crs

    # Helper wrapper for alignment
    def align(data, tf, crs):
        return align_array(data, tf, crs, target_shape, target_transform, target_crs)

    # DEM
    print(" -> Loading DEM...")
    with rasterio.open(DEM_PATH) as src:
        dem_aligned = align(src.read(1), src.transform, src.crs)
        
    # Sentinel-1
    print(" -> Loading Sentinel-1...")
    with rasterio.open(S1_PATH) as src:
        s1_aligned = align(src.read(1), src.transform, src.crs)
        
    # Sentinel-2 (Master grid inherently)
    print(" -> Loading Sentinel-2 Optical...")
    s2_data = []
    for band in S2_BANDS:
        with rasterio.open(S2_DIR / band) as src:
            s2_data.append(src.read(1))
    s2_aligned = np.stack(s2_data)
    
    # Soil Moisture
    print(" -> Loading Soil Moisture...")
    sm_files = list(SM_DIR.glob("**/*.tif"))
    sm_aligned_list = []
    for f in sm_files:
        with rasterio.open(f) as src:
            sm_aligned_list.append(align(src.read(1), src.transform, src.crs))
    sm_stack_aligned = np.stack(sm_aligned_list)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sm_aligned_mean = np.nanmean(sm_stack_aligned, axis=0)

    # Rainfall
    print(" -> Loading Rainfall...")
    ds_rain = xr.open_dataset(RAIN_PATH)
    rain_var = list(ds_rain.data_vars)[0]
    rain_data_3d = ds_rain[rain_var].values
    
    lon, lat = ds_rain['lon'].values, ds_rain['lat'].values
    dx = abs(lon[1] - lon[0]) if len(lon) > 1 else 0.25
    dy = abs(lat[1] - lat[0]) if len(lat) > 1 else 0.25
    rain_transform = from_origin(lon.min() - dx/2, lat.max() + dy/2, dx, dy)
    
    # Distinct Rainfall Features
    rain_recent_raw = np.nansum(rain_data_3d[-7:], axis=0)
    rain_long_raw = np.nansum(rain_data_3d, axis=0)
    rain_recent = align(rain_recent_raw, rain_transform, "EPSG:4326")
    rain_long = align(rain_long_raw, rain_transform, "EPSG:4326")

    print("\nNormalizing Channels...")
    dem_norm = normalize_channel(dem_aligned)
    s1_norm = normalize_channel(s1_aligned)
    sm_norm = normalize_channel(sm_aligned_mean)
    rain_r_norm = normalize_channel(rain_recent)
    rain_l_norm = normalize_channel(rain_long)
    
    s2_norm = np.zeros_like(s2_aligned, dtype=np.float32)
    for i in range(3):
        s2_norm[i] = normalize_percentile(s2_aligned[i])

    print("\nStacking Final Tensor...")
    x = np.concatenate([
        np.expand_dims(dem_norm, axis=0),
        np.expand_dims(s1_norm, axis=0),
        np.expand_dims(s2_norm[0], axis=0),
        np.expand_dims(s2_norm[1], axis=0),
        np.expand_dims(s2_norm[2], axis=0),
        np.expand_dims(rain_r_norm, axis=0),
        np.expand_dims(rain_l_norm, axis=0),
        np.expand_dims(sm_norm, axis=0)
    ], axis=0)

    np.save(TENSOR_OUTPUT_PATH, x)
    print(f"✅ Tensor completely rebuilt and saved to {TENSOR_OUTPUT_PATH} (Shape: {x.shape})")
    return x

if __name__ == "__main__":
    build_tensor()
