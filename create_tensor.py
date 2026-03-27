from pathlib import Path
import numpy as np
import xarray as xr
import rasterio

from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import warnings

# --- STEP 1: Check Data ---

# Setup base data directory
data_dir = Path("./puthumala_training_data")

# Verify paths
dem_path = data_dir / "DEM/Copernicus_DEM_30m.tif"
s1_path = data_dir / "Sentinel-1/Sentinel-1_2019-12-08_SAR.tif"
s2_dir = data_dir / "Sentinel-2"
rain_path = data_dir / "Rainfall/imdlib_rain_2019-01-01_to_2019-12-31_polygon.nc"
sm_zip_path = data_dir / "Soil_Mositure/2019.zip"

print(f"DEM exists: {dem_path.exists()}")
print(f"S1 exists: {s1_path.exists()}")
print(f"S2 directory exists: {s2_dir.exists()}")
print(f"Rainfall exists: {rain_path.exists()}")
print(f"Soil Moisture ZIP exists: {sm_zip_path.exists()}")

# --- STEP 2: Load DEM ---
print("\n--- STEP 2: DEM ---")
with rasterio.open(dem_path) as src_dem:
    dem_data = src_dem.read(1)  # Read the first band
    dem_meta = src_dem.meta     # Save metadata to align other arrays later
    
    print(f"DEM Shape: {dem_data.shape}")
    print(f"DEM CRS: {src_dem.crs}")
    print(f"DEM Resolution: {src_dem.res}")

# --- STEP 3: Sentinel-2 (B2, B3, B4) ---
print("\n--- STEP 3: Sentinel-2 (B2, B3, B4) ---")
s2_bands = ['B02.tif', 'B03.tif', 'B04.tif']
s2_data_list = []
for band in s2_bands:
    band_path = s2_dir / band
    with rasterio.open(band_path) as src_s2:
        s2_data_list.append(src_s2.read(1))
# Stack into a single numpy array: shape will be (Channels, H, W)
s2_data = np.stack(s2_data_list)
print(f"Sentinel-2 Shape: {s2_data.shape}")

# --- STEP 4: Load Sentinel-1 ---
print("\n--- STEP 4: Sentinel-1 ---")
with rasterio.open(s1_path) as src_s1:
    s1_data = src_s1.read(1)

print(f"Sentinel-1 Shape: {s1_data.shape}")
print(f"Sentinel-1 Min (Raw): {np.nanmin(s1_data):.2f}, Max (Raw): {np.nanmax(s1_data):.2f}")

# Min-Max Normalize to [0, 1]
s1_min, s1_max = np.nanmin(s1_data), np.nanmax(s1_data)
s1_data = (s1_data - s1_min) / (s1_max - s1_min + 1e-8)

print(f"Sentinel-1 Min (Norm): {np.nanmin(s1_data):.2f}, Max (Norm): {np.nanmax(s1_data):.2f}")

# --- STEP 5: Load Rainfall ---
print("\n--- STEP 5: Rainfall ---")
# Open the NetCDF dataset
ds_rain = xr.open_dataset(rain_path)
# Extract the rainfall variable (usually the first/only data variable in imdlib files)
rain_var_name = list(ds_rain.data_vars)[0]
rain_data_3d = ds_rain[rain_var_name].values  # Shape likely: (time, lat, lon)
print(f"Rainfall 3D Shape (Time, Lat, Lon): {rain_data_3d.shape}")
# Aggregate into a single 2D map (summing over the time axis = index 0)
rain_data = np.nansum(rain_data_3d, axis=0)
print(f"Rainfall Aggregated Shape: {rain_data.shape}")

# --- STEP 6: Load Soil Moisture ---
print("\n--- STEP 6: Soil Moisture ---")

# Find all extracted .tif files inside the directory
sm_dir = data_dir / "Soil_Mositure"
sm_files = list(sm_dir.glob("**/*.tif"))
sm_data_list = []

for sm_file in sm_files:
    with rasterio.open(sm_file) as src_sm:
        sm_data_list.append(src_sm.read(1))

# Stack everything into one array (Time, H, W)
sm_stack = np.stack(sm_data_list)
print(f"Soil Moisture Stack Shape: {sm_stack.shape}")

# Aggregate to a single mean soil moisture map
sm_data_mean = np.nanmean(sm_stack, axis=0)
print(f"Soil Moisture Aggregated Shape: {sm_data_mean.shape}")

# --- STEP 7: Align ALL to Sentinel-2 Grid ---
print("\n--- STEP 7: Alignment (Target = Sentinel-2) ---")

from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import warnings

# 1. Get Master Grid Metadata from Sentinel-2
with rasterio.open(s2_dir / s2_bands[0]) as src_s2:
    target_shape = (src_s2.height, src_s2.width)  # Should ideally be (200, 581)
    target_transform = src_s2.transform
    target_crs = src_s2.crs

# Helper function to align an array to the Sentinel-2 master grid
def align_array(src_data, src_transform, src_crs):
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

# 2. Align DEM
with rasterio.open(dem_path) as src:
    dem_aligned = align_array(src.read(1), src.transform, src.crs)

# 3. Align Sentinel-1 (Re-normalize will happen in Step 10)
with rasterio.open(s1_path) as src:
    s1_aligned = align_array(src.read(1), src.transform, src.crs)

# 4. Sentinel-2 (Master grid, we can just use s2_data directly)
s2_aligned = s2_data

# 5. Align Soil Moisture
# We will align the stack first. We'll fix the NaNs properly in Step 8.
sm_aligned_list = []
for sm_file in sm_files:
    with rasterio.open(sm_file) as src:
        sm_aligned_list.append(align_array(src.read(1), src.transform, src.crs))
sm_stack_aligned = np.stack(sm_aligned_list)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    # We still have the empty slice warning here, but we'll deliberately fix it in Step 8
    sm_aligned_mean = np.nanmean(sm_stack_aligned, axis=0)

# 6. Align Rainfall (NetCDF)
lon = ds_rain['lon'].values
lat = ds_rain['lat'].values
dx = abs(lon[1] - lon[0]) if len(lon) > 1 else 0.25
dy = abs(lat[1] - lat[0]) if len(lat) > 1 else 0.25
rain_transform = from_origin(lon.min() - dx/2, lat.max() + dy/2, dx, dy)
rain_crs = "EPSG:4326"

# For now, align the aggregated 2D map from Step 5. We will refine rainfall in Step 9.
rain_aligned = align_array(rain_data, rain_transform, rain_crs)

print(f"Aligned DEM Shape: {dem_aligned.shape}")
print(f"Aligned Sentinel-1 Shape: {s1_aligned.shape}")
print(f"Aligned Sentinel-2 Shape: {s2_aligned.shape}")
print(f"Aligned Soil Moisture Shape: {sm_aligned_mean.shape}")
print(f"Aligned Rainfall Shape: {rain_aligned.shape}")

# --- STEP 8: Fix Soil Moisture NaNs ---
print("\n--- STEP 8: Fix Soil Moisture NaNs ---")

# 1. Identify missing pixels
missing_mask = np.isnan(sm_aligned_mean)
missing_count = np.sum(missing_mask)
total_count = sm_aligned_mean.size
missing_pct = (missing_count / total_count) * 100

print(f"Missing Soil Moisture pixels: {missing_pct:.2f}% ({missing_count} / {total_count})")

# 2. Replace NaNs carefully (using global valid mean)
if missing_count > 0:
    valid_mean = np.nanmean(sm_aligned_mean)
    sm_aligned_mean[missing_mask] = valid_mean
    print(f"Filled NaNs with global valid mean: {valid_mean:.4f}")
else:
    print("No NaNs to fill.")

# Sanity check
print(f"NaNs remaining in Soil Moisture: {np.isnan(sm_aligned_mean).sum()}")

# --- REVISED STEP 9: Improve Rainfall Features ---
print("\n--- STEP 9: Rainfall Features (Distinct) ---")

# 1. Aggregate recent (Trigger = last 7 days) and long-term (Climate = Full Year)
rain_recent_raw = np.nansum(rain_data_3d[-7:], axis=0)
rain_long_raw   = np.nansum(rain_data_3d, axis=0) # Now captures full historical climatic load

# 2. Align
rain_recent = align_array(rain_recent_raw, rain_transform, rain_crs)
rain_long   = align_array(rain_long_raw, rain_transform, rain_crs)

print(f"Rainfall Recent (7d) Shape: {rain_recent.shape}")
print(f"Rainfall Long-term (Total) Shape: {rain_long.shape}")

# --- REVISED STEP 10: Robust Normalization ---
print("\n--- STEP 10: Robust Normalization ---")

def normalize_percentile(x):
    """Percentile scaling to ignore massive outliers (especially in optical S2 data)."""
    x = np.nan_to_num(x, nan=np.nanmean(x) if not np.isnan(x).all() else 0.0)
    p2, p98 = np.percentile(x, (2, 98))
    
    if p98 > p2:
        x_clipped = np.clip(x, p2, p98)
        return (x_clipped - p2) / (p98 - p2 + 1e-8)
    else:
        return np.zeros_like(x)

def normalize_channel(data):
    """Standard Min-Max for well-behaved layers."""
    data = np.nan_to_num(data, nan=np.nanmean(data) if not np.isnan(data).all() else 0.0)
    c_min, c_max = np.min(data), np.max(data)
    if c_max > c_min:
        return (data - c_min) / (c_max - c_min)
    return np.zeros_like(data)

# Normalize DEM, S1, Rain, SM with standard min-max
dem_norm = normalize_channel(dem_aligned)
s1_norm = normalize_channel(s1_aligned)
sm_norm = normalize_channel(sm_aligned_mean)
rain_recent_norm = normalize_channel(rain_recent)
rain_long_norm = normalize_channel(rain_long)

# 🚀 Use Percentile Normalization EXCLUSIVELY for Sentinel-2
s2_norm = np.zeros_like(s2_aligned, dtype=np.float32)
for i in range(3):
    # s2_aligned[i] contains raw S2 values (or scaled by 10000, either way percentile scaling fixes it)
    s2_norm[i] = normalize_percentile(s2_aligned[i])

print(f"DEM Min/Max: {dem_norm.min():.2f} / {dem_norm.max():.2f}")
print(f"S1 Min/Max: {s1_norm.min():.2f} / {s1_norm.max():.2f}")
print(f"S2 B2 Min/Max: {s2_norm[0].min():.2f} / {s2_norm[0].max():.2f}")
print(f"SM Min/Max: {sm_norm.min():.2f} / {sm_norm.max():.2f}")
print(f"Rain Recent Min/Max: {rain_recent_norm.min():.2f} / {rain_recent_norm.max():.2f}")
print(f"Rain Long Min/Max: {rain_long_norm.min():.2f} / {rain_long_norm.max():.2f}")

# --- STEP 11 & 12: Final Channel Order & Stacking ---
print("\n--- STEP 11 & 12: Stack Final Tensor ---")

# Order:
# 0 -> DEM
# 1 -> Sentinel-1
# 2 -> Sentinel-2 B2
# 3 -> Sentinel-2 B3
# 4 -> Sentinel-2 B4
# 5 -> Rain (recent)
# 6 -> Rain (long-term)
# 7 -> Soil Moisture

dem_c    = np.expand_dims(dem_norm, axis=0)
s1_c     = np.expand_dims(s1_norm, axis=0)
s2_b2    = np.expand_dims(s2_norm[0], axis=0)
s2_b3    = np.expand_dims(s2_norm[1], axis=0)
s2_b4    = np.expand_dims(s2_norm[2], axis=0)
rain_r_c = np.expand_dims(rain_recent_norm, axis=0)
rain_l_c = np.expand_dims(rain_long_norm, axis=0)
sm_c     = np.expand_dims(sm_norm, axis=0)

# Concatenate along the channel axis (axis=0)
x = np.concatenate([
    dem_c, s1_c, s2_b2, s2_b3, s2_b4, rain_r_c, rain_l_c, sm_c
], axis=0)

print(f"Final Tensor Shape: {x.shape}")
print(f"Final Tensor Dtype: {x.dtype}")

# --- STEP 13: Sanity Checks ---
print("\n--- STEP 13: Sanity Checks ---")
channel_names = ["DEM", "S1", "S2_B2", "S2_B3", "S2_B4", "Rain_Recent", "Rain_Long", "Soil_Moisture"]

for i, ch_name in enumerate(channel_names):
    ch_data = x[i]
    c_min, c_max = np.min(ch_data), np.max(ch_data)
    c_mean, c_std = np.mean(ch_data), np.std(ch_data)
    has_nan = np.isnan(ch_data).any()
    has_inf = np.isinf(ch_data).any()
    
    print(f"[{i}] {ch_name:13s} | Min: {c_min:.2f}, Max: {c_max:.2f}, Mean: {c_mean:.4f}, Std: {c_std:.4f} | NaNs: {has_nan}, Infs: {has_inf}")

global_nan = np.isnan(x).sum()
global_inf = np.isinf(x).sum()
print(f"\nGlobal NaNs: {global_nan} | Global Infs: {global_inf}")

# --- STEP 14: Visual Validation ---
print("\n--- STEP 14: Visual Validation ---")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Plot DEM (Elevation mapping)
img1 = axes[0].imshow(x[0], cmap='terrain')
axes[0].set_title("Channel 0: DEM (Normalized)")
fig.colorbar(img1, ax=axes[0])
axes[0].axis('off')

# 2. Plot Sentinel-2 Red Band (Vegetation/Land cover index)
img2 = axes[1].imshow(x[4], cmap='Reds')
axes[1].set_title("Channel 4: Sentinel-2 B4 (Red)")
fig.colorbar(img2, ax=axes[1])
axes[1].axis('off')

# 3. Plot Soil Moisture
img3 = axes[2].imshow(x[7], cmap='YlGnBu')
axes[2].set_title("Channel 7: Soil Moisture")
fig.colorbar(img3, ax=axes[2])
axes[2].axis('off')

plt.tight_layout()
plt.savefig("tensor_channels_validation.png")
print("Saved validation plot to 'tensor_channels_validation.png'")

# --- STEP 15: Save Tensor ---
print("\n--- STEP 15: Save Tensor ---")
output_file = "puthumala_tensor.npy"
np.save(output_file, x)
print(f"🎯 SUCCESS! Final tensor saved to {output_file} (Size: {x.nbytes / 1e6:.2f} MB)")
