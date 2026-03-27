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

