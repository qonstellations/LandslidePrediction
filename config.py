from pathlib import Path

# --- DIRECTORIES & PATHS ---
DATA_DIR = Path("./puthumala_training_data")
DEM_PATH = DATA_DIR / "DEM/Copernicus_DEM_30m.tif"
S1_PATH = DATA_DIR / "Sentinel-1/Sentinel-1_2019-12-08_SAR.tif"
S2_DIR = DATA_DIR / "Sentinel-2"
S2_BANDS = ['B02.tif', 'B03.tif', 'B04.tif']
RAIN_PATH = DATA_DIR / "Rainfall/imdlib_rain_2019-01-01_to_2019-12-31_polygon.nc"
SM_DIR = DATA_DIR / "Soil_Mositure"

TENSOR_OUTPUT_PATH = "puthumala_tensor.npy"
MASK_PATH = "mask.png"

# --- ML PIPELINE CONFIG ---
# Patch extraction
PATCH_SIZE = 64
STRIDE = 16

# Training Hyperparameters
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 16
RANDOM_SEED = 42

# Router Logic
ROUTER_NOISE_STD = 0.01
