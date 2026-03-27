import numpy as np
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from config import PATCH_SIZE, STRIDE

def get_mask(mask_path, target_shape):
    """Load mask, binarize, and re-scale to perfectly match the tensor grid."""
    mask_img = Image.open(mask_path).convert("L")
    mask = np.array(mask_img)
    mask = (mask > 128).astype(int)
    # OpenCV expects (Width, Height) which is (Col, Row)
    mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask

def extract_patches(x, mask):
    """Runs a sliding window over the map to generate instances for ML."""
    _, H, W = x.shape
    X_patches, y_labels = [], []
    
    for i in range(0, H - PATCH_SIZE + 1, STRIDE):
        for j in range(0, W - PATCH_SIZE + 1, STRIDE):
            patch = x[:, i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            mask_patch = mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            
            # Danger criteria (label = 1 if ANY landslide pixel in sector)
            label = 1 if np.any(mask_patch == 1) else 0
            
            X_patches.append(patch)
            y_labels.append(label)
            
    return np.stack(X_patches), np.array(y_labels)

class LandslideDataset(Dataset):
    """Standard PyTorch map dataset wrapper."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
