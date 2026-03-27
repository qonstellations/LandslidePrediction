import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import *
from utils import set_seed
from data_builder import build_tensor
from dataset import get_mask, extract_patches, LandslideDataset
from model import LandslideMoE

def train_moe():
    """Main Orchestration Loop for ML Training."""
    set_seed(RANDOM_SEED)
    print("\n========== LANDSLIDE MoE PIPELINE ==========")
    
    # 1. ENSURE TENSOR
    if not os.path.exists(TENSOR_OUTPUT_PATH):
        x = build_tensor()
    else:
        print(f"[DATA] Loading cached map tensor: {TENSOR_OUTPUT_PATH}")
        x = np.load(TENSOR_OUTPUT_PATH)
        
    # 2. PREPARE PATCHES & LABELS
    target_shape = (x.shape[1], x.shape[2])
    print(f"\n[MAP] Tensor Shape: {x.shape}")
    print("[MAP] Extracting sliding patches and formatting labels...")
    
    mask = get_mask(MASK_PATH, target_shape)
    X_patches, y_labels = extract_patches(x, mask)
    
    num_pos = np.sum(y_labels == 1)
    num_neg = np.sum(y_labels == 0)
    pos_weight_val = (num_neg / num_pos) if num_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
    
    print(f"[EXTRACT] Patches: {X_patches.shape[0]} | Target Windows: {X_patches.shape[2]}x{X_patches.shape[3]}")
    print(f"[LABELS]  Landslide(+): {num_pos} | Safe(-): {num_neg} | Computed Loss Weight: {pos_weight_val:.2f}")
    
    # 3. DATALOADER
    dataset = LandslideDataset(X_patches, y_labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. INITIALIZE MODEL
    print("\n[MODEL] Compiling Mixture of Experts...")
    model = LandslideMoE()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 5. TRAINING LOOP
    print("\n[TRAIN] Beginning Optmization Loop:")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        e1_dominance, e2_dominance = 0, 0
        
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            
            outputs, weights = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Metrics
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Track routing tendencies
            e1_dominance += (weights[:, 0] > weights[:, 1]).sum().item()
            e2_dominance += (weights[:, 1] > weights[:, 0]).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            e1_pct = (e1_dominance / total) * 100
            e2_pct = (e2_dominance / total) * 100
            print(f"  -> Epoch [{epoch+1:2d}/{EPOCHS}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | (Router: E1={e1_pct:.1f}%, E2={e2_pct:.1f}%)")

    print("\n========== PIPELINE COMPLETE ==========")
    if epoch_acc > 50.0:
        print("✅ SUCCESS: Model successfully identified spatial land-cover patterns!")
    else:
        print("❌ WARNING: Model failed to learn. Investigate data anomalies.")

if __name__ == "__main__":
    train_moe()
