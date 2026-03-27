from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Load the tensor we saved
x = np.load("puthumala_tensor.npy")

mask_img = Image.open("mask.png").convert("L")
mask = np.array(mask_img)

mask = (mask > 128).astype(int)
mask = cv2.resize(mask, (581, 200), interpolation=cv2.INTER_NEAREST)


# --- STEP 1 & 2: Extract Patches and Labels ---
print("\n--- STEP 1 & 2: Extract Patches ---")

patch_size = 64
stride = 32

_, H, W = x.shape
X_patches_list = []
y_labels_list = []

# Sliding window extraction
for i in range(0, H - patch_size + 1, stride):
    for j in range(0, W - patch_size + 1, stride):
        
        # Extract 8-channel data patch
        patch = x[:, i:i+patch_size, j:j+patch_size]
        
        # Extract corresponding mask patch
        mask_patch = mask[i:i+patch_size, j:j+patch_size]
        
        # If ANY pixel in the 64x64 area is a landslide (1), label the whole patch 1
        label = 1 if np.any(mask_patch == 1) else 0
        
        X_patches_list.append(patch)
        y_labels_list.append(label)

X_patches = np.stack(X_patches_list)  # (N, 8, 64, 64)
y_labels = np.array(y_labels_list)    # (N,)

print(f"X_patches shape: {X_patches.shape}")
print(f"y_labels shape: {y_labels.shape}")

# --- STEP 3: Print Stats ---
print("\n--- STEP 3: Patch Statistics ---")
num_patches = len(y_labels)
num_pos = np.sum(y_labels == 1)
num_neg = np.sum(y_labels == 0)

print(f"Total patches: {num_patches}")
print(f"Positive (Landslide) patches: {num_pos} ({(num_pos/num_patches)*100:.1f}%)")
print(f"Negative (Safe) patches: {num_neg} ({(num_neg/num_patches)*100:.1f}%)")

# --- STEP 4: Handle Class Imbalance ---
print("\n--- STEP 4: Class Imbalance Strategy ---")

# We use weighted loss because the dataset is small. Undersampling would waste data.
if num_pos > 0:
    pos_weight_val = num_neg / num_pos
else:
    pos_weight_val = 1.0  # Fallback if no landslides found (shouldn't happen!)

pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)

print(f"Strategy: Using weighted BCE Loss.")
print(f"Calculated pos_weight = {pos_weight_val:.2f}")

# --- STEP 5: Create PyTorch Dataset ---
print("\n--- STEP 5: Create PyTorch Dataset ---")
class LandslideDataset(Dataset):
    def __init__(self, X, y):
        # Convert numpy arrays to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Shape (N, 1) for BCE Loss
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
# Create dataset and a simple dataloader
dataset = LandslideDataset(X_patches, y_labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
print("Dataset and DataLoader instantiated.")
# --- STEP 6: Simple CNN Model ---
print("\n--- STEP 6: Simple CNN Model ---")
class LandslideCNN(nn.Module):
    def __init__(self):
        super(LandslideCNN, self).__init__()
        
        # Block 1
        # Input: (Batch, 8, 64, 64)
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (16, 32, 32)
        
        # Block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (32, 16, 16)
        
        self.flatten = nn.Flatten()
        
        # Fully Connected Layer (32 channels * 16 * 16 = 8192)
        self.fc = nn.Linear(32 * 16 * 16, 1)
        # Note: No Sigmoid here! BCEWithLogitsLoss takes raw logits.
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x
model = LandslideCNN()
print(model)

# --- STEP 7, 8, 9: Training Loop & Metrics ---
print("\n--- STEP 7 & 8: Training Loop ---")
# Hyperparameters
epochs = 20
learning_rate = 0.001
# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Metrics Calculation
        running_loss += loss.item() * inputs.size(0)
        
        # To get predictions from raw logits, we apply Sigmoid then threshold at 0.5
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    
    # Print progress every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:2d}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
# --- STEP 9: Sanity Check ---
print("\n--- STEP 9: Sanity Check ---")
print(f"Final Training Accuracy: {epoch_acc:.2f}%")
if epoch_acc > 50.0:
    print("✅ SUCCESS: Model learned the underlying patterns! (Accuracy > 50%)")
else:
    print("❌ WARNING: Model failed to learn. Check gradients or data distributions.")