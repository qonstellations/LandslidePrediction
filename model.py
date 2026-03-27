import torch
import torch.nn as nn
from config import ROUTER_NOISE_STD

class FeatureExtractor(nn.Module):
    """Core CNN downsampler: Compresses 8-ch patches to 128-dim vectors."""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        
        self.fc_features = nn.Linear(32 * 16 * 16, 128)
        self.relu_features = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu_features(self.fc_features(x))
        return x

class LandslideMoE(nn.Module):
    """Mixture of Experts architecture routing predictions."""
    def __init__(self):
        super(LandslideMoE, self).__init__()
        self.feature_extractor = FeatureExtractor() 
        
        # Expert 1: Generalist
        self.expert1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Expert 2: Deep specialist (Catches Non-Linear Edge Cases)
        self.expert2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Router: Outputs raw logits (Noise injected in forward)
        self.router = nn.Linear(128, 2)
        
    def forward(self, x):
        # 1. 128-D Representation
        features = self.feature_extractor(x)
        
        # 2. Individual Expert Guesses
        out1 = self.expert1(features) 
        out2 = self.expert2(features) 
        
        # 3. Dynamic Routing
        router_logits = self.router(features) 
        if self.training:
            router_logits = router_logits + ROUTER_NOISE_STD * torch.randn_like(router_logits)
            
        weights = torch.softmax(router_logits, dim=1) 
        w1 = weights[:, 0:1] 
        w2 = weights[:, 1:2] 
        
        # 4. MoE Composition
        final_out = (w1 * out1) + (w2 * out2)
        return final_out, weights
