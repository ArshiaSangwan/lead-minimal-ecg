#!/usr/bin/env python3
"""
1D ResNet for multi-label ECG classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Basic residual block with 1D convolutions."""
    
    def __init__(self, in_ch, out_ch, kernel_size=15, stride=1, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        
        # Skip connection (1x1 conv if dimensions change)
        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class ResNet1D(nn.Module):
    """
    1D ResNet for ECG classification.
    
    Input:  (batch, n_leads, seq_len)
    Output: (batch, n_classes) logits
    """
    
    def __init__(self, n_leads=12, n_classes=5, base_filters=64,
                 kernel_size=15, num_blocks=4, dropout=0.2):
        super().__init__()
        
        # Initial projection
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, base_filters, 15, 1, 7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Stack of residual blocks with increasing channels
        self.blocks = nn.ModuleList()
        in_ch = base_filters
        for i in range(num_blocks):
            out_ch = base_filters * (2 ** i)
            stride = 2 if i > 0 else 1
            self.blocks.append(ResBlock(in_ch, out_ch, kernel_size, stride, dropout))
            in_ch = out_ch
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        final_ch = base_filters * (2 ** (num_blocks - 1))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_ch, n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)
    
    def get_features(self, x):
        """Return features before the classification head."""
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.pool(x).squeeze(-1)


class LightweightCNN(nn.Module):
    """
    Minimal CNN for quick experiments.
    Much faster to train but less accurate.
    """
    
    def __init__(self, n_leads=12, n_classes=5, hidden_dim=128, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_leads, 32, 7, 2, 3),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, 2, 2),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, 5, 2, 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(
    model_name: str = "resnet1d",
    n_leads: int = 12,
    n_classes: int = 5,
    **kwargs
) -> nn.Module:
    """Factory function to get model by name."""
    
    if model_name.lower() == "resnet1d":
        model = ResNet1D(n_leads=n_leads, n_classes=n_classes, **kwargs)
    elif model_name.lower() == "lightweight":
        model = LightweightCNN(n_leads=n_leads, n_classes=n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"Model: {model_name}")
    print(f"  Input leads: {n_leads}")
    print(f"  Output classes: {n_classes}")
    print(f"  Parameters: {count_parameters(model):,}")
    
    return model


if __name__ == "__main__":
    # Test models
    print("=" * 60)
    print("Testing ResNet1D (12-lead)")
    print("=" * 60)
    model = get_model("resnet1d", n_leads=12, n_classes=5)
    x = torch.randn(8, 12, 1000)  # (batch, leads, seq_len)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    print("\n" + "=" * 60)
    print("Testing ResNet1D (2-lead)")
    print("=" * 60)
    model = get_model("resnet1d", n_leads=2, n_classes=5)
    x = torch.randn(8, 2, 1000)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    print("\n" + "=" * 60)
    print("Testing Lightweight CNN")
    print("=" * 60)
    model = get_model("lightweight", n_leads=12, n_classes=5)
    x = torch.randn(8, 12, 1000)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
