#!/usr/bin/env python3
"""
1D Neural Networks for multi-label ECG classification.
Includes multiple architectures for baseline comparisons:
- ResNet1D: Main architecture
- InceptionTime1D: Multi-scale convolutions
- SE-ResNet1D: ResNet with Squeeze-and-Excitation
- LightweightCNN: Fast baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialDropout1d(nn.Module):
    """Drops entire channels (better for time series than regular dropout)."""
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # x shape: (batch, channels, seq_len)
        mask = torch.rand(x.size(0), x.size(1), 1, device=x.device) > self.p
        return x * mask.float() / (1 - self.p)


class ResBlock(nn.Module):
    """Residual block with spatial dropout and optional stochastic depth."""
    
    def __init__(self, in_ch, out_ch, kernel_size=15, stride=1, dropout=0.2, drop_path=0.0):
        super().__init__()
        pad = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = SpatialDropout1d(dropout)  # Spatial dropout instead of regular
        self.drop_path = drop_path  # Stochastic depth probability
        
        # Skip connection
        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        
        # Stochastic depth: randomly drop the residual branch during training
        if self.training and self.drop_path > 0:
            if torch.rand(1).item() < self.drop_path:
                return F.relu(identity)
        
        return F.relu(out + identity)


class ResNet1D(nn.Module):
    """
    1D ResNet for ECG classification with proper regularization.
    
    Input:  (batch, n_leads, seq_len)
    Output: (batch, n_classes) logits
    """
    
    def __init__(self, n_leads=12, n_classes=5, base_filters=32,
                 kernel_size=15, num_blocks=3, dropout=0.3, drop_path=0.1):
        super().__init__()
        
        # Smaller initial projection
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, base_filters, 15, 2, 7, bias=False),  # stride=2 for downsampling
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Fewer blocks with stochastic depth
        self.blocks = nn.ModuleList()
        in_ch = base_filters
        for i in range(num_blocks):
            out_ch = base_filters * (2 ** min(i, 2))  # Cap channel growth at 4x
            stride = 2 if i > 0 else 1
            # Linearly increase drop_path probability
            block_drop_path = drop_path * (i / max(num_blocks - 1, 1))
            self.blocks.append(ResBlock(in_ch, out_ch, kernel_size, stride, dropout, block_drop_path))
            in_ch = out_ch
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        final_ch = base_filters * (2 ** min(num_blocks - 1, 2))
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


# ============================================================================
# SQUEEZE-AND-EXCITATION RESNET (SE-ResNet)
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


class SEResBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation."""
    
    def __init__(self, in_ch, out_ch, kernel_size=15, stride=1, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch)
        self.drop = SpatialDropout1d(dropout)
        
        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + identity)


class SEResNet1D(nn.Module):
    """SE-ResNet for ECG classification."""
    
    def __init__(self, n_leads=12, n_classes=5, base_filters=32,
                 kernel_size=15, num_blocks=3, dropout=0.3):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, base_filters, 15, 2, 7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.blocks = nn.ModuleList()
        in_ch = base_filters
        for i in range(num_blocks):
            out_ch = base_filters * (2 ** min(i, 2))
            stride = 2 if i > 0 else 1
            self.blocks.append(SEResBlock(in_ch, out_ch, kernel_size, stride, dropout))
            in_ch = out_ch
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        final_ch = base_filters * (2 ** min(num_blocks - 1, 2))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_ch, n_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# ============================================================================
# INCEPTIONTIME (Multi-scale convolutions)
# ============================================================================

class InceptionBlock(nn.Module):
    """Inception module with multiple kernel sizes for ECG."""
    
    def __init__(self, in_ch, out_ch, kernel_sizes=[9, 19, 39], bottleneck=32):
        super().__init__()
        
        self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, bias=False)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck, out_ch, k, padding=k//2, bias=False)
            for k in kernel_sizes
        ])
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch, 1, bias=False)
        )
        
        n_filters = out_ch * len(kernel_sizes) + out_ch
        self.bn = nn.BatchNorm1d(n_filters)
        
        self.skip = nn.Conv1d(in_ch, n_filters, 1, bias=False) if in_ch != n_filters else nn.Identity()
        self.skip_bn = nn.BatchNorm1d(n_filters)
    
    def forward(self, x):
        identity = self.skip_bn(self.skip(x))
        
        bottleneck = self.bottleneck(x)
        
        outputs = [conv(bottleneck) for conv in self.convs]
        outputs.append(self.maxpool_conv(x))
        
        out = torch.cat(outputs, dim=1)
        out = self.bn(out)
        
        return F.relu(out + identity)


class InceptionTime1D(nn.Module):
    """InceptionTime architecture for ECG classification."""
    
    def __init__(self, n_leads=12, n_classes=5, n_filters=32, 
                 num_blocks=3, dropout=0.3, **kwargs):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, n_filters, 7, 2, 3, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        kernel_sizes = [9, 19, 39]
        n_inception_filters = n_filters * (len(kernel_sizes) + 1)
        
        self.blocks = nn.ModuleList()
        in_ch = n_filters
        for i in range(num_blocks):
            out_ch = n_filters * (i + 1)
            self.blocks.append(InceptionBlock(in_ch, out_ch, kernel_sizes))
            in_ch = out_ch * (len(kernel_sizes) + 1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, n_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


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
    
    model_name_lower = model_name.lower()
    
    if model_name_lower == "resnet1d":
        model = ResNet1D(n_leads=n_leads, n_classes=n_classes, **kwargs)
    elif model_name_lower == "seresnet1d" or model_name_lower == "se-resnet1d":
        model = SEResNet1D(n_leads=n_leads, n_classes=n_classes, **kwargs)
    elif model_name_lower == "inceptiontime" or model_name_lower == "inception1d":
        model = InceptionTime1D(n_leads=n_leads, n_classes=n_classes, **kwargs)
    elif model_name_lower == "lightweight":
        model = LightweightCNN(n_leads=n_leads, n_classes=n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: resnet1d, seresnet1d, inceptiontime, lightweight")
    
    print(f"Model: {model_name}")
    print(f"  Input leads: {n_leads}")
    print(f"  Output classes: {n_classes}")
    print(f"  Parameters: {count_parameters(model):,}")
    
    return model


# ============================================================================
# MODEL COMPARISON UTILITIES
# ============================================================================

def get_all_model_names() -> list:
    """Return list of all available model names."""
    return ["resnet1d", "seresnet1d", "inceptiontime", "lightweight"]


def model_summary(n_leads: int = 12, n_classes: int = 5) -> dict:
    """Get summary of all models."""
    summary = {}
    for name in get_all_model_names():
        model = get_model(name, n_leads, n_classes)
        summary[name] = {
            'parameters': count_parameters(model),
            'input_shape': (1, n_leads, 1000),
        }
        # Compute FLOPs estimate
        x = torch.randn(1, n_leads, 1000)
        summary[name]['output_shape'] = tuple(model(x).shape)
    return summary


if __name__ == "__main__":
    # Test all models
    print("=" * 60)
    print("TESTING ALL MODEL ARCHITECTURES")
    print("=" * 60)
    
    for model_name in get_all_model_names():
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print("=" * 60)
        model = get_model(model_name, n_leads=12, n_classes=5)
        x = torch.randn(8, 12, 1000)
        y = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
