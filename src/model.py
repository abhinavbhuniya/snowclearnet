"""
SnowClearNet – DATSRF Model Architecture & Loader
==================================================
Defines the DATSRF (Dual Attention Transformer for Snow Removal Fusion)
architecture and provides utilities for loading the trained model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention using channel-wise pooling."""

    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(combined))


class DualAttentionBlock(nn.Module):
    """Combines channel + spatial attention after two convolutions."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out)
        out = self.sa(out)
        return self.relu(out + residual)


class FusionBlock(nn.Module):
    """Multi-scale fusion block with 1×1 and 3×3 branches."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Conv2d(out_ch * 2, out_ch, 1, bias=False)

    def forward(self, x):
        return self.fuse(torch.cat([self.branch1(x), self.branch3(x)], dim=1))


# ---------------------------------------------------------------------------
# Main DATSRF network
# ---------------------------------------------------------------------------

class DATSRF(nn.Module):
    """
    Dual Attention Transformer for Snow Removal Fusion.
    Encoder → bottleneck (dual-attention) → decoder with skip connections.
    """

    def __init__(self, in_channels=3, base_channels=64, num_blocks=4):
        super().__init__()

        # Encoder
        self.enc1 = FusionBlock(in_channels, base_channels)
        self.enc2 = FusionBlock(base_channels, base_channels * 2)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck – stacked dual-attention blocks
        self.bottleneck = nn.Sequential(
            *[DualAttentionBlock(base_channels * 2) for _ in range(num_blocks)]
        )

        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = FusionBlock(base_channels * 2 + base_channels * 2, base_channels)
        self.dec1 = FusionBlock(base_channels + base_channels, base_channels)

        # Output head
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)                       # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))            # (B, 128, H/2, W/2)

        # Bottleneck
        b = self.bottleneck(e2)                  # (B, 128, H/2, W/2)

        # Decode
        d2 = self.dec2(torch.cat([self.up(b), e2], dim=1))  # skip from e2 — but sizes need to match after up
        # Upsample d2 back to original resolution before concat with e1
        d2_up = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d2_up, e1], dim=1))

        return self.out_conv(d1)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(weight_path: str, device: torch.device) -> DATSRF:
    """
    Instantiate DATSRF and load pre-trained weights.
    Falls back to random-initialised model if weights file is missing.
    """
    model = DATSRF(in_channels=3, base_channels=64, num_blocks=4)

    try:
        state = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[✓] Loaded model weights from {weight_path}")
    except FileNotFoundError:
        print(f"[!] Weight file '{weight_path}' not found – using randomly initialised model (demo mode).")
    except RuntimeError as e:
        print(f"[!] Could not load weights ({e}) – using randomly initialised model (demo mode).")

    model = model.to(device)
    model.eval()
    return model
