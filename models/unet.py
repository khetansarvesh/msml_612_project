"""
UNet architecture for class-conditioned diffusion models.

This serves as a baseline model to compare against the Diffusion Transformer (DiT).
The UNet is a standard CNN-based architecture commonly used in diffusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .embeddings import get_time_embedding


class ResBlock(nn.Module):
    """Residual block with time and class conditioning."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, t_emb):
        """
        Args:
            x: [B, C, H, W]
            t_emb: [B, time_emb_dim] - combined time and class embeddings
        
        Returns:
            [B, out_channels, H, W]
        """
        residual = self.residual_conv(x)
        
        # First conv block
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time/class conditioning
        t_emb_proj = self.time_proj(F.silu(t_emb))[:, :, None, None]  # [B, C, 1, 1]
        h = h + t_emb_proj
        
        # Second conv block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + residual


class DownBlock(nn.Module):
    """Downsampling block with residual connections."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, num_res_blocks=2, downsample=True):
        super().__init__()
        
        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim
            )
            for i in range(num_res_blocks)
        ])
        
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None
    
    def forward(self, x, t_emb):
        for res_block in self.res_blocks:
            x = res_block(x, t_emb)
        
        if self.downsample is not None:
            skip = x
            x = self.downsample(x)
            return x, skip
        else:
            return x, x


class UpBlock(nn.Module):
    """Upsampling block with skip connections."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, num_res_blocks=2, upsample=True):
        super().__init__()
        
        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_channels + out_channels if i == 0 else out_channels,  # First block gets skip connection
                out_channels,
                time_emb_dim
            )
            for i in range(num_res_blocks)
        ])
        
        if upsample:
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.upsample = None
    
    def forward(self, x, skip, t_emb):
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        for res_block in self.res_blocks:
            x = res_block(x, t_emb)
        
        if self.upsample is not None:
            x = self.upsample(x)
        
        return x


class UNet(nn.Module):
    """
    UNet model for class-conditioned diffusion.
    
    Args:
        image_size: Size of input images (assumed square)
        image_channels: Number of input channels (1 for grayscale, 3 for RGB)
        base_channels: Base number of channels (will be multiplied for deeper layers)
        channel_multipliers: Multipliers for channels at each resolution level
        num_res_blocks: Number of residual blocks per level
        num_classes: Number of classes for conditioning
        time_emb_dim: Dimension of time embeddings
    """
    
    def __init__(
        self,
        image_size=32,
        image_channels=3,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        num_classes=10,
        time_emb_dim=128
    ):
        super().__init__()
        
        self.image_size = image_size
        self.image_channels = image_channels
        self.time_emb_dim = time_emb_dim
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4)
        )
        
        # Class embedding
        self.class_emb = nn.Embedding(num_classes, time_emb_dim)
        self.class_proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4)
        )
        
        # Combined embedding dimension
        cond_emb_dim = time_emb_dim * 4
        
        # Initial convolution
        self.init_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        in_ch = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            downsample = (i < len(channel_multipliers) - 1)
            
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, cond_emb_dim, num_res_blocks, downsample)
            )
            channels.append(out_ch)
            in_ch = out_ch
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(in_ch, in_ch, cond_emb_dim),
            ResBlock(in_ch, in_ch, cond_emb_dim)
        )
        
        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            upsample = (i > 0)
            
            self.up_blocks.append(
                UpBlock(in_ch, out_ch, cond_emb_dim, num_res_blocks, upsample)
            )
            in_ch = out_ch
        
        # Output projection
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t, y):
        """
        Forward pass.
        
        Args:
            x: Input images [B, C, H, W]
            t: Timesteps [B] or scalar
            y: Class labels [B] or scalar
        
        Returns:
            Predicted noise of same shape as x
        """
        B = x.shape[0]
        device = x.device
        
        # Convert timesteps and labels to tensors on correct device
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t] * B, device=device, dtype=torch.long)
        else:
            t = t.to(device).long()
        
        if not isinstance(y, torch.Tensor):
            y = torch.tensor([y] * B, device=device, dtype=torch.long)
        else:
            y = y.to(device).long()
        
        # Get time and class embeddings
        t_emb = get_time_embedding(t, self.time_emb_dim)  # [B, time_emb_dim]
        t_emb = self.time_proj(t_emb)  # [B, time_emb_dim * 4]
        
        c_emb = self.class_emb(y)  # [B, time_emb_dim]
        c_emb = self.class_proj(c_emb)  # [B, time_emb_dim * 4]
        
        # Combine time and class conditioning
        cond_emb = t_emb + c_emb  # [B, time_emb_dim * 4]
        
        # Initial convolution
        h = self.init_conv(x)
        
        # Encoder
        skips = [h]
        for down_block in self.down_blocks:
            h, skip = down_block(h, cond_emb)
            skips.append(skip)
        
        # Bottleneck
        for bottleneck_block in self.bottleneck:
            h = bottleneck_block(h, cond_emb)
        
        # Decoder
        for up_block in self.up_blocks:
            skip = skips.pop()
            h = up_block(h, skip, cond_emb)
        
        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h
