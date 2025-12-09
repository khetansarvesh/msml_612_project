"""
Diffusion Transformer (DIT) model for class-conditioned image generation.

This implementation uses a Vision Transformer architecture with:
- Patchified input images
- Sinusoidal time embeddings
- Class conditioning via embeddings
- Multi-head grouped query attention (GQA)
"""

import torch
import torch.nn as nn
from einops import rearrange
from .attention import TransformerEncoderLayer


def get_time_embedding(time_steps, temb_dim):
    """
    Generate sinusoidal time embeddings.
    
    Args:
        time_steps: 1D tensor of timesteps [B]
        temb_dim: Dimension of output embeddings
    
    Returns:
        Time embeddings of shape [B, temb_dim]
    """
    factor = 10000 ** (
        torch.arange(
            start=0, 
            end=temb_dim // 2, 
            dtype=torch.float32, 
            device=time_steps.device
        ) / (temb_dim // 2)
    )
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    return torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)


class DIT(nn.Module):
    """
    Diffusion Transformer (DIT) model.
    
    Args:
        image_size: Size of input images (assumed square)
        image_channels: Number of input channels (1 for grayscale, 3 for RGB)
        patch_size: Size of patches to extract
        hidden_dim: Hidden dimension of transformer
        depth: Number of transformer layers
        num_heads: Number of attention heads
        num_classes: Number of classes for conditioning
    """
    
    def __init__(
        self, 
        image_size=32,
        image_channels=3,
        patch_size=4,
        hidden_dim=768,
        depth=2,
        num_heads=8,
        num_classes=10
    ):
        super().__init__()
        
        self.image_size = image_size
        self.image_channels = image_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = image_channels * patch_size * patch_size
        
        # Patch embedding: projects flattened patches to hidden_dim
        self.patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Position embedding for patches + 1 time token
        # We concatenate time token at the end, so we need num_patches + 1 positions
        self.position_embedding = nn.Parameter(
            data=torch.randn(1, self.num_patches + 1, hidden_dim),
            requires_grad=True
        )
        
        self.embedding_dropout = nn.Dropout(p=0.1)
        
        # Time embedding projection with activation
        self.time_proj = nn.Sequential(
            nn.Linear(128, 400),
            nn.GELU(),
            nn.Linear(400, hidden_dim)
        )
        
        # Class conditioning projection with activation
        self.class_emb = nn.Embedding(num_classes, 128)
        self.class_proj = nn.Sequential(
            nn.Linear(128, 400),
            nn.GELU(),
            nn.Linear(400, hidden_dim)
        )
        
        # Transformer encoder layers (create separate instances)
        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    dim=hidden_dim, 
                    num_heads_q=num_heads, 
                    num_heads_kv=max(2, num_heads // 4),  # GQA: fewer KV heads
                    head_dim=hidden_dim // num_heads
                )
                for _ in range(depth)
            ]
        )
        
        # Output projection: back to patch dimensions
        self.proj_out = nn.Linear(hidden_dim, patch_dim)
    
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
        
        # Get time embeddings
        t_emb = get_time_embedding(t, 128)  # [B, 128]
        time_vec = self.time_proj(t_emb)    # [B, hidden_dim]
        
        # Get class embeddings and add to time embeddings
        cls_emb = self.class_emb(y)         # [B, 128]
        cls_vec = self.class_proj(cls_emb)  # [B, hidden_dim]
        
        # Combine time and class conditioning
        cond_vec = time_vec + cls_vec       # [B, hidden_dim]
        cond_token = cond_vec.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Patchify: [B, C, H, W] -> [B, num_patches, patch_dim]
        patches = rearrange(
            x, 
            'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)', 
            ph=self.patch_size, 
            pw=self.patch_size
        )
        
        # Patch embedding
        patch_tokens = self.patch_embedding(patches)  # [B, num_patches, hidden_dim]
        
        # Concatenate conditioning token at the end
        tokens = torch.cat([patch_tokens, cond_token], dim=1)  # [B, num_patches+1, hidden_dim]
        
        # Add position embeddings
        tokens = tokens + self.position_embedding
        
        # Dropout
        tokens = self.embedding_dropout(tokens)
        
        # Transformer
        tokens = self.transformer_encoder(tokens)
        
        # Remove conditioning token and project back to patches
        patch_tokens_out = tokens[:, :-1, :]  # Remove last token (conditioning)
        patches_out = self.proj_out(patch_tokens_out)  # [B, num_patches, patch_dim]
        
        # Unpatchify: [B, num_patches, patch_dim] -> [B, C, H, W]
        output = rearrange(
            patches_out,
            'b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)',
            ph=self.patch_size,
            pw=self.patch_size,
            nh=self.image_size // self.patch_size,
            nw=self.image_size // self.patch_size
        )
        
        return output