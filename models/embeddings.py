import torch

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
