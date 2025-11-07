import math
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torchvision.utils import make_grid
import torchvision
from tqdm import tqdm
from PIL import Image
import numpy as np
from utils import set_seed


def get_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02, schedule: str = "linear") -> Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_steps)
    elif schedule == "cosine":
        steps = torch.arange(num_steps + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos(((steps / num_steps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = (alphas_cumprod / alphas_cumprod[0]).float()
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def generate_samples(
    model: torch.nn.Module,
    num_samples: int,
    im_size: int,
    im_channels: int,
    device: torch.device,
    num_steps: int = 1000,
    num_grid_rows: int = 8,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    beta_schedule: str = "linear",
    save_path: Optional[str] = None,
    class_label: Optional[int] = None,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Run DDPM-style ancestral sampling and return a PIL image grid.
    model: expects signature model(x: Tensor, t: Tensor) -> noise_pred
    t should be a 1-D tensor of timesteps (dtype torch.long) with length == batch_size OR broadcastable.
    """
    model.eval()
    device = torch.device(device)

    if seed is not None:
        set_seed(int(seed))
        
    betas = get_beta_schedule(num_steps, beta_start, beta_end, schedule=beta_schedule).to(device)  # (num_steps,)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)  # (num_steps,)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

    with torch.no_grad():
        xt = torch.randn((num_samples, im_channels, im_size, im_size), device=device)

        # passing class info to model at every denoising step (accept single int class for all samples)
        y_tensor = torch.full((num_samples,), int(class_label), device=device, dtype=torch.long)

        for i in tqdm(list(reversed(range(num_steps))), desc="Sampling"):
            t_idx = int(i)
            t_tensor = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)

            # pass class conditioning (y_tensor) if available
            noise_pred = model(xt, t_tensor, y=y_tensor)

            # calculating Xt-1 using the derived formula
            mean = (xt - (betas[t_idx] * noise_pred) / sqrt_one_minus_alpha_cumprod[t_idx]) / torch.sqrt(1.0 - betas[t_idx])
            variance = ((1.0 - alpha_cumprod[t_idx - 1]) / (1.0 - alpha_cumprod[t_idx])) * betas[t_idx]
            sigma = torch.sqrt(variance) if t_idx > 0 else 0.0
            xt = mean + sigma * torch.randn_like(xt)

        ims = torch.clamp(xt, -1.0, 1.0).detach().cpu()
        ims = 0.5 * ims + 0.5  # [-1,1] -> [0,1]
        grid = make_grid(ims, nrow=num_grid_rows)
        img = torchvision.transforms.ToPILImage()(grid)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(save_path)

    return img


def generate_timestep_grid(
    model: torch.nn.Module,
    im_size: int,
    im_channels: int,
    device: torch.device,
    num_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    beta_schedule: str = "linear",
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Generate a 10x10 grid showcasing 10 generated samples for each 10 digit classes 
    through 10 timesteps (0, 100, 200, 300, ..., 1000).
    
    Returns a PIL Image of the combined grid.
    """
    model.eval()
    device = torch.device(device)

    if seed is not None:
        set_seed(int(seed))
    
    betas = get_beta_schedule(num_steps, beta_start, beta_end, schedule=beta_schedule).to(device)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

    # Timesteps to capture: 0, 100, 200, ..., 1000
    timestep_indices = [int(num_steps * i / 10) for i in range(10)]  # [0, 100, 200, ..., 900]
    # But we want exactly [0, 100, 200, ..., 1000], so let's adjust:
    # For num_steps=1000: [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100] (reversed indices)
    # When reversed during sampling, these correspond to: [0, 100, 200, ..., 900, 1000]
    timestep_indices = [int(num_steps * (10 - i) / 10) for i in range(10)]

    # Shape for grid: (num_classes, num_timesteps, channels, height, width)
    grid_samples = torch.zeros((10, len(timestep_indices), im_channels, im_size, im_size), device=device)
    
    with torch.no_grad():
        for class_label in range(10):
            print(f"Generating samples for class {class_label}...")
            
            # Start with random noise
            xt = torch.randn((1, im_channels, im_size, im_size), device=device)
            y_tensor = torch.full((1,), class_label, device=device, dtype=torch.long)
            
            # Track which timesteps to capture
            timestep_capture_idx = 0
            
            for i in tqdm(list(reversed(range(num_steps))), desc=f"Class {class_label}", leave=False):
                t_idx = int(i)
                t_tensor = torch.full((1,), t_idx, device=device, dtype=torch.long)
                
                # Check if this is a timestep we want to capture
                if t_idx in timestep_indices:
                    capture_position = timestep_indices.index(t_idx)
                    # Capture the current state (before denoising)
                    grid_samples[class_label, capture_position] = xt.squeeze(0)
                
                # Perform denoising step
                noise_pred = model(xt, t_tensor, y=y_tensor)
                
                mean = (xt - (betas[t_idx] * noise_pred) / sqrt_one_minus_alpha_cumprod[t_idx]) / torch.sqrt(1.0 - betas[t_idx])
                variance = ((1.0 - alpha_cumprod[t_idx - 1]) / (1.0 - alpha_cumprod[t_idx])) * betas[t_idx]
                sigma = torch.sqrt(variance) if t_idx > 0 else 0.0
                xt = mean + sigma * torch.randn_like(xt)
    
    # Normalize images from [-1, 1] to [0, 1]
    grid_samples = torch.clamp(grid_samples, -1.0, 1.0)
    grid_samples = 0.5 * grid_samples + 0.5
    
    # Reshape to (100, channels, height, width) for make_grid
    # This gives us a 10x10 grid (10 classes x 10 timesteps)
    grid_samples_flat = grid_samples.reshape(100, im_channels, im_size, im_size)
    
    # Create the grid with 10 columns (one for each timestep)
    grid = make_grid(grid_samples_flat, nrow=10, padding=2, pad_value=1)
    img = torchvision.transforms.ToPILImage()(grid)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)
        print(f"Grid saved to {save_path}")
    
    return img