"""Utility helpers for the project.

Provides logging, simple config I/O and a robust `set_seed` helper to make
experiments reproducible across Python, NumPy and PyTorch (CPU and CUDA).
"""
import os
import random
import yaml
import numpy as np
import torch


def log(message: str) -> None:
    print(f"[LOG] {message}")


def save_config(config, filepath: str) -> None:
    with open(filepath, 'w') as f:
        yaml.dump(config, f)


def load_config(filepath: str):
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for Python, NumPy and PyTorch for reproducibility.

    Args:
        seed: integer seed to set.
        deterministic: if True, enable PyTorch deterministic algorithms (may
            raise errors if an operation has no deterministic implementation).

    Notes:
        - The function sets PYTHONHASHSEED, random, numpy, and torch seeds.
        - It also configures cudnn flags to reduce non-determinism.
        - For full determinism across all ops, set deterministic=True, but be
          aware some ops may not support it and PyTorch may raise.
    """
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CUDNN config
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)

    # Encourage use of deterministic algorithms where available (PyTorch 1.8+)
    try:
        torch.use_deterministic_algorithms(bool(deterministic))
    except Exception:
        # Older PyTorch versions may not have this function or some ops may
        # not support deterministic behavior; ignore in that case.
        pass


def detect_model_type(checkpoint_path: str) -> str:
    """
    Detect model type from checkpoint path.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        'dit' or 'unet'
    """
    path_str = str(checkpoint_path).lower()
    
    if 'dit' in path_str:
        return 'dit'
    elif 'unet' in path_str:
        return 'unet'
    else:
        # Default to dit if cannot determine
        log("Warning: Could not detect model type from path. Defaulting to DiT.")
        log("Tip: Include 'dit' or 'unet' in your checkpoint path for automatic detection.")
        return 'dit'


def load_model(model_type: str, model_config: dict, device: torch.device) -> torch.nn.Module:
    """
    Load the appropriate model based on model type.
    
    Args:
        model_type: 'dit' or 'unet'
        model_config: Model configuration dictionary
        device: Device to load model on
        
    Returns:
        Initialized model
    """
    from models.dit_model import DIT
    from models.unet import UNet
    
    if model_type == 'dit':
        log("Initializing DiT model...")
        model = DIT(
            image_size=model_config['image_size'],
            image_channels=model_config['image_channels'],
            patch_size=4,
            hidden_dim=model_config['hidden_dim'],
            depth=model_config['depth'],
            num_heads=model_config['num_heads'],
            num_classes=10
        ).to(device)
    elif model_type == 'unet':
        log("Initializing UNet model...")
        model = UNet(
            image_size=model_config['image_size'],
            image_channels=model_config['image_channels'],
            base_channels=model_config.get('base_channels', 64),
            channel_multipliers=tuple(model_config.get('channel_multipliers', [1, 2, 4, 8])),
            num_res_blocks=model_config.get('num_res_blocks', 2),
            num_classes=10
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model