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