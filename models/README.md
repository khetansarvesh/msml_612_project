# UNet vs DiT Model Comparison

This project includes two diffusion model architectures for comparison:

## Models

### 1. **DiT (Diffusion Transformer)** - `models/dit_model.py`

- **Architecture**: Vision Transformer-based
- **Key Features**:
  - Patchified image inputs (4x4 patches)
  - Multi-head Grouped Query Attention (GQA)
  - Sinusoidal time and class embeddings
  - Position embeddings for patches
  - Patch reconstruction via linear projection

### 2. **UNet** - `models/unet.py`

- **Architecture**: Convolutional with U-shaped encoder-decoder
- **Key Features**:
  - Residual blocks with GroupNorm
  - Skip connections between encoder and decoder
  - Multi-scale processing (4 resolution levels)
  - Time and class conditioning via FiLM (Feature-wise Linear Modulation)
  - Progressive downsampling and upsampling

## Usage

### Training with DiT (default)

```python
from models.dit_model import DIT

model = DIT(
    image_size=32,
    image_channels=3,
    patch_size=4,
    hidden_dim=768,
    depth=2,
    num_heads=8,
    num_classes=10
)
```

### Training with UNet

```python
from models.unet import UNet

model = UNet(
    image_size=32,
    image_channels=3,
    base_channels=64,
    channel_multipliers=(1, 2, 4, 8),
    num_res_blocks=2,
    num_classes=10
)
```

## Model Comparison

| Feature              | DiT              | UNet                 |
| -------------------- | ---------------- | -------------------- |
| **Architecture**     | Transformer      | CNN                  |
| **Inductive Bias**   | Minimal          | Strong (local conv)  |
| **Parameters**       | ~4M (depth=2)    | ~10M (base=64)       |
| **Skip Connections** | Self-attention   | Explicit U-Net skips |
| **Compute**          | O(n²) attention  | O(n) convolutions    |
| **Best For**         | Global coherence | Local details        |

## Expected Performance

- **UNet**: Typically faster to train, strong baseline for images
- **DiT**: More scalable, potentially better at long-range dependencies

## Running Experiments

To compare both models, modify `run.py` to import and use either model:

```python
# For UNet
from models.unet import UNet
model = UNet(...)

# For DiT
from models.dit_model import DIT
model = DIT(...)
```

Both models have the same interface: `forward(x, t, y)` where:

- `x`: Input images [B, C, H, W]
- `t`: Timesteps [B]
- `y`: Class labels [B]
