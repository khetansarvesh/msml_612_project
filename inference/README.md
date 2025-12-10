# Inference Usage Guide

## Overview

The inference scripts in this folder work with both **DiT** and **UNet** models automatically!

## Quick Start

### Auto-detect Model Type from Path

The script automatically detects which model to use based on your checkpoint path:

```bash
# For DiT model
python inference/infer_runner.py dit

# For UNet model
python inference/infer_runner.py unet
```

### Manual Model Type Selection

You can explicitly specify the model type:

```bash
python inference/infer_runner.py \
    --checkpoint path/to/checkpoint.pth \
    --model-type dit

python inference/infer_runner.py \
    --checkpoint path/to/checkpoint.pth \
    --model-type unet
```

## Command Line Options

| Option          | Description                         | Default       |
| --------------- | ----------------------------------- | ------------- |
| `--checkpoint`  | Path to model checkpoint (required) | -             |
| `--model-type`  | Model type: `dit` or `unet`         | Auto-detected |
| `--config`      | Path to config file                 | `config.yaml` |
| `--class-label` | Class to generate (0-9)             | From config   |
| `--num-samples` | Number of samples                   | From config   |
| `--seed`        | Random seed                         | From config   |
| `--output-dir`  | Output directory                    | Auto-detected |

## Output Files

The script generates two types of visualizations:

1. **`generated_samples.png`** - Grid of samples for a specific class
2. **`timestep_grid_10x10.png`** - 10x10 grid showing denoising progression across all classes

### Output Location

- For DiT: `outputs/dit_outputs/`
- For UNet: `outputs/unet_outputs/`
- Custom: Specify with `--output-dir`

## Examples

### Generate samples for a specific class

```bash
python inference/infer_runner.py \
    --checkpoint outputs/dit_outputs/trained_models/best_model.pth \
    --class-label 3 \
    --num-samples 64
```

### Use custom output directory

```bash
python inference/infer_runner.py \
    --checkpoint outputs/unet_outputs/trained_models/epoch_50.pth \
    --output-dir results/unet_epoch50
```

### Use different random seed

```bash
python inference/infer_runner.py \
    --checkpoint outputs/dit_outputs/trained_models/best_model.pth \
    --seed 12345
```

## How Model Detection Works

The script looks for keywords in the checkpoint path:

- Contains `dit` → Loads DiT model
- Contains `unet` → Loads UNet model
- Otherwise → Defaults to DiT (with warning)

**Tip**: Include the model type in your checkpoint path for automatic detection!
