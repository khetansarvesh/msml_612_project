# Evaluation Usage Guide

## Overview

The evaluation scripts work with both **DiT** and **UNet** models automatically!

## Quick Start

### Auto-detect Model Type from Path

The script automatically detects which model to use based on your checkpoint path:

```bash
# For DiT model
python evals/eval_runner.py --checkpoint outputs/dit_outputs/trained_models/best_model.pth

# For UNet model
python evals/eval_runner.py --checkpoint outputs/unet_outputs/trained_models/best_model.pth
```

## Command Line Options

| Option                 | Description                         | Default       |
| ---------------------- | ----------------------------------- | ------------- |
| `--checkpoint`         | Path to model checkpoint (required) | -             |
| `--model-type`         | Model type: `dit` or `unet`         | Auto-detected |
| `--config`             | Path to config file                 | `config.yaml` |
| `--num-samples`        | Number of samples to generate       | 5000          |
| `--batch-size`         | Batch size for evaluation           | 50            |
| `--seed`               | Random seed                         | 42            |
| `--output-dir`         | Output directory                    | Auto-detected |
| `--calculate-fid`      | Calculate FID score                 | True          |
| `--calculate-is`       | Calculate Inception Score           | True          |
| `--calculate-accuracy` | Calculate class accuracy            | False         |

## Evaluation Metrics

The script calculates the following metrics:

### 1. **FID (Fréchet Inception Distance)**

- Measures similarity between real and generated image distributions
- Lower is better
- Range: 0 to ∞

### 2. **Inception Score (IS)**

- Measures quality and diversity of generated images
- Higher is better
- Reported as mean ± std

### 3. **Class Conditioning Accuracy** (Optional)

- Measures how well the model generates the requested class
- Requires a trained CIFAR-10 classifier
- Higher is better

## Output Files

The script generates:

1. **`real_vs_fake.png`** - Side-by-side comparison of real vs generated images
2. **`evaluation_results.json`** - JSON file with all metrics

### Output Location

- For DiT: `outputs/dit_outputs/`
- For UNet: `outputs/unet_outputs/`
- Custom: Specify with `--output-dir`

## Examples

### Quick evaluation with default settings

```bash
python evals/eval_runner.py \
    --checkpoint outputs/dit_outputs/trained_models/best_model.pth
```

### Full evaluation with all metrics

```bash
python evals/eval_runner.py \
    --checkpoint outputs/unet_outputs/trained_models/best_model.pth \
    --num-samples 10000 \
    --calculate-accuracy
```

### Custom output directory and seed

```bash
python evals/eval_runner.py \
    --checkpoint outputs/dit_outputs/trained_models/epoch_100.pth \
    --output-dir results/dit_epoch100 \
    --seed 12345
```

### Minimal evaluation (faster)

```bash
python evals/eval_runner.py \
    --checkpoint outputs/unet_outputs/trained_models/best_model.pth \
    --num-samples 1000 \
    --batch-size 100
```

## Evaluation Results

Results are saved as JSON with the following structure:

```json
{
  "model_type": "dit",
  "num_samples": 5000,
  "checkpoint": "outputs/dit_outputs/trained_models/best_model.pth",
  "seed": 42,
  "fid_score": 45.32,
  "inception_score_mean": 5.67,
  "inception_score_std": 0.12
}
```

## Comparing Models

To compare DiT and UNet performance:

1. Run evaluation on both models with the same settings:

```bash
# Evaluate DiT
python evals/eval_runner.py \
    --checkpoint outputs/dit_outputs/trained_models/best_model.pth \
    --num-samples 10000 \
    --seed 42

# Evaluate UNet
python evals/eval_runner.py \
    --checkpoint outputs/unet_outputs/trained_models/best_model.pth \
    --num-samples 10000 \
    --seed 42
```

2. Compare the results in:
   - `outputs/dit_outputs/evaluation_results.json`
   - `outputs/unet_outputs/evaluation_results.json`

## Notes

- FID calculation requires generating images which can be slow
- Use smaller `--num-samples` for faster evaluation during development
- Increase `--num-samples` to 10,000+ for publication-quality metrics
- Set `--seed` for reproducible evaluations
