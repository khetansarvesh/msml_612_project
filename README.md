# Class-Conditioned Diffusion Model on CIFAR-10

A PyTorch implementation of Diffusion Transformers (DiT) with class conditioning for CIFAR-10 image generation.

## Features

- ✅ **Diffusion Transformer (DiT)** architecture with Grouped Query Attention
- ✅ **Class conditioning** via embedding fusion
- ✅ **Multi-GPU training** with DistributedDataParallel (DDP)
- ✅ **Comprehensive evaluation** metrics (FID, IS, visual comparisons)
- ✅ **Training time tracking** and detailed logging
- ✅ **Configurable** hyperparameters via YAML

---

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python test_dataset.py  # Test CIFAR-10 loading
```

---

## Training

### Single GPU / CPU

```bash
torchrun --nproc_per_node=1 run.py
```

### Multi-GPU (Linux with CUDA)

```bash
torchrun --nproc_per_node=4 run.py  # For 4 GPUs
```

### Training Outputs

- **Checkpoints**: `model/cifar10_dit_ckpt.pth` (periodic), `model/best_model.pth` (best loss)
- **Loss curves**: `outputs/training_loss.png` (includes timing statistics)
- **Sample grids**: `outputs/sample_grid.png` `outputs/timestep_grid_10x10.png`

---

## Evaluation

After training, evaluate your model:

```bash
python evaluate_model.py \\
    --checkpoint model/best_model.pth \\
    --num-samples 5000 \\
    --calculate-fid \\
    --calculate-is
```

### Evaluation Outputs

- **FID Score**: Fréchet Inception Distance (lower is better)
- **Inception Score**: IS mean ± std (higher is better)
- **Visual comparison**: `outputs/real_vs_fake.png`
- **Results JSON**: `outputs/evaluation_results.json`

### Example Results File

```json
{
  "num_samples": 5000,
  "checkpoint": "model/best_model.pth",
  "fid_score": 45.23,
  "inception_score_mean": 6.87,
  "inception_score_std": 0.12
}
```

---

## Configuration

Edit `config.yaml` to change hyperparameters:

```yaml
model:
  image_size: 32 # Image resolution
  image_channels: 3 # RGB
  hidden_dim: 384 # Transformer hidden dimension
  depth: 12 # Number of transformer layers
  num_heads: 8 # Number of attention heads

training:
  num_epochs: 40
  learning_rate: 2.0e-4
  batch_size: 32

diffusion:
  num_steps: 1000
  beta_start: 1.0e-4
  beta_end: 0.02
```

---

## Project Structure

```
msml_612_project/
├── config.yaml              # Hyperparameters
├── model.py                 # DiT architecture
├── attention.py             # GQA implementation
├── dataset.py               # CIFAR-10 data loader
├── trainer.py               # Training loop with time tracking
├── infer.py                 # Sampling/generation
├── evaluation.py            # FID, IS, accuracy metrics
├── evaluate_model.py        # Evaluation script
├── run.py                   # Main training script
├── utils.py                 # Helpers (seed, config)
├── requirements.txt         # Dependencies
└── outputs/                 # Generated images and plots
```

---

## Model Architecture

### Diffusion Transformer (DiT)

- **Patch size**: 4×4
- **Patches**: 64 (for 32×32 images)
- **Position embeddings**: 65 (64 patches + 1 time/class token)
- **Attention**: Grouped Query Attention (GQA)
- **Conditioning**: Time + class embeddings fused and added as token

### Class Conditioning

```python
# Time embedding
t_emb = sinusoidal_embedding(t, 128)
time_vec = MLP(t_emb)  # 128 → 400 → hidden_dim

# Class embedding
cls_emb = Embedding(y, 128)
cls_vec = MLP(cls_emb)  # 128 → 400 → hidden_dim

# Combine
cond_token = time_vec + cls_vec
```

---

## Metrics for Report

### Already Tracked

✅ **Training Time**: Total hours, time per epoch  
✅ **Loss Curves**: Saved as plot with statistics  
✅ **Generated Samples**: Class-conditioned grids  
✅ **Timestep Progression**: Denoising visualization

### Can Calculate

✅ **FID Score**: Run `evaluate_model.py`  
✅ **Inception Score**: Run with `--calculate-is`  
✅ **Visual Comparisons**: Automatic in evaluation  
⚠️ **Class Accuracy**: Requires trained CIFAR-10 classifier

---

## Troubleshooting

### NCCL not available (macOS)

Use `gloo` backend instead. The code will auto-detect and use `gloo` on CPU/macOS.

### Out of memory

Reduce `batch_size` in `config.yaml`

### Slow training

- Use GPU if available
- Reduce `num_steps` (e.g., 500 instead of 1000)
- Reduce `depth` or `hidden_dim`

---

## CIFAR-10 Classes

0. Airplane
1. Automobile
2. Bird
3. Cat
4. Deer
5. Dog
6. Frog
7. Horse
8. Ship
9. Truck

---

## References

- **DiT**: Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)
- **DDPM**: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- **FID**: GANs Trained by a Two Time-Scale Update Rule (Heusel et al., 2017)
- **Inception Score**: Improved Techniques for Training GANs (Salimans et al., 2016)

---

# Training Commands for 4 GPUs

## Start Training (4 GPUs)

```bash
# Make sure you're on a machine with 4 CUDA GPUs
cd /Users/sarveshkhetan/work/msml_612_project
source venv/bin/activate

# Train with 4 GPUs
torchrun --nproc_per_node=4 run.py
```

## Expected Performance

### Training Speed

- **Single GPU**: ~5-6 minutes per epoch
- **4 GPUs**: ~1.5-2 minutes per epoch ⚡
- **Total time (200 epochs)**: ~5-7 hours

### Batch Size

- **Per GPU**: 64
- **Effective batch size**: 256 (64 × 4 GPUs)
- **Better than papers** using batch size 128-256

### Checkpoints Saved

Will save at epochs: **40, 80, 120, 160, 200**

## After Training - Evaluation

```bash
# Evaluate all checkpoints for ablation study
python evaluate_model.py --checkpoint model/checkpoint_epoch_40.pth --num-samples 5000
python evaluate_model.py --checkpoint model/checkpoint_epoch_80.pth --num-samples 5000
python evaluate_model.py --checkpoint model/checkpoint_epoch_120.pth --num-samples 5000
python evaluate_model.py --checkpoint model/checkpoint_epoch_160.pth --num-samples 5000
python evaluate_model.py --checkpoint model/checkpoint_epoch_200.pth --num-samples 5000
```

## Expected Results Table

Based on research, you should see:

| Epochs | FID Score | Inception Score | Quality            |
| ------ | --------- | --------------- | ------------------ |
| 40     | ~65       | ~4.5            | Fair               |
| 80     | ~50       | ~5.5            | Good               |
| 120    | ~40       | ~6.5            | Very Good          |
| 160    | ~35       | ~7.0            | Excellent          |
| 200    | ~30-32    | ~7.5            | State-of-art range |

_Note: These are rough estimates. Actual results depend on model size and hyperparameters._

## Monitoring Training

Training will print:

```
Finished epoch: 1 | Loss: 0.245123 | Time: 105.32s
Finished epoch: 2 | Loss: 0.198745 | Time: 98.45s
...
```

Watch for:

- ✅ Loss decreasing steadily
- ✅ Time per epoch staying consistent
- ✅ No CUDA out of memory errors

## Troubleshooting

### If OOM (Out of Memory):

```yaml
# Reduce batch_size in config.yaml
batch_size: 48 # or 32
```

### If GPUs not all being used:

```bash
# Check GPU usage
nvidia-smi

# Should show 4 processes
```

### If training seems slow:

- Check GPU utilization with `nvidia-smi`
- Ensure `num_workers: 4` in config (parallel data loading)
