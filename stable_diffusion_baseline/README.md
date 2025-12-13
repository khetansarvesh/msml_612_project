Below is a **clean, professional GitHub README** tailored exactly to **your repo structure, scripts, and metrics**.
You can copy-paste this directly as `README.md`.

---

# Diffusion Models: UNet + MHA vs DiT + GQA

This repository provides a **controlled, implementation-level comparison** between two diffusion model denoisers:

* **Baseline:** Stable-Diffusion–style **UNet with spatial Multi-Head Attention (MHA)**
* **Proposed:** **Diffusion Transformer (DiT)** with **Grouped Query Attention (GQA)**

Both models are trained under **identical diffusion and evaluation settings** on MNIST, enabling a fair comparison of **sample quality (FID)**, **GPU memory usage**, and **training throughput**.

---

## Repository Structure

```
.
├── dit/
│   ├── train.py            # Train DiT + GQA diffusion model
│   ├── model.py            # DiT architecture with GQA
│   ├── diffusion.py        # DDPM training + DDIM sampling
│   ├── utils.py            # DDP utilities and image saving
│   └── outputs/
│       └── samples_e*.png  # Generated samples per epoch
│
├── unet/
│   ├── train_unet.py       # Train UNet + MHA diffusion model
│   └── outputs/
│       └── samples_e*.png  # Generated samples per epoch
│
├── checkpoints/
│   ├── dit/
│   │   ├── best.pt
│   │   ├── last.pt
│   │   └── logs/
│   │       └── train_metrics.csv
│   └── unet/
│       ├── best.pt
│       ├── last.pt
│       └── logs/
│           └── train_metrics.csv
│
├── data/                   # MNIST dataset (auto-downloaded)
└── README.md
```

---

## Features

* **DDPM training** with **1000 diffusion timesteps**
* **DDIM sampling** with 50 steps (η = 0)
* **Class-conditional generation**
* **Per-epoch sampling grids** (digits 0–9 per row)
* **FID computation using Inception-v3** (no torchmetrics)
* **Automatic CSV logging** of key metrics:

  * FID
  * Peak GPU memory usage
  * Epoch time
  * Training throughput (images/sec)

---

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

pip install torch torchvision numpy scipy tqdm
```

GPU with CUDA is recommended.

---

## Training: DiT + GQA

Run the following from the `dit/` directory:

```bash
python train.py
```

This will:

* Train a **Diffusion Transformer with Grouped Query Attention**
* Save sample images after every epoch
* Log metrics to:

  ```
  checkpoints/dit/logs/train_metrics.csv
  ```
* Save model checkpoints:

  * `best.pt` (best FID)
  * `last.pt` (final epoch)

### Configurable Arguments

Key arguments in `train.py`:

* `--epochs` (default: 30)
* `--batch_size` (default: 256)
* `--dim` (default: 512)
* `--depth` (default: 6)
* `--num_heads` (default: 8)
* `--num_kv_heads` (default: 2)
* `--fid_n` (default: 100)

---

## Training: UNet + MHA (Baseline)

Run the following from the `unet/` directory:

```bash
python train_unet.py
```

This will:

* Train a **Stable-Diffusion–style UNet with spatial MHA**
* Save sample images after every epoch
* Log metrics to:

  ```
  checkpoints/unet/logs/train_metrics.csv
  ```
* Save model checkpoints:

  * `best.pt`
  * `last.pt`

The UNet baseline uses:

* Convolutional residual blocks
* Spatial self-attention via `nn.MultiheadAttention`
* A significantly smaller parameter footprint

---

## Logged Metrics (CSV)

Each training script produces a CSV with the following columns:

```
epoch,
epoch_time_s,
step_time_ms,
peak_vram_mb,
imgs_per_sec,
fid
```

These metrics allow direct comparison of:

* **Sample quality** (FID ↓)
* **Memory efficiency** (VRAM ↓)
* **Training speed** (imgs/sec ↑)

---

## Key Findings (Summary)

* **DiT + GQA achieves substantially lower FID** than UNet + MHA
* **~4× lower peak GPU memory usage** despite higher parameter count
* **Comparable or higher training throughput**
* Demonstrates the effectiveness of **attention-efficient transformers** in diffusion models

---

## Novelty Statement

While transformer-based diffusion models exist, this work explores the integration of **Grouped Query Attention** within a diffusion transformer and provides a **controlled empirical comparison** against a Stable-Diffusion–style UNet baseline under identical training and evaluation settings.

---

## References

* Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020
* Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022
* Peebles & Xie, *Scalable Diffusion Models with Transformers*, ICCV 2023
* Shazeer, *Multi-Query Attention*, 2019

---

## License

This repository is intended for **educational and research purposes**.

---

If you want, I can:

* add **example plots** (FID vs epoch)
* make a **results table** auto-generated from CSV
* tailor this README to your **MSML612 submission** or **GitHub profile tone**
