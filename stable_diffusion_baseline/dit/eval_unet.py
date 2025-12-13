# eval_unet_ddpm.py
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import inception_v3
from scipy import linalg

from torchvision.utils import save_image

# -------------------------------------------------
# Import YOUR model definitions
# (paste or import if in another file)
# -------------------------------------------------

from model import AttentionUNet, DDPM
# If this script is standalone, you can also paste
# the AttentionUNet + DDPM classes directly here.


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def inception_features(x, inception):
    """
    x: (B,1,28,28) in [0,1]
    returns: (B,2048)
    """
    x = x.repeat(1, 3, 1, 1)                     # grayscale → RGB
    x = F.interpolate(x, size=(299, 299),
                      mode="bilinear",
                      align_corners=False)
    x = (x - 0.5) / 0.5                           # ImageNet-ish norm
    return inception(x)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Standard FID (SciPy sqrtm).
    """
    mu1 = mu1.cpu().numpy()
    mu2 = mu2.cpu().numpy()
    sigma1 = sigma1.cpu().numpy()
    sigma2 = sigma2.cpu().numpy()

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm(
            (sigma1 + offset) @ (sigma2 + offset)
        )

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(
        sigma1 + sigma2 - 2 * covmean
    )
    return float(fid)


@torch.no_grad()
def compute_stats(loader, inception, device, max_items):
    feats = []
    seen = 0

    for x, _ in loader:
        x = x.to(device)
        f = inception_features(x, inception)
        feats.append(f)
        seen += x.size(0)
        if seen >= max_items:
            break

    feats = torch.cat(feats, dim=0)[:max_items]
    mu = feats.mean(dim=0)
    xc = feats - mu
    sigma = (xc.t() @ xc) / (feats.size(0) - 1)
    return mu, sigma


# -------------------------------------------------
# Main
# -------------------------------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--num_gen", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--save_samples", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------
    # Load model
    # -------------------------------------------------
    unet = AttentionUNet().to(device)
    ddpm = DDPM(unet).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    ddpm.load_state_dict(ckpt)
    ddpm.eval()

    # -------------------------------------------------
    # Real MNIST statistics
    # -------------------------------------------------
    transform = transforms.ToTensor()

    test_ds = datasets.MNIST(
        args.data_dir,
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    inception = inception_v3(weights="DEFAULT",
                             transform_input=False).to(device)
    inception.fc = torch.nn.Identity()
    inception.eval()

    print("Computing real data statistics...")
    mu_r, sig_r = compute_stats(
        test_loader,
        inception,
        device,
        max_items=min(args.num_gen, len(test_ds))
    )

    # -------------------------------------------------
    # Generate samples
    # -------------------------------------------------
    print("Generating samples...")
    feats = []
    generated = 0
    samples_for_vis = []

    while generated < args.num_gen:
        b = min(args.batch, args.num_gen - generated)
        x = ddpm.sample(b, device)       # [0,1] approx
        f = inception_features(x, inception)
        feats.append(f)
        generated += b

        if args.save_samples and len(samples_for_vis) < 100:
            samples_for_vis.append(x.cpu())

    feats = torch.cat(feats, dim=0)[:args.num_gen]
    mu_g = feats.mean(dim=0)
    xc = feats - mu_g
    sig_g = (xc.t() @ xc) / (feats.size(0) - 1)

    fid = frechet_distance(mu_r, sig_r, mu_g, sig_g)
    print(f"\nFID (DDPM U-Net, MHA): {fid:.4f}")

    # -------------------------------------------------
    # Optional visualization
    # -------------------------------------------------
    if args.save_samples:
        imgs = torch.cat(samples_for_vis, dim=0)[:100]
        save_image(
            imgs,
            "eval_samples_unet.png",
            nrow=10,
            normalize=True
        )
        print("Saved eval_samples_unet.png")


if __name__ == "__main__":
    main()
