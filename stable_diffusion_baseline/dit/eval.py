# eval.py
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from scipy import linalg  # <-- REQUIRED for sqrtm

from model import DiT
from diffusion import Diffusion
from mnist_classifier import SmallMNISTClassifier


# ----------------------------
# Utilities
# ----------------------------

def to_minus_one_to_one(x):
    return x * 2 - 1


def inception_features(x, inception):
    """
    x: (B,1,28,28) in [-1,1]
    returns: (B,2048)
    """
    x = (x + 1) / 2.0                # [0,1]
    x = x.repeat(1, 3, 1, 1)         # grayscale -> RGB
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    x = (x - 0.5) / 0.5              # approx ImageNet norm
    feats = inception(x)
    return feats


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Standard FID computation using SciPy sqrtm.
    This is the canonical implementation used in literature.
    """
    mu1 = mu1.cpu().numpy()
    mu2 = mu2.cpu().numpy()
    sigma1 = sigma1.cpu().numpy()
    sigma2 = sigma2.cpu().numpy()

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Handle numerical imaginary part
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


@torch.no_grad()
def compute_stats(loader, inception, device, max_items):
    feats_list = []
    n = 0
    for x, _ in loader:
        x = x.to(device)
        f = inception_features(x, inception)
        feats_list.append(f)
        n += x.size(0)
        if n >= max_items:
            break

    feats = torch.cat(feats_list, dim=0)[:max_items]
    mu = feats.mean(dim=0)
    xc = feats - mu
    sigma = (xc.t() @ xc) / (feats.size(0) - 1)
    return mu, sigma


# ----------------------------
# Main
# ----------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--num_gen", type=int, default=5000)
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--image_size", type=int, default=28)
    ap.add_argument("--patch_size", type=int, default=4)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_kv_heads", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # Load DiT
    # ----------------------------
    model = DiT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_ch=1,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_classes=10,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    diffusion = Diffusion(timesteps=args.timesteps, device=device)

    # ----------------------------
    # Real MNIST stats
    # ----------------------------
    tfm = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(to_minus_one_to_one),
    ])

    test_ds = datasets.MNIST(
        args.data_dir,
        train=False,
        download=True,
        transform=tfm
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,   # Windows-safe
        pin_memory=True
    )

    inception = inception_v3(weights="DEFAULT", transform_input=False).to(device)
    inception.fc = torch.nn.Identity()
    inception.eval()

    print("Computing real data statistics...")
    mu_r, sig_r = compute_stats(
        test_loader,
        inception,
        device,
        max_items=min(args.num_gen, len(test_ds))
    )

    # ----------------------------
    # Generated stats
    # ----------------------------
    feats_list = []
    total = 0

    print("Generating samples...")
    while total < args.num_gen:
        b = min(args.batch, args.num_gen - total)

        # Row-wise conditioning
        y = torch.arange(0, 10, device=device).repeat((b + 9) // 10)[:b]

        samples = diffusion.ddim_sample(
            model,
            (b, 1, args.image_size, args.image_size),
            y,
            steps=args.ddim_steps,
            eta=0.0
        )

        feats = inception_features(samples, inception)
        feats_list.append(feats)
        total += b

    feats = torch.cat(feats_list, dim=0)[:args.num_gen]
    mu_g = feats.mean(dim=0)
    xc = feats - mu_g
    sig_g = (xc.t() @ xc) / (feats.size(0) - 1)

    # ----------------------------
    # FID
    # ----------------------------
    fid = frechet_distance(mu_r, sig_r, mu_g, sig_g)
    print(f"\nFID (Inception-v3): {fid:.4f}")

    # ----------------------------
    # Optional classifier score
    # ----------------------------
    clf = SmallMNISTClassifier().to(device)
    clf.eval()

    correct = 0
    counted = 0
    for i in range(0, feats.size(0), args.batch):
        x = samples[i:i + args.batch]
        y = torch.arange(0, 10, device=device).repeat((x.size(0) + 9) // 10)[:x.size(0)]
        pred = clf(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        counted += x.size(0)

    print(f"Classifier match (untrained baseline): {correct / max(1, counted):.4f}")


if __name__ == "__main__":
    main()
