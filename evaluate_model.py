"""
Evaluate trained diffusion model.

Usage:
    python evaluate_model.py --checkpoint model/cifar10_dit_ckpt.pth
"""

import argparse
import torch
import yaml
import numpy as np
from pathlib import Path
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from PIL import Image

from model import DIT
from dataset import CIFAR10DataLoader
from evaluation import calculate_fid, calculate_inception_score, calculate_class_conditioning_accuracy
from infer import generate_samples
from utils import set_seed


def generate_batch_for_evaluation(model, num_samples, config, device, class_label=None):
    """Generate a batch of images for evaluation."""
    model_config = config['model']
    diff_config = config['diffusion']
    
    # Generate noise
    noise = torch.randn(num_samples, model_config['image_channels'], 
                       model_config['image_size'], model_config['image_size']).to(device)
    
    # Sample using the model
    model.eval()
    
    # Use simplified sampling (you can use generate_samples function too)
    from infer import get_beta_schedule
    
    betas = get_beta_schedule(diff_config['num_steps'], 
                              diff_config['beta_start'], 
                              diff_config['beta_end'],
                              diff_config['beta_schedule']).to(device)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
    
    xt = noise
    
    # If class_label is specified, use it for all samples
    if class_label is not None:
        y = torch.full((num_samples,), class_label, device=device, dtype=torch.long)
    else:
        # Random classes
        y = torch.randint(0, 10, (num_samples,), device=device)
    
    with torch.no_grad():
        for i in reversed(range(diff_config['num_steps'])):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = model(xt, t, y)
            
            # Denoise
            mean = (xt - (betas[i] * noise_pred) / sqrt_one_minus_alpha_cumprod[i]) / torch.sqrt(1.0 - betas[i])
            
            if i > 0:
                variance = ((1.0 - alpha_cumprod[i - 1]) / (1.0 - alpha_cumprod[i])) * betas[i]
                sigma = torch.sqrt(variance)
                xt = mean + sigma * torch.randn_like(xt)
            else:
                xt = mean
    
    # Clamp to [-1, 1]
    xt = torch.clamp(xt, -1.0, 1.0)
    
    return xt, y


def load_real_data(num_samples=5000, device='cpu'):
    """Load real CIFAR-10 images."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Sample random subset
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    images = torch.stack([dataset[i][0] for i in indices])
    labels = torch.tensor([dataset[i][1] for i in indices])
    
    return images.to(device), labels.to(device)


def create_comparison_grid(real_images, fake_images, save_path='outputs/real_vs_fake.png'):
    """Create side-by-side comparison of real and generated images."""
    from torchvision.utils import make_grid
    
    # Take first 32 images from each
    n_show = min(32, real_images.shape[0], fake_images.shape[0])
    real_subset = real_images[:n_show]
    fake_subset = fake_images[:n_show]
    
    # Denormalize from [-1, 1] to [0, 1]
    real_subset = (real_subset + 1) / 2.0
    fake_subset = (fake_subset + 1) / 2.0
    
    # Create grids
    real_grid = make_grid(real_subset, nrow=8, padding=2, pad_value=1)
    fake_grid = make_grid(fake_subset, nrow=8, padding=2, pad_value=1)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(real_grid.permute(1, 2, 0).cpu().numpy())
    ax1.set_title('Real CIFAR-10 Images', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(fake_grid.permute(1, 2, 0).cpu().numpy())
    ax2.set_title('Generated Images', fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison grid saved to {save_path}")


def main(args):
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load model
    print("\nLoading model...")
    model_config = config['model']
    model = DIT(
        image_size=model_config['image_size'],
        image_channels=model_config['image_channels'],
        patch_size=4,
        hidden_dim=model_config['hidden_dim'],
        depth=model_config['depth'],
        num_heads=model_config['num_heads'],
        num_classes=10
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = args.checkpoint
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Generate fake images
    print(f"\nGenerating {args.num_samples} images...")
    fake_images_list = []
    fake_labels_list = []
    
    batch_size = 100
    for i in range(0, args.num_samples, batch_size):
        n = min(batch_size, args.num_samples - i)
        fake_batch, labels_batch = generate_batch_for_evaluation(model, n, config, device)
        fake_images_list.append(fake_batch.cpu())
        fake_labels_list.append(labels_batch.cpu())
    
    fake_images = torch.cat(fake_images_list, dim=0)
    fake_labels = torch.cat(fake_labels_list, dim=0)
    print(f"Generated {len(fake_images)} images")
    
    # Load real images
    print(f"\nLoading {args.num_samples} real images...")
    real_images, real_labels = load_real_data(args.num_samples, device='cpu')
    
    # Create comparison visualization
    print("\nCreating comparison visualization...")
    create_comparison_grid(real_images, fake_images, 'outputs/real_vs_fake.png')
    
    # Calculate FID
    if args.calculate_fid:
        fid_score = calculate_fid(real_images, fake_images, device=device, batch_size=args.batch_size)
    
    # Calculate IS
    if args.calculate_is:
        is_mean, is_std = calculate_inception_score(fake_images, device=device, batch_size=args.batch_size)
    
    # Calculate class conditioning accuracy (if you have a trained CIFAR-10 classifier)
    if args.calculate_accuracy:
        accuracy = calculate_class_conditioning_accuracy(fake_images, fake_labels, device=device, batch_size=args.batch_size)
    
    # Save results
    results = {
        'num_samples': args.num_samples,
        'checkpoint': str(checkpoint_path),
    }
    
    if args.calculate_fid:
        results['fid_score'] = float(fid_score)
    if args.calculate_is:
        results['inception_score_mean'] = float(is_mean)
        results['inception_score_std'] = float(is_std)
    if args.calculate_accuracy:
        results['class_accuracy'] = float(accuracy)
    
    # Save to file
    import json
    results_path = 'outputs/evaluation_results.json'
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print('='*60)
    for key, value in results.items():
        print(f"{key}: {value}")
    print('='*60)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate diffusion model')
    parser.add_argument('--checkpoint', type=str, default='model/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--num-samples', type=int, default=5000,
                       help='Number of samples to generate for evaluation')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Batch size for evaluation')
    parser.add_argument('--calculate-fid', action='store_true', default=True,
                       help='Calculate FID score')
    parser.add_argument('--calculate-is', action='store_true', default=True,
                       help='Calculate Inception Score')
    parser.add_argument('--calculate-accuracy', action='store_true', default=False,
                       help='Calculate class conditioning accuracy (requires trained classifier)')
    
    args = parser.parse_args()
    main(args)
