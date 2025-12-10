"""
Evaluate trained diffusion model.

This script automatically detects the model type (DiT or UNet) from the checkpoint path
and calculates evaluation metrics (FID, IS, class accuracy).

Usage:
    python evals/eval_runner.py --checkpoint outputs/dit_outputs/trained_models/best_model.pth
    python evals/eval_runner.py --checkpoint outputs/unet_outputs/trained_models/best_model.pth
"""

import argparse
import torch
import yaml
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
import json

from evals.evaluation import calculate_fid, calculate_inception_score, calculate_class_conditioning_accuracy
from inference.infer import get_beta_schedule
from utils import set_seed, detect_model_type, load_model





def generate_batch_for_evaluation(model, num_samples, config, device, class_label=None):
    """Generate a batch of images for evaluation."""
    model_config = config['model']
    diff_config = config['diffusion']
    
    # Generate noise
    noise = torch.randn(num_samples, model_config['image_channels'], 
                       model_config['image_size'], model_config['image_size']).to(device)
    
    # Sample using the model
    model.eval()
        
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


def create_comparison_grid(real_images, fake_images, save_path):
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
    # Detect model type
    if args.model_type:
        model_type = args.model_type
    else:
        model_type = detect_model_type(args.checkpoint)
    
    print(f"Model type: {model_type.upper()}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Auto-detect from checkpoint path
        checkpoint_path = Path(args.checkpoint)
        if 'dit_outputs' in str(checkpoint_path):
            output_dir = Path('outputs/dit_outputs')
        elif 'unet_outputs' in str(checkpoint_path):
            output_dir = Path('outputs/unet_outputs')
        else:
            output_dir = Path('outputs')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")
    
    # Load model
    print("\nLoading model...")
    model_config = config['model']
    model = load_model(model_type, model_config, device)
    
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
    comparison_path = output_dir / 'real_vs_fake.png'
    create_comparison_grid(real_images, fake_images, comparison_path)
    
    # Calculate metrics
    results = {
        'model_type': model_type,
        'num_samples': args.num_samples,
        'checkpoint': str(checkpoint_path),
        'seed': args.seed,
    }
    
    # Calculate FID
    if args.calculate_fid:
        fid_score = calculate_fid(real_images, fake_images, device=device, batch_size=args.batch_size)
        results['fid_score'] = float(fid_score)
    
    # Calculate IS
    if args.calculate_is:
        is_mean, is_std = calculate_inception_score(fake_images, device=device, batch_size=args.batch_size)
        results['inception_score_mean'] = float(is_mean)
        results['inception_score_std'] = float(is_std)
    
    # Calculate class conditioning accuracy
    if args.calculate_accuracy:
        accuracy = calculate_class_conditioning_accuracy(fake_images, fake_labels, device=device, batch_size=args.batch_size)
        results['class_accuracy'] = float(accuracy)
    
    # Save results
    results_path = output_dir / 'evaluation_results.json'
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
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (e.g., outputs/dit_outputs/trained_models/best_model.pth)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--model-type', type=str, default=None, choices=['dit', 'unet'],
                       help='Model type (auto-detected from path if not specified)')
    parser.add_argument('--num-samples', type=int, default=5000,
                       help='Number of samples to generate for evaluation')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (auto-detected from checkpoint path if not specified)')
    parser.add_argument('--calculate-fid', action='store_true', default=True,
                       help='Calculate FID score')
    parser.add_argument('--calculate-is', action='store_true', default=True,
                       help='Calculate Inception Score')
    parser.add_argument('--calculate-accuracy', action='store_true', default=False,
                       help='Calculate class conditioning accuracy (requires trained classifier)')
    
    args = parser.parse_args()
    main(args)
