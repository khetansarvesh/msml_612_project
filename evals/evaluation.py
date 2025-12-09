"""
Evaluation metrics for diffusion models.

Includes:
- FID (Fréchet Inception Distance)
- Inception Score (IS)
- Class conditioning accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class InceptionV3Feature(nn.Module):
    """Inception v3 model for feature extraction."""
    
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.eval()
        
        # Remove the final classification layer
        self.blocks = nn.ModuleList([
            nn.Sequential(*list(inception.children())[:-1]),
        ])
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 3, 299, 299]
        Returns:
            features: [B, 2048]
        """
        # Ensure input is the right size for Inception
        if x.shape[2:] != (299, 299):
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Get features from the last pooling layer
        with torch.no_grad():
            x = self.blocks[0](x)
            if x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.squeeze(3).squeeze(2)
        
        return x


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Fréchet Distance between two Gaussian distributions.
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def get_inception_features(images, model, device, batch_size=50):
    """
    Extract Inception features from images.
    
    Args:
        images: Tensor of images [N, C, H, W] in range [-1, 1] or [0, 1]
        model: InceptionV3Feature model
        device: torch device
        batch_size: Batch size for processing
    
    Returns:
        features: np.array [N, 2048]
    """
    model.eval()
    
    # Normalize to ImageNet stats (Inception expects this)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # If images are in [-1, 1], convert to [0, 1]
    if images.min() < 0:
        images = (images + 1) / 2
    
    features = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
            batch = images[i:i+batch_size].to(device)
            
            # Normalize
            batch_norm = torch.stack([normalize(img) for img in batch])
            
            # Get features
            feat = model(batch_norm)
            features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def calculate_fid(real_images, fake_images, device='cpu', batch_size=50):
    """
    Calculate FID score between real and generated images.
    
    Args:
        real_images: Tensor [N, C, H, W]
        fake_images: Tensor [M, C, H, W]
        device: torch device
        batch_size: Batch size for feature extraction
    
    Returns:
        fid_score: float
    """
    print("\nCalculating FID Score...")
    
    # Load Inception model
    inception_model = InceptionV3Feature().to(device)
    
    # Get features
    print("Extracting features from real images...")
    real_features = get_inception_features(real_images, inception_model, device, batch_size)
    
    print("Extracting features from generated images...")
    fake_features = get_inception_features(fake_images, inception_model, device, batch_size)
    
    # Calculate statistics
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    print(f"FID Score: {fid_score:.4f}")
    return fid_score


def calculate_inception_score(images, device='cpu', batch_size=50, splits=10):
    """
    Calculate Inception Score.
    
    IS = exp(E[KL(p(y|x) || p(y))])
    
    Args:
        images: Tensor [N, C, H, W]
        device: torch device
        batch_size: Batch size
        splits: Number of splits for calculating mean and std
    
    Returns:
        (mean_is, std_is): tuple of floats
    """
    print("\nCalculating Inception Score...")
    
    # Load Inception model with classification head
    inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    
    # Normalize to ImageNet stats
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # If images are in [-1, 1], convert to [0, 1]
    if images.min() < 0:
        images = (images + 1) / 2
    
    # Get predictions
    preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Getting predictions"):
            batch = images[i:i+batch_size].to(device)
            
            # Normalize and resize
            batch_norm = torch.stack([normalize(img) for img in batch])
            if batch_norm.shape[2:] != (299, 299):
                batch_norm = F.interpolate(batch_norm, size=(299, 299), mode='bilinear', align_corners=False)
            
            # Get predictions
            pred = F.softmax(inception_model(batch_norm), dim=1)
            preds.append(pred.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # Calculate IS for each split
    split_scores = []
    split_size = preds.shape[0] // splits
    
    for k in range(splits):
        part = preds[k * split_size: (k + 1) * split_size, :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * np.log(pyx / (py + 1e-10) + 1e-10)))
        split_scores.append(np.exp(np.mean(scores)))
    
    mean_is = np.mean(split_scores)
    std_is = np.std(split_scores)
    
    print(f"Inception Score: {mean_is:.4f} ± {std_is:.4f}")
    return mean_is, std_is


def calculate_class_conditioning_accuracy(generated_images, target_labels, device='cpu', batch_size=50):
    """
    Verify class conditioning by classifying generated images.
    
    Args:
        generated_images: Tensor [N, C, H, W]
        target_labels: Tensor [N] - the class labels we conditioned on
        device: torch device
        batch_size: Batch size
    
    Returns:
        accuracy: float - percentage of correctly classified images
    """
    print("\nCalculating Class Conditioning Accuracy...")
    
    # Load a pretrained classifier (using ResNet18 trained on CIFAR-10)
    # Note: You might want to use a better CIFAR-10 classifier
    from torchvision.models import resnet18
    classifier = resnet18(pretrained=False, num_classes=10).to(device)
    
    # Load pretrained CIFAR-10 weights if available
    # For now, this is a placeholder - you'd need to train or download a CIFAR-10 classifier
    print("Warning: Using untrained classifier. For accurate results, use a pretrained CIFAR-10 classifier.")
    classifier.eval()
    
    # Normalize
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # If images are in [-1, 1], convert to [0, 1]
    if generated_images.min() < 0:
        generated_images = (generated_images + 1) / 2
    
    # Get predictions
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(generated_images), batch_size), desc="Classifying"):
            batch = generated_images[i:i+batch_size].to(device)
            
            # Normalize
            batch_norm = torch.stack([normalize(img) for img in batch])
            
            # Get predictions
            outputs = classifier(batch_norm)
            preds = torch.argmax(outputs, dim=1)
            predictions.append(preds.cpu())
    
    predictions = torch.cat(predictions)
    
    # Calculate accuracy
    correct = (predictions == target_labels).sum().item()
    accuracy = 100.0 * correct / len(target_labels)
    
    print(f"Class Conditioning Accuracy: {accuracy:.2f}%")
    return accuracy
