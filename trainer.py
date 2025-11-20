import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, data_loader, optimizer, device, checkpoint_path, save_every_n_epochs, num_epochs=40, rank=0):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path
        self.save_every_n_epochs = save_every_n_epochs
        self.rank = rank
        self.epoch_losses = []  # Track losses for visualization
        self.best_loss = float('inf')
        self.checkpoint_dir = Path(checkpoint_path)

    def train(self):
        self.model.train()
        
        for epoch in range(self.num_epochs):
            # Set epoch for DistributedSampler to ensure proper shuffling
            if hasattr(self.data_loader.sampler, 'set_epoch'):
                self.data_loader.sampler.set_epoch(epoch)
                
            losses = []
            # Expect dataloader to yield (image, label)
            for im, y in tqdm(self.data_loader, disable=(self.rank != 0)):
                self.optimizer.zero_grad()
                im = im.float().to(self.device)
                y = y.long().to(self.device)
                noise = torch.randn_like(im).to(self.device)
                t = torch.randint(low=0, high=1000, size=(im.shape[0],), device=self.device)
                noisy_im = self.add_noise(im, noise, t)
                # pass labels to the model so it learns class-conditioned denoising
                noise_pred = self.model(noisy_im, t, y)
                loss = torch.nn.MSELoss()(noise_pred, noise)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            
            epoch_loss = np.mean(losses)
            self.epoch_losses.append(epoch_loss)
            if self.rank == 0:
                print(f'Finished epoch: {epoch + 1} | Loss: {epoch_loss}')
            
            # Save checkpoint at specified intervals (only rank 0)
            if self.rank == 0 and (epoch + 1) % self.save_every_n_epochs == 0:
                self.save_periodic_checkpoint(epoch + 1)
            
            # Save best checkpoint if loss improved (only rank 0)
            if self.rank == 0 and epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_best_checkpoint()

    def add_noise(self, im, noise, t):
        betas = torch.linspace(0.0001, 0.02, 1000).to(self.device)
        alpha_cum_prod = torch.cumprod(1. - betas, dim=0).to(self.device)
        return torch.sqrt(alpha_cum_prod[t])[:, None, None, None] * im + torch.sqrt(1 - alpha_cum_prod[t])[:, None, None, None] * noise

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), self.checkpoint_path)
    
    def save_best_checkpoint(self):
        """Save the best checkpoint (lowest loss)."""
        best_checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        torch.save(self.model.state_dict(), best_checkpoint_path)
        print(f"Best model saved to {best_checkpoint_path}")
    
    def save_periodic_checkpoint(self, epoch):
        """Save checkpoint at specified epoch."""
        periodic_checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(self.model.state_dict(), periodic_checkpoint_path)
        print(f"Checkpoint saved to {periodic_checkpoint_path}")
    
    def plot_training_loss(self, save_path='outputs/training_loss.png'):
        """Plot and save training loss visualization."""

        # Create outputs directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.epoch_losses) + 1), self.epoch_losses, 'b-', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Training loss plot saved to {save_path}")
