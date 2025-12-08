import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, data_loader, optimizer, device, checkpoint_path, save_every_n_epochs, 
                 num_epochs=40, rank=0, num_steps=1000, beta_start=1e-4, beta_end=0.02):
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
        
        # Diffusion parameters (from config)
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alpha_cum_prod = torch.cumprod(1. - self.betas, dim=0).to(device)
        
        # Training time tracking
        self.epoch_times = []  # Time per epoch
        self.total_training_time = 0.0

    def train(self):
        import time
        self.model.train()
        
        training_start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
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
                t = torch.randint(low=0, high=self.num_steps, size=(im.shape[0],), device=self.device)
                noisy_im = self.add_noise(im, noise, t)
                # pass labels to the model so it learns class-conditioned denoising
                noise_pred = self.model(noisy_im, t, y)
                loss = torch.nn.MSELoss()(noise_pred, noise)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            
            epoch_loss = np.mean(losses)
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            self.epoch_losses.append(epoch_loss)
            
            if self.rank == 0:
                print(f'Finished epoch: {epoch + 1} | Loss: {epoch_loss:.6f} | Time: {epoch_time:.2f}s')
            
            # Save checkpoint at specified intervals (only rank 0)
            if self.rank == 0 and (epoch + 1) % self.save_every_n_epochs == 0:
                self.save_periodic_checkpoint(epoch + 1)
            
            # Save best checkpoint if loss improved (only rank 0)
            if self.rank == 0 and epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_best_checkpoint()
        
        # Record total training time
        self.total_training_time = time.time() - training_start_time
        if self.rank == 0:
            print(f'\nTotal training time: {self.total_training_time / 3600:.2f} hours')
            print(f'Average time per epoch: {np.mean(self.epoch_times):.2f}s')

    def add_noise(self, im, noise, t):
        """Add noise to images according to the diffusion schedule."""
        sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod[t])[:, None, None, None]
        sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod[t])[:, None, None, None]
        return sqrt_alpha_cum_prod * im + sqrt_one_minus_alpha_cum_prod * noise

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
        """Plot and save training loss visualization with timing stats."""

        # Create outputs directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Loss curve
        epochs = range(1, len(self.epoch_losses) + 1)
        ax1.plot(epochs, self.epoch_losses, 'b-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add stats text box
        stats_text = f'Final Loss: {self.epoch_losses[-1]:.6f}\n'
        stats_text += f'Best Loss: {self.best_loss:.6f}\n'
        stats_text += f'Total Time: {self.total_training_time / 3600:.2f}h'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Time per epoch
        if len(self.epoch_times) > 0:
            ax2.plot(epochs, self.epoch_times, 'g-', linewidth=2, label='Time per Epoch')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Time (seconds)', fontsize=12)
            ax2.set_title('Time per Epoch', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add time stats
            time_stats = f'Avg: {np.mean(self.epoch_times):.2f}s\n'
            time_stats += f'Min: {np.min(self.epoch_times):.2f}s\n'
            time_stats += f'Max: {np.max(self.epoch_times):.2f}s'
            ax2.text(0.02, 0.98, time_stats, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training loss plot saved to {save_path}")
