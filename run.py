import os
from pathlib import Path
import yaml
import torch
from dataset import MNISTDataLoader
from model import DIT
from trainer import Trainer
from infer import generate_samples
from utils import set_seed, log

def main():

    # Load configuration from YAML
    config_path = Path("config.yaml")    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    log(f"Loaded configuration from {config_path}")
    
    # Set a global seed early for reproducibility
    seed = int(os.getenv('SEED', str(config['seed'])))
    set_seed(seed, deterministic=config.get('deterministic_algorithms', True))
    log(f"Random seed set to: {seed}")

    # Determine device
    device_str = os.getenv('DEVICE', config['device'])
    if device_str == 'cuda' and not torch.cuda.is_available():
        log("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(device_str)
    log(f"Using device: {device}")
    
    # Initialize DataLoader
    log("Initializing DataLoader...")
    data_config = config['data']
    data_loader_obj = MNISTDataLoader(
        batch_size=data_config['batch_size'],
        shuffle=data_config['shuffle']
    )
    data_loader = data_loader_obj.get_data_loader()

    # Initialize Model
    log("Initializing Model...")
    model_config = config['model']
    model = DIT().to(device)

    # Initialize optimizer
    train_config = config['training']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config['learning_rate'],
        betas=train_config.get('adam_betas', [0.9, 0.999]),
        eps=train_config.get('adam_eps', 1e-8),
        weight_decay=train_config.get('weight_decay', 0.0)
    )

    # Initialize Trainer
    log("Initializing Trainer...")
    trainer = Trainer(
        model,
        data_loader,
        optimizer,
        device,
        num_epochs=train_config['num_epochs'],
        checkpoint_path=train_config['checkpoint_path'],
        save_every_n_epochs=train_config['save_every_n_epochs']
    )
    
    # Start training
    log("Starting Training...")
    trainer.train()
    
    # Save training loss visualization
    log("Saving training loss visualization...")
    trainer.plot_training_loss('outputs/training_loss.png')

    # Generate samples after training
    log("Generating Samples...")
    infer_config = config['inference']
    diff_config = config['diffusion']
    
    img = generate_samples(
        model=model,
        num_samples=infer_config['num_samples'],
        im_size=model_config['image_size'],
        im_channels=model_config['image_channels'],
        device=device,
        num_steps=diff_config['num_steps'],
        num_grid_rows=infer_config['num_grid_rows'],
        beta_start=diff_config['beta_start'],
        beta_end=diff_config['beta_end'],
        beta_schedule=diff_config['beta_schedule'],
        save_path=infer_config['save_path'],
        class_label=infer_config['class_label'],
        seed=infer_config.get('seed') or seed
    )
    
    log(f"Sample grid saved to {infer_config['save_path']}")
    img.show()



if __name__ == "__main__":
    main()