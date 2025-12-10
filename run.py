import os
from pathlib import Path
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP #wraps your model so gradients get combined across GPUs automatically
from data_loader.dataset import CustomDataLoader
from models.dit_model import DIT
from models.unet import UNet
from train.trainer import Trainer
from utils import set_seed, log

def setup_distributed():
    """
    Initializes distributed training environment for PyTorch.
    
    Returns:
        device, rank, world_size, local_rank
    """
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    # Initialize process group if using multiple processes
    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return device, rank, world_size, local_rank

def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group() #shuts down the communication channels and cleans up resources

def main():
    # Setup distributed training
    device, rank, world_size, local_rank = setup_distributed()

    # Load configuration from YAML
    config_path = Path("configs/dit_config.yaml")    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    
    if rank == 0:
        log(f"Loaded configuration from {config_path}")
    
    # Set a global seed early for reproducibility
    seed = int(os.getenv('SEED', str(config['seed'])))
    set_seed(seed + rank, deterministic=config.get('deterministic_algorithms', True)) # Seed + rank for different seeds per process
    
    if rank == 0:
        log(f"Random seed set to: {seed}")
        log(f"Using device: {device} (Rank: {rank}, World Size: {world_size})")
    
    # Initialize DataLoader
    if rank == 0:
        log("Initializing DataLoader...")
    data_config = config['data']
    data_loader_obj = CustomDataLoader(
        batch_size=data_config['batch_size'],
        shuffle=data_config['shuffle'],
        rank=rank,
        world_size=world_size,
        dataset=data_config.get('dataset')  # Support MNIST or CIFAR-10
    )
    data_loader = data_loader_obj.get_data_loader()

    # Initialize Model based on model_type
    if rank == 0:
        log("Initializing Model...")
    model_config = config['model']
    model_type = model_config.get('model_type', 'dit').lower()
    
    if model_type == 'unet':
        # Initialize UNet
        model = UNet(
            image_size=model_config['image_size'],
            image_channels=model_config['image_channels'],
            base_channels=model_config.get('base_channels', 64),
            channel_multipliers=model_config.get('channel_multipliers', (1, 2, 4, 8)),
            num_res_blocks=model_config.get('num_res_blocks', 2),
            num_classes=10
        ).to(device)
        if rank == 0:
            log("Using UNet architecture")
    else:
        # Initialize DiT (default)
        model = DIT(
            image_size=model_config['image_size'],
            image_channels=model_config['image_channels'],
            patch_size=4,  # Standard patch size
            hidden_dim=model_config['hidden_dim'],
            depth=model_config['depth'],
            num_heads=model_config['num_heads'],
            num_classes=10
        ).to(device)
        if rank == 0:
            log("Using DiT architecture")
    
    # Wrap model with DDP only if using distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if rank == 0:
            log("Model wrapped with DistributedDataParallel")
    else:
        if rank == 0:
            log("Running in single-process mode (no DDP)")

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
    if rank == 0:
        log("Initializing Trainer...")
    diff_config = config['diffusion']
    trainer = Trainer(
        model,
        data_loader,
        optimizer,
        device,
        num_epochs=train_config['num_epochs'],
        checkpoint_path=train_config['checkpoint_path'],
        save_every_n_epochs=train_config['save_every_n_epochs'],
        save_every_epoch=train_config.get('save_every_epoch', False),
        rank=rank,
        num_steps=diff_config['num_steps'],
        beta_start=diff_config['beta_start'],
        beta_end=diff_config['beta_end']
    )
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(train_config['checkpoint_path']).parent
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint if resuming (synchronize across all ranks)
    if dist.is_initialized():
        dist.barrier()  # Wait for rank 0 to create directory
    
    resume_from = train_config.get('resume_from')
    if resume_from:
        if rank == 0:
            log(f"Attempting to resume from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)
    elif train_config.get('auto_resume', False):
        # Auto-resume from latest checkpoint if it exists
        latest_checkpoint = checkpoint_dir / 'latest_checkpoint.pth'
        if latest_checkpoint.exists():
            if rank == 0:
                log(f"Auto-resuming from latest checkpoint: {latest_checkpoint}")
            trainer.load_checkpoint(str(latest_checkpoint))
        elif rank == 0:
            log("No checkpoint found for auto-resume. Starting from scratch.")
    
    # Start training
    if rank == 0:
        log("Starting Training...")
    trainer.train()
    
    # Save training loss visualization (only rank 0)
    if rank == 0:
        log("Saving training loss visualization...")
        trainer.plot_training_loss('outputs/training_loss.png')

    cleanup_distributed()

if __name__ == "__main__":
    main()