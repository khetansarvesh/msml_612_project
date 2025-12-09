import os
from pathlib import Path
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP #wraps your model so gradients get combined across GPUs automatically
from data_loader.dataset import CIFAR10DataLoader
from models.dit_model import DIT
from trainer import Trainer
from inference.infer import generate_samples, generate_timestep_grid
from utils import set_seed, log

def setup_distributed():
    # Initialize process group using environment variables (set by torchrun)
    dist.init_process_group(backend="nccl", init_method="env://")
    #nccl - NVIDIA’s fast library for GPU-to-GPU messaging (best for GPUs).
    # Get rank and world size
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    return device, rank, world_size, local_rank

def cleanup_distributed():
    dist.barrier() #right before destroying to ensure all processes have finished
    dist.destroy_process_group() #shuts down the communication channels and cleans up resources

def main():
    # Setup distributed training
    device, rank, world_size, local_rank = setup_distributed()

    # Load configuration from YAML
    config_path = Path("config.yaml")    
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
    data_loader_obj = CIFAR10DataLoader(
        batch_size=data_config['batch_size'],
        shuffle=data_config['shuffle'],
        rank=rank,
        world_size=world_size
    )
    data_loader = data_loader_obj.get_data_loader()

    # Initialize Model
    if rank == 0:
        log("Initializing Model...")
    model_config = config['model']
    model = DIT(
        image_size=model_config['image_size'],
        image_channels=model_config['image_channels'],
        patch_size=4,  # Standard patch size
        hidden_dim=model_config['hidden_dim'],
        depth=model_config['depth'],
        num_heads=model_config['num_heads'],
        num_classes=10  # CIFAR-10 has 10 classes
    ).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

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

    # TODO: Uncomment this section to generate samples after training completes
    # Generate samples after training (only rank 0)
    # if rank == 0:
    #     log("Generating Samples...")
    #     infer_config = config['inference']
    #     diff_config = config['diffusion']
    #     
    #     # Use model.module for inference to unwrap DDP
    #     inference_model = model.module
    #     
    #     img = generate_samples(
    #         model=inference_model,
    #         num_samples=infer_config['num_samples'],
    #         im_size=model_config['image_size'],
    #         im_channels=model_config['image_channels'],
    #         device=device,
    #         num_steps=diff_config['num_steps'],
    #         num_grid_rows=infer_config['num_grid_rows'],
    #         beta_start=diff_config['beta_start'],
    #         beta_end=diff_config['beta_end'],
    #         beta_schedule=diff_config['beta_schedule'],
    #         save_path=infer_config['save_path'],
    #         class_label=infer_config['class_label'],
    #         seed=infer_config.get('seed') or seed
    #     )
    #     
    #     log(f"Sample grid saved to {infer_config['save_path']}")
    #     # img.show() # Disable show in distributed environment usually, or keep if local
    #
    #     # Generate timestep grid (10x10: 10 classes x 10 timesteps)
    #     log("Generating timestep grid (10 classes x 10 timesteps)...")
    #     timestep_grid_img = generate_timestep_grid(
    #         model=inference_model,
    #         im_size=model_config['image_size'],
    #         im_channels=model_config['image_channels'],
    #         device=device,
    #         num_steps=diff_config['num_steps'],
    #         beta_start=diff_config['beta_start'],
    #         beta_end=diff_config['beta_end'],
    #         beta_schedule=diff_config['beta_schedule'],
    #         save_path='outputs/timestep_grid_10x10.png',
    #         seed=infer_config.get('seed') or seed
    #     )
    #     log("Timestep grid saved to outputs/timestep_grid_10x10.png")
    #     # timestep_grid_img.show()

    cleanup_distributed()

if __name__ == "__main__":
    main()