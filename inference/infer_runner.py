import argparse
from pathlib import Path
import yaml
import torch
from inference.infer import generate_samples, generate_timestep_grid
from utils import set_seed, log, load_model

def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained model')
    parser.add_argument('model', type=str, choices=['dit', 'unet'],help='Model type to run: dit or unet')
    args = parser.parse_args()
    
    model_type = args.model
    log(f"Running inference for: {model_type.upper()}")
    
    # Load model-specific config
    config_path = Path(f'configs/{model_type}_config.yaml')    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    log(f"Loaded configuration from {config_path}")
    
    # Extract config sections
    infer_config = config['inference']
    diff_config = config['diffusion']
    model_config = config['model']
    
    # Determine output directory based on model type
    output_dir = Path(__file__).parent / f'{model_type}_inference'
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    log(f"Random seed set to: {config['seed']}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Using device: {device}")
    
    # Load model
    model = load_model(model_type, model_config, device)
    
    # Load checkpoint from config
    checkpoint_path = Path(config['training']['checkpoint_path']) / 'best_model.pth'    
    log(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load state dict and handle DDP checkpoints (remove "module." prefix)
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Check if this is a DDP checkpoint (keys start with "module.")
    if list(state_dict.keys())[0].startswith('module.'):
        log("Detected DDP checkpoint, removing 'module.' prefix...")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Set output paths based on class being generated
    class_label = infer_config['class_label']
    sample_save_path = output_dir / f'generated_samples_class_{class_label}.png'
    timestep_save_path = output_dir / f'timestep_grid_class_{class_label}.png'
    
    # Generate class-conditioned samples
    log(f"Generating {infer_config['num_samples']} samples for class {infer_config['class_label']}...")
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
        save_path=str(sample_save_path),
        class_label=infer_config['class_label'],
        seed=config['seed']
    )
    log(f"Sample grid saved to {sample_save_path}")
    
    # Generate timestep grid (10x10: 10 classes x 10 timesteps)
    log("Generating timestep grid (10 classes x 10 timesteps)...")
    timestep_grid_img = generate_timestep_grid(
        model=model,
        im_size=model_config['image_size'],
        im_channels=model_config['image_channels'],
        device=device,
        num_steps=diff_config['num_steps'],
        beta_start=diff_config['beta_start'],
        beta_end=diff_config['beta_end'],
        beta_schedule=diff_config['beta_schedule'],
        save_path=str(timestep_save_path),
        seed=config['seed']
    )
    log(f"Timestep grid saved to {timestep_save_path}")
    log("Done!")


if __name__ == "__main__":
    main()
