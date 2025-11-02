# Contents of /mnist_dit_project/mnist_dit_project/src/runner/run.py

import os
import torch
from dataset import MNISTDataLoader
from model import DIT
from trainer import Trainer
from infer import generate_samples

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize DataLoader
    data_loader_obj = MNISTDataLoader(batch_size=32, shuffle=True)
    data_loader = data_loader_obj.get_data_loader()

    # Initialize Model
    model = DIT().to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Initialize Trainer
    trainer = Trainer(model, data_loader, optimizer, device)
    
    # Load existing model weights if available
    checkpoint_path = os.path.join('mnist', 'dit_ckpt.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Start training
    trainer.train()

    # Generate samples after training
    img = generate_samples(
        model=model,
        num_samples=16,
        im_size=32,
        im_channels=3,
        device=device,
        num_steps=1000,
        num_grid_rows=4,
        save_path="outputs/sample_grid.png",
    )
    
    img.show()



if __name__ == "__main__":
    main()