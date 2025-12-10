import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler #for splitting dataset across GPUs

class CustomDataLoader:
    def __init__(self, batch_size=32, shuffle=True, rank=0, world_size=1, dataset='cifar10'):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rank = rank #rank id of this process across all machines
        self.world_size = world_size #total number of processes across all GPUs
        self.dataset_name = dataset.lower()
        
        # Setup transforms based on dataset
        if self.dataset_name == 'mnist':
            # MNIST: grayscale 28x28
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # Single channel for grayscale
            ])
            self.dataset = datasets.MNIST(root='./data', train=True, transform=self.transform, download=True)
        else:  # CIFAR-10
            # CIFAR-10: RGB 32x32
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels to [-1, 1]
            ])
            self.dataset = datasets.CIFAR10(root='./data', train=True, transform=self.transform, download=True)
        
        if self.world_size > 1:
            self.sampler = DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            shuffle_loader = False # Sampler handles shuffling
        else:
            self.sampler = None
            shuffle_loader = self.shuffle

        self.data_loader = DataLoader(
            dataset=self.dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle_loader, 
            sampler=self.sampler,
            pin_memory=True,
            num_workers=4
        )

    def get_data_loader(self):
        return self.data_loader