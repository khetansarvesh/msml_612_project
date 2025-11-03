import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MNISTDataLoader:
    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dataset = datasets.MNIST(root='./data', train=True, transform=self.transform, download=True)
        self.data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def get_data_loader(self):
        return self.data_loader