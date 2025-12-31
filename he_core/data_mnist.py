import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_mnist_loaders(batch_size: int = 64, data_root: str = './data', max_samples: int = None):
    """
    Returns (train_loader, test_loader) for MNIST.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize to typical mean/std of MNIST
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        train_ds = datasets.MNIST(data_root, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Error loading MNIST (network issue?): {e}")
        print("Creating Fake Random Dataset for debugging...")
        # Fallback for offline environments if download fails
        class FakeDataset(torch.utils.data.Dataset):
            def __init__(self, length=1000):
                self.len = length
            def __len__(self): return self.len
            def __getitem__(self, idx):
                return torch.randn(1, 28, 28), torch.randint(0, 10, (1,)).item()
        
        train_ds = FakeDataset(1000)
        test_ds = FakeDataset(200)

    print(f"Loaded Train Dataset: {type(train_ds)}")
    if max_samples:
        indices = np.random.choice(len(train_ds), max_samples, replace=False)
        train_ds = Subset(train_ds, indices)
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
