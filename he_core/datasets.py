"""
Raw Dataset Loaders.
Bypasses torchvision dependence.
"""

import gzip
import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset

class RawMNIST(Dataset):
    """
    Reads MNIST from raw .gz files in root/raw/.
    """
    def __init__(self, root, train=True, transform_func=None):
        self.root = root
        self.transform_func = transform_func
        
        prefix = 'train' if train else 't10k'
        img_path = os.path.join(root, 'raw', f'{prefix}-images-idx3-ubyte.gz')
        lbl_path = os.path.join(root, 'raw', f'{prefix}-labels-idx1-ubyte.gz')
        
        print(f"Loading RawMNIST from {img_path}...")
        
        with gzip.open(img_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            buffer = f.read()
            # Copy to ensure writable/contiguous if needed, though frombuffer is usually readonly
            data = np.frombuffer(buffer, dtype=np.uint8).copy()
            data = data.reshape(num, 1, rows, cols) # (N, 1, 28, 28)
            
        with gzip.open(lbl_path, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            buffer = f.read()
            labels = np.frombuffer(buffer, dtype=np.uint8).copy()
            
        self.data = torch.from_numpy(data).float() / 255.0
        self.targets = torch.from_numpy(labels).long()
        
        print(f"Loaded {len(self.data)} samples.")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        img = self.data[index] # (1, 28, 28)
        target = self.targets[index]
        
        if self.transform_func:
            img = self.transform_func(img)
            
        return img, target
