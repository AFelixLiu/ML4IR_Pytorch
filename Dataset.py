"""
@author AFelixLiu
@date 2024 12月 18
"""

import torch
from torch.utils.data import Dataset


class PAHDataset(Dataset):
    def __init__(self, smiles, labels):
        self.smiles = smiles
        self.labels = labels

    def __getitem__(self, index):
        item = {
            'smi': torch.tensor(self.smiles[index], dtype=torch.float),
            'label': torch.tensor(self.labels[index], dtype=torch.float)
        }
        return item

    def __len__(self):
        return len(self.smiles)
