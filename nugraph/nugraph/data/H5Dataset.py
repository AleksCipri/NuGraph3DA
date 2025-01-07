from typing import Callable, Optional

import h5py
from pynuml import io

import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.data import HeteroData

class H5Dataset(Dataset):
    def __init__(self,
                 filename: str,
                 samples: list[str],
                 transform: Optional[Callable] = None):
        super().__init__(transform=transform)
        self._interface = io.H5Interface(h5py.File(filename))
        self._samples = samples

    def len(self) -> int:
        return len(self._samples)

    def get(self, idx: int) -> 'pyg.data.HeteroData':
        return self._interface.load_heterodata(self._samples[idx])


# CombinedDataset for pairing elements from DatasetA and DatasetB (total length is equal to the shorter dataset)
class CombinedDataset(Dataset):
    def __init__(self, datasetA, datasetB):
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __len__(self):
        # Use the minimum length of the two datasets
        return min(len(self.datasetA), len(self.datasetB))

    def __getitem__(self, idx):
        dataA = self.datasetA[idx]
        dataB = self.datasetB[idx]
        return dataA, dataB  # Return both data objects

# Combined Dataset that cycles over the shorter dataset
class CombinedDatasetCycle(Dataset):
    def __init__(self, datasetA, datasetB):
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __len__(self):
        return max(len(self.datasetA), len(self.datasetB))

    def __getitem__(self, idx):
        dataA = self.datasetA[idx % len(self.datasetA)]
        dataB = self.datasetB[idx % len(self.datasetB)]
        return dataA, dataB
