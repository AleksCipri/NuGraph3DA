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




#New Domain Adaptation functions which handle two datasets
class CombinedDataset(Dataset):
    """
    Dataset that pairs two datasets together and returns corresponding items.
    Iteration stops at the length of the shorter dataset.
    """
    
    def __init__(self, datasetA, datasetB):
        """
        Initialize the combined dataset.

        Args:
            datasetA (Dataset): First dataset.
            datasetB (Dataset): Second dataset.
        """
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __len__(self):
        """
        Return the length of the combined dataset.

        Returns:
            int: Minimum length of the two datasets.
        """
        return min(len(self.datasetA), len(self.datasetB))

    def __getitem__(self, idx):
        """
        Retrieve a paired item from both datasets at the given index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing (item_from_datasetA, item_from_datasetB)
        """
        dataA = self.datasetA[idx]
        dataB = self.datasetB[idx]
        return dataA, dataB 


class CombinedDatasetCycle(Dataset):
    """
    Dataset that pairs two datasets together and cycles over the shorter dataset.
    Iteration continues for the length of the longer dataset by wrapping around the shorter one.
    """
    def __init__(self, datasetA, datasetB):
        """
        Initialize the combined cyclic dataset.

        Args:
            datasetA (Dataset): First dataset.
            datasetB (Dataset): Second dataset.
        """
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __len__(self):
        """
        Return the length of the combined cyclic dataset.

        Returns:
            int: Maximum length of the two datasets.
        """
        return max(len(self.datasetA), len(self.datasetB))

    def __getitem__(self, idx):
        """
        Retrieve a paired item from both datasets at the given index,
        cycling over the shorter dataset if necessary.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing (item_from_datasetA, item_from_datasetB)
        """
        dataA = self.datasetA[idx % len(self.datasetA)]
        dataB = self.datasetB[idx % len(self.datasetB)]
        return dataA, dataB
