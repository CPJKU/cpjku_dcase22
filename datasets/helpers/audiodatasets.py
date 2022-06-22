import os
from torch.utils.data import Dataset as TorchDataset
from os.path import expanduser
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd


# Class creates or loads cached dataset from a specific location
class FilesCachedDataset(TorchDataset):
    def __init__(self, get_dataset_func, dataset_name, ap_name, cache_path):
        """
        @param get_dataset_func: dataset function to create spectrograms/resampled audios
        @param dataset_name: name of the dataset to save the cached files
        @param ap_name: completes the name of the cached files
        @param cache_path: path to cache location
        """
        self.dataset = None

        def get_dataset():
            if self.dataset is None:
                self.dataset = get_dataset_func()
            return self.dataset

        self.get_dataset_func = get_dataset
        self.ap_name = ap_name
        cache_path = expanduser(cache_path)
        self.cache_path = os.path.join(cache_path, dataset_name, "files_cache", self.ap_name)

        original_umask = None
        try:
            original_umask = os.umask(0)
            os.makedirs(self.cache_path, exist_ok=True)
        finally:
            os.umask(original_umask)

    def __getitem__(self, index):
        cpath = os.path.join(self.cache_path, str(index) + ".pt")
        try:
            return torch.load(cpath)
        except FileNotFoundError:
            tup = self.get_dataset_func()[index]
            torch.save(tup, cpath)
            return tup

    def __len__(self):
        return len(self.get_dataset_func())


class SimpleSelectionDataset(TorchDataset):
    """A dataset that selects a subsample from a dataset based on a set of sample ids.
        Supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset, available_indices):
        """
        @param dataset: dataset to load data from
        @param available_indices: available indices in case meta file restriction exist or randomly selected
        return: x, file name, label, device, city, mapped index
        """
        if available_indices is not None:
            self.available_indices = available_indices
        else:
            self.available_indices = np.arange(len(dataset))
        self.dataset = dataset

    def __getitem__(self, index):
        x, file, label, device, city = self.dataset[self.available_indices[index]]
        return x, file, label, device, city, self.available_indices[index]

    def __len__(self):
        return len(self.available_indices)
