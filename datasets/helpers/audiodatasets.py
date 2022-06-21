import os
from torch.utils.data import Dataset as TorchDataset
from os.path import expanduser
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

# Class creates or load cached dataset from a specific location
class FilesCachedDataset(TorchDataset):
    def __init__(self, get_dataset_func, dataset_name, ap_name, cache_path):
        """
        @param get_dataset_func: dataset function can be spectogram of basic
        @param dataset_name: name of the dataset to save the cached files
        @param ap_name: completes the name of the cached files
        @param cache_path: were data needs to be load from or saved to
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
        supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset, available_indices, global_norm, gl_norm):
        """
        @param dataset: dataset to load data from
        @param available_indices: available indices in case meta file restriction exist or randomly selected
        @param global_norm: csv file containing global norm for the dataset if doesnt exist, makes it
        @param gl_norm: boolian value indicating global norm is needed or not
        return: spectogram, file name, label of the input, device ad city
        """
        if available_indices is not None:
            self.available_indices = available_indices
        else:
            self.available_indices = np.arange(len(dataset))
        self.dataset = dataset
        self.gl_norm = gl_norm
        if self.gl_norm:
            first_run = True
            if not os.path.exists(global_norm):
                for i in tqdm(available_indices):
                    x, file, label, device, city = self.dataset[i]
                    if first_run:
                        y = np.zeros((len(available_indices), x.shape[1], x.shape[2]))
                    y[i, :, :] = x
                    first_run = 0
                self.mean = y.mean()
                self.std = y.std()
                d = {'mean': [self.mean], 'std': [self.std]}
                print('Mean and Std of the dataset:', self.mean, self.std)
                data = pd.DataFrame(data=d)
                with open(global_norm, 'w') as file:
                    data.to_csv(global_norm)
            else:
                data = pd.read_csv(global_norm)
                self.mean = data['mean'].to_numpy()
                self.std = data['std'].to_numpy()
                print('Mean and Std of the dataset:', self.mean, self.std)

    def __getitem__(self, index):
        x, file, label, device, city = self.dataset[self.available_indices[index]]
        if self.gl_norm:
            x = (x-self.mean[0])/(self.std[0]+1e-5)
        return x, file, label, device, city, self.available_indices[index]

    def __len__(self):
        return len(self.available_indices)


class PreprocessDataset(TorchDataset):
    """A bases preprocessing dataset representing a preprocessing step of a Dataset preprossessed on the fly.
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __init__(self, dataset, preprocessor):
        """
        @param dataset: dataset containing the files
        @param preprocessor: preprocess function
        #return preprocessed output for the dataset
        """
        self.dataset = dataset
        if not callable(preprocessor):
            print("preprocessor: ", preprocessor)
            raise ValueError('preprocessor should be callable')
        self.preprocessor = preprocessor

    def __getitem__(self, index):
        return self.preprocessor(self.dataset[index])

    def __len__(self):
        return len(self.dataset)
