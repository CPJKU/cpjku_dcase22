import pandas as pd
import os
from sklearn import preprocessing
from torch.utils.data import Dataset as TorchDataset
import torch
import torchaudio.transforms as T
import numpy as np

from ba3l.ingredients.datasets import Dataset
from datasets.helpers.audiodatasets import FilesCachedDataset, SimpleSelectionDataset
from sacred.config import DynamicIngredient, CMD
from helpers import utils

dataset = Dataset('audiodataset')


@dataset.config
def default_config():
    name = 'd22t1_r'  # dataset name
    # processed audio content files cached in this location
    cache_root_path = "/share/rk6/shared/kofta_cached_datasets/"
    # base directory of the dataset as downloaded and reassembled
    base_dir = "/share/rk6/shared/dcase22_reassembled/"
    audio_path = os.path.join(base_dir, "TAU-urban-acoustic-scenes-2022-mobile-development")
    meta_csv = os.path.join(audio_path, "meta.csv")
    audio_processor = DynamicIngredient(path="datasets.helpers.audioprocessors.base.default_processor")
    process_func = CMD(".audio_processor.get_processor_default")
    train_files_csv = os.path.join(audio_path, "evaluation_setup", "fold1_train.csv")
    test_files_csv = os.path.join(audio_path, "evaluation_setup", "fold1_evaluate.csv")
    use_full_dev_dataset = 0
    subsample = 0
    # eval files
    eval_base_dir = "/share/rk6/shared/dcase22/"
    eval_files_csv = eval_base_dir + "TAU-urban-acoustic-scenes-2022-mobile-evaluation/meta.csv"
    eval_audio_path = eval_base_dir + "TAU-urban-acoustic-scenes-2022-mobile-evaluation/"


@dataset.config
def process_config():
    # audio processor capable of creating spectrograms -
    # in this example we only use it for resampling and
    # create the spectrograms on the fly in teachers_gang.py
    audio_processor = dict(sr=32000,
                           resample_only=True)


@dataset.command
class BasicDCASE22Dataset(TorchDataset):
    """
    Basic DCASE22 Dataset
    """

    def __init__(self, meta_csv):
        """
        @param meta_csv: meta csv file for the dataset
        return: name of the file, label, device and cities
        """
        df = pd.read_csv(meta_csv, sep="\t")
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(df[['scene_label']].values.reshape(-1))
        self.devices = le.fit_transform(df[['source_label']].values.reshape(-1))
        self.cities = le.fit_transform(df['identifier'].apply(lambda loc: loc.split("-")[0]).values.reshape(-1))
        self.files = df[['filename']].values.reshape(-1)
        self.df = df

    def __getitem__(self, index):
        return self.files[index], self.labels[index], self.devices[index], self.cities[index]

    def __len__(self):
        return len(self.files)


@dataset.command
class ContentDataset(TorchDataset):
    """
    gets the audio content from files using audioprocessor
    meta_csv: meta file containing index
    audio_path: audio path to the files
    process_function: function used to process raw audio
    return: x, file name, label, device, city
    """
    def __init__(self, meta_csv, audio_path, process_func):
        self.ds = BasicDCASE22Dataset(meta_csv)
        self.process_func = process_func
        self.audio_path = audio_path

    def __getitem__(self, index):
        file, label, device, city = self.ds[index]
        x = self.process_func(os.path.join(self.audio_path, file))
        return x, file, label, device, city

    def __len__(self):
        return len(self.ds)
    

@dataset.command
class EvalContentDataset(TorchDataset):
    """
    gets the audio content from files using audioprocessor
    meta_csv: meta file containing index
    audio_path: audio path to the files
    process_function: function used to process raw audio
    return: x, file name   ---> No further information available for evaluation
    """
    def __init__(self, meta_csv, audio_path, process_func):
        df = pd.read_csv(meta_csv, sep="\t")
        self.files = df[['filename']].values.reshape(-1)
        self.process_func = process_func
        self.audio_path = audio_path

    def __getitem__(self, index):
        file = self.files[index]
        x = self.process_func(os.path.join(self.audio_path, file))
        return x, file

    def __len__(self):
        return len(self.files)


# command to retrieve dataset from cached files
@dataset.command
def get_file_cached_dataset(name, audio_processor, cache_root_path):
    print("get_file_cached_dataset::", name, audio_processor['identifier'], "sr=", audio_processor['sr'],
          cache_root_path)
    ds = FilesCachedDataset(ContentDataset, name, audio_processor['identifier'], cache_root_path)
    return ds


# commands to create the datasets for training, testing and evaluation
@dataset.command
def get_training_set_raw():
    ds = get_base_training_set_raw()
    return ds


@dataset.command
def get_base_training_set_raw(meta_csv,  train_files_csv, subsample, use_full_dev_dataset):
    train_files = pd.read_csv(train_files_csv, sep='\t')['filename'].values.reshape(-1)
    meta = pd.read_csv(meta_csv, sep="\t")
    train_indices = list(meta[meta['filename'].isin(train_files)].index)
    if subsample:
        train_indices = np.random.choice(train_indices, size=subsample, replace=False)
    if use_full_dev_dataset:
        train_indices = np.arange(len(meta['filename']))
    ds = SimpleSelectionDataset(get_file_cached_dataset(), train_indices)
    return ds


@dataset.command
def get_test_set_raw():
    ds = get_base_test_set_raw()
    return ds


@dataset.command
def get_base_test_set_raw(meta_csv,  test_files_csv, subsample):
    test_files = pd.read_csv(test_files_csv, sep='\t')['filename'].values.reshape(-1)
    meta = pd.read_csv(meta_csv, sep="\t")
    test_indices = list(meta[meta['filename'].isin(test_files)].index)
    if subsample:
        test_indices = np.random.choice(test_indices, size=subsample, replace=False)
    ds = SimpleSelectionDataset(get_file_cached_dataset(), test_indices)
    return ds


@dataset.command
def get_eval_set_raw():
    ds = get_base_eval_set_raw()
    return ds


# evaluation
@dataset.command
def get_base_eval_set_raw(eval_files_csv, eval_audio_path, name, audio_processor, cache_root_path):
    name = name + "_eval22test"
    print("get_base_eval_set 22::", name, audio_processor['identifier'], "sr=", audio_processor['sr'],
          cache_root_path)

    def get_dataset_func():
        return EvalContentDataset(eval_files_csv, eval_audio_path)
    ds = FilesCachedDataset(get_dataset_func, name, audio_processor['identifier'], cache_root_path)
    return ds
