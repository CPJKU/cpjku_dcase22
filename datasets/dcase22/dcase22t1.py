import pandas as pd
import os
from sklearn import preprocessing
from torch.utils.data import Dataset as TorchDataset
import torch
import torchaudio.transforms as T
import numpy as np

from ba3l.ingredients.datasets import Dataset
from datasets.helpers.audiodatasets import FilesCachedDataset, SimpleSelectionDataset, PreprocessDataset
from sacred.config import DynamicIngredient, CMD
from helpers import utils

LMODE = os.environ.get("LMODE", False)

dataset = Dataset('audiodataset')


@dataset.config
def default_config():
    name = 'd22t1'  # dataset name
    # cached spectograms loaded or saved in this location
    cache_root_path = "/share/rk6/shared/kofta_cached_datasets/"

    # base directory of the dataset as downloaded
    base_dir = "/share/rk6/shared/dcase22/"
    audio_path = os.path.join(base_dir, "TAU-urban-acoustic-scenes-2022-mobile-development")
    meta_csv = os.path.join(audio_path, "meta.csv")
    # path to the global norm file, if doesnt exist it will be saved here
    global_norm = os.path.join(audio_path, "global_norm.csv")
    audio_processor = DynamicIngredient(path="datasets.helpers.audioprocessors.base.default_processor")
    process_func = CMD(".audio_processor.get_processor_default")
    train_files_csv = os.path.join(audio_path, "evaluation_setup", "fold1_train.csv")
    test_files_csv = os.path.join(audio_path, "evaluation_setup", "fold1_evaluate.csv")
    time_shift = 0
    freq_shift = 0
    time_mask = 0
    freq_mask = 0
    subsample = 0
    gl_norm = 0
    use_full_dev_dataset = 0
    # eval files
    eval_base_dir = "/share/rk6/shared/dcase22/"
    if LMODE:
        eval_base_dir = "/system/user/publicdata/CP/DCASE/dcase22/"
    eval_files_csv = eval_base_dir + "TAU-urban-acoustic-scenes-2022-mobile-evaluation/meta.csv"
    eval_audio_path = eval_base_dir + "TAU-urban-acoustic-scenes-2022-mobile-evaluation/"



if LMODE:
    @dataset.config
    def LMODE_default_config():
        cache_root_path = "/system/user/publicdata/CP/DCASE/cached_datasets/"

@dataset.config
def process_config():
    audio_processor = dict(n_fft=2048,
                           sr=22050,
                           mono=True,
                           log_spec=False,
                           n_mels=256,
                           hop_length=512)

@dataset.command
class BasicDCASE22Dataset(TorchDataset):
    """
    Basic DCASE22 Dataset
    """

    def __init__(self, meta_csv):
        """
        @param meta_csv: meta csv file for the dataset
        return: name of the file label device and cities from the file name
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
class SpectrogramDataset(TorchDataset):
    """
    gets the spectrogram from files using audioprocessor
    meta_csv: meta file containing index
    audio_path: audio path to the files
    process_function: function used to process raw audio
    return: x(spectograms) , file name, label, device, city
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
class EvalSpectrogramDataset(TorchDataset):
    """
    gets the spectrogram from files using audioprocessor
    meta_csv: meta file containing index
    audio_path: audio path to the files
    process_function: function used to process raw audio
    return: x(spectograms) , file name,  *During evalutation label is not available
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
    ds = FilesCachedDataset(SpectrogramDataset, name, audio_processor['identifier'], cache_root_path)
    return ds

@dataset.command
def get_base_training_set(meta_csv,  train_files_csv, global_norm, gl_norm, subsample_train=0):
    train_files = pd.read_csv(train_files_csv, sep='\t')['filename'].values.reshape(-1)
    meta = pd.read_csv(meta_csv, sep="\t")
    train_indices = list(meta[meta['filename'].isin(train_files)].index)
    if subsample_train:
        train_indices = np.random.choice(train_indices, size=subsample_train, replace=False)
    ds = SimpleSelectionDataset(get_file_cached_dataset(), train_indices, global_norm, gl_norm,)
    return ds


@dataset.command
def get_base_test_set(meta_csv,  test_files_csv, global_norm, gl_norm,):
    test_files = pd.read_csv(test_files_csv, sep='\t')['filename'].values.reshape(-1)
    meta = pd.read_csv(meta_csv, sep="\t")
    test_indices = list(meta[meta['filename'].isin(test_files)].index)
    ds = SimpleSelectionDataset(get_file_cached_dataset(), test_indices, global_norm, gl_norm)
    return ds

# command to roll the x
def get_roll_func(axis=2, shift=None, shift_range=12):
    def roll_func(b):
        x, file, label, device, city, index = b
        x = torch.as_tensor(x)
        sf = shift
        if shift is None:
            sf = int(np.random.randint(-shift_range, shift_range))
        return x.roll(sf, axis), file, label, device, city, index

    return roll_func


def get_mask_func(axis=2, mask_size=10):
    if axis == 2:
        masking = T.TimeMasking(time_mask_param=mask_size)
    else:
        masking = T.FrequencyMasking(freq_mask_param=mask_size)

    def mask_func(b):
        x, file, label, device, city, index = b
        x = torch.as_tensor(x)
        # using two small masks, instead of one large mask
        return masking(masking(x)), file, label, device, city, index

    return mask_func

# Data augmentation command.
@dataset.command
def get_training_set(time_shift, freq_shift, time_mask, freq_mask):
    ds = get_base_training_set()
    # save random spectrogram as image before and after preprocessing
    # time and frequency rolling
    if time_shift > 0:
        print("******* Time shifting (shift range: {}) ********".format(time_shift))
        ds = PreprocessDataset(ds, get_roll_func(axis=2, shift_range=time_shift))
    if freq_shift > 0:
        print("******* Freq shifting (shift range: {}) ********".format(freq_shift))
        ds = PreprocessDataset(ds, get_roll_func(axis=1, shift_range=freq_shift))
    # time and frequency masking
    print(time_mask)
    if time_mask > 0:
        print("******* Time masking (max mask size: {}) ********".format(time_mask))
        ds = PreprocessDataset(ds, get_mask_func(axis=2, mask_size=time_mask))
    if freq_mask > 0:
        print("******* Frequency masking (max mask size: {}) ********".format(freq_mask))
        ds = PreprocessDataset(ds, get_mask_func(axis=1, mask_size=freq_mask))
    return ds


@dataset.command
def get_base_training_set_raw(meta_csv,  train_files_csv, subsample, use_full_dev_dataset, global_norm, gl_norm):
    train_files = pd.read_csv(train_files_csv, sep='\t')['filename'].values.reshape(-1)
    meta = pd.read_csv(meta_csv, sep="\t")
    train_indices = list(meta[meta['filename'].isin(train_files)].index)
    if subsample:
        train_indices = np.random.choice(train_indices, size=subsample, replace=False)
    if use_full_dev_dataset:
        train_indices = np.arange(len(meta['filename']))
    ds = SimpleSelectionDataset(get_file_cached_dataset(), train_indices, global_norm, gl_norm)
    return ds


@dataset.command
def get_base_test_set_raw(meta_csv,  test_files_csv, subsample, global_norm, gl_norm):
    test_files = pd.read_csv(test_files_csv, sep='\t')['filename'].values.reshape(-1)
    meta = pd.read_csv(meta_csv, sep="\t")
    test_indices = list(meta[meta['filename'].isin(test_files)].index)
    if subsample:
        test_indices = np.random.choice(test_indices, size=subsample, replace=False)
    ds = SimpleSelectionDataset(get_file_cached_dataset(), test_indices, global_norm, gl_norm)
    return ds


@dataset.command
def get_training_set_raw(time_shift):
    ds = get_base_training_set_raw()
    # time rolling
    if time_shift > 0:
        print("******* Time shifting (shift range: {}) ********".format(time_shift))
        ds = PreprocessDataset(ds, get_roll_func(axis=1, shift_range=time_shift))
    return ds



# gets randomly selected data from SimpleSelectionDataset
@dataset.command
def get_base_development_set_raw(subsample, global_norm, gl_norm):
    ds = get_file_cached_dataset()
    subsample_indices = None
    if subsample:
        subsample_indices = np.random.choice(np.arange(len(ds)), size=subsample, replace=False)
        ds = SimpleSelectionDataset(ds, subsample_indices, global_norm, gl_norm)
    return ds, subsample_indices


@dataset.command
def get_development_set_raw(global_norm, gl_norm):
    ds, subsample_indices = get_base_development_set_raw()
    ds = SimpleSelectionDataset(ds, subsample_indices, global_norm, gl_norm)
    return ds


# evaluation
@dataset.command
def get_base_eval_set_raw(eval_files_csv, eval_audio_path, name, audio_processor, cache_root_path):
    name = name+"_eval22test"
    print("get_base_eval_set 22::", name, audio_processor['identifier'], "sr=", audio_processor['sr'],
          cache_root_path)
    def get_dataset_func():
        return EvalSpectrogramDataset(eval_files_csv, eval_audio_path)
    ds = FilesCachedDataset(get_dataset_func, name, audio_processor['identifier'], cache_root_path)
    return ds


@dataset.command
def get_test_set():
    ds = get_base_test_set()
    return ds

@dataset.command
def get_eval_set_raw():
    ds = get_base_eval_set_raw()
    return ds

@dataset.command
def get_test_set_raw():
    ds = get_base_test_set_raw()
    return ds



