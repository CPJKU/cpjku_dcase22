import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import torch
from torch.distributions.beta import Beta
import numpy as np
from pytorch_lightning.callbacks import Callback
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from helpers import nessi

image_folder = "images"
os.makedirs(image_folder, exist_ok=True)


class MyStaticPostQuantizationCallback(Callback):
    def __init__(self, get_calibration_loader, calibration_batches=100):
        self.calibration_loader = get_calibration_loader()
        self.calibration_batches = calibration_batches

    def quantize_model(self, pl_module):
        print("*********** Before Quantization: ***********")
        if hasattr(pl_module, 'mel'):
            pl_module.mel.cpu()

        # get the shape of spectrograms
        sample = next(iter(self.calibration_loader))[0][0].unsqueeze(0)
        sample = sample[:, :, :sample.size(2) // 10]
        shape = pl_module.mel_forward(sample).size()

        # get original macs and params
        macc_orig, n_params_orig = nessi.get_model_size(pl_module.net, input_size=(1, shape[1], shape[2], shape[3]))
        print("macc_orig: ", macc_orig)
        print("n_params_orig: ", n_params_orig)

        # print size of model before quantization
        print_size_of_model(pl_module.net)
        pl_module.net.fuse_model()

        # get macs and params after fusing model
        macc, n_params = nessi.get_model_size(
            pl_module.net, input_size=(1, shape[1], shape[2], shape[3]))
        print("macc after fuse : ", macc)
        print("n_params after fuse: ", n_params)

        pl_module.net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(pl_module.net, inplace=True)

        pl_module.net.cpu()
        if hasattr(pl_module, 'mel'):
            pl_module.mel.cpu()
        for i, batch in enumerate(tqdm(self.calibration_loader, total=self.calibration_batches)):
            x, files, y, device_indices, cities, indices = batch
            # split to 1-second pieces
            x = rearrange(x, 'b c (slices t) -> (b slices) c t', slices=10)
            x = x.cpu()
            if hasattr(pl_module, 'mel'):
                x = pl_module.mel_forward(x)

            with torch.no_grad():
                pl_module.net(x)
            # stop after a certain number of calibration samples
            if i == self.calibration_batches:
                break

        torch.quantization.convert(pl_module.net, inplace=True)
        print("*********** After Quantization: ***********")
        return dict(macc_orig=macc_orig, n_params_orig=n_params_orig,
          macc_fuse=macc, n_params_fuse=n_params, model_size_bytes=print_size_of_model(pl_module.net))

    def on_test_start(self, trainer, pl_module):
        self.quantize_model(pl_module)



def mixstyle(x, p=0.5, alpha=0.1, eps=1e-6):
    if np.random.rand() > p:
        return x
    batch_size = x.size(0)

    # changed from dim=[2,3] to dim=[1,3] from channel-wise statistics to frequency-wise statistics
    f_mu = x.mean(dim=[1, 3], keepdim=True)
    f_var = x.var(dim=[1, 3], keepdim=True)

    f_sig = (f_var + eps).sqrt()  # compute instance standard deviation
    f_mu, f_sig = f_mu.detach(), f_sig.detach()  # block gradients
    x_normed = (x - f_mu) / f_sig  # normalize input
    lmda = Beta(alpha, alpha).sample((batch_size, 1, 1, 1)).to(x.device)  # sample instance-wise convex weights
    perm = torch.randperm(batch_size).to(x.device)  # generate shuffling indices
    f_mu_perm, f_sig_perm = f_mu[perm], f_sig[perm]  # shuffling
    mu_mix = f_mu * lmda + f_mu_perm * (1 - lmda)  # generate mixed mean
    sig_mix = f_sig * lmda + f_sig_perm * (1 - lmda)  # generate mixed standard deviation
    return x_normed * sig_mix + mu_mix  # denormalize input using the mixed statistics


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    model_size_bytes = os.path.getsize("temp.p")
    print('Size (MB):', model_size_bytes/1e6)
    os.remove('temp.p')
    return model_size_bytes


def mixup(size, alpha):
    rn_indices = torch.randperm(size)
    lambd = np.random.beta(alpha, alpha, size).astype(np.float32)
    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
    lam = torch.FloatTensor(lambd)
    # data = data * lam + data2 * (1 - lam)
    # targets = targets * lam + targets2 * (1 - lam)
    return rn_indices, lam


def spawn_get(seedseq, n_entropy, dtype):
    child = seedseq.spawn(1)[0]
    state = child.generate_state(n_entropy, dtype=np.uint32)

    if dtype == np.ndarray:
        return state
    elif dtype == int:
        state_as_int = 0
        for shift, s in enumerate(state):
            state_as_int = state_as_int + int((2 ** (32 * shift) * s))
        return state_as_int
    else:
        raise ValueError(f'not a valid dtype "{dtype}"')
