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
    def __init__(self, get_calibration_loader, calibration_batches=100, split_to_second=False):
        self.calibration_loader = get_calibration_loader()
        self.calibration_batches = calibration_batches
        self.split_to_second = split_to_second

    def quantize_model(self, pl_module):
        print("*********** Before Quantization: ***********")
        shape = [1, 256, 44]  # pl_module.mel_forward(sample).size()
        macc_orig, n_params_orig = nessi.get_model_size(pl_module.net, input_size=(1, shape[0], shape[1], shape[2]))
        print("macc_orig: ", macc_orig)
        print("n_params_orig: ", n_params_orig)
        # exit()
        print_size_of_model(pl_module.net)
        pl_module.net.cpu()
        if hasattr(pl_module, 'mel'):
            pl_module.mel.cpu()
        pl_module.net.fuse_model()

        macc, n_params = nessi.get_model_size(
            pl_module.net, input_size=(1, shape[0], shape[1], shape[2]))
        print("macc after fuse : ", macc)
        print("n_params after fuse: ", n_params)

        pl_module.net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(pl_module.net, inplace=True)
        pl_module.cpu()
        for i, batch in enumerate(tqdm(self.calibration_loader, total=self.calibration_batches)):
            x, files, y, device_indices, cities, indices = batch
            if self.split_to_second:
                x = rearrange(x, 'b c (slices t) -> (b slices) c t', slices=10)
                # take only one tenth of samples, since we want to have the same number of calibration samples
                # for the 1 and 10 second cases
                x = x[0:x.size(0):10]
            if hasattr(pl_module, 'mel'):
                x = pl_module.mel_forward(x)
            with torch.no_grad():
                pl_module.net(x.cpu())
            # stop after a certain number of calibration samples
            if i == self.calibration_batches:
                break

        torch.quantization.convert(pl_module.net, inplace=True)
        print("*********** After Quantization: ***********")
        return dict(macc_orig=macc_orig, n_params_orig=n_params_orig,
          macc_fuse=macc, n_params_fuse=n_params, model_size_bytes=print_size_of_model(pl_module.net))

    def on_test_start(self, trainer, pl_module):
        self.quantize_model(pl_module)

    #def on_predict_start(self, trainer, pl_module):
    #    self.quantize_model(pl_module)


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


def plot_spec(spec, title):
    if type(spec) == torch.Tensor:
        spec = spec.numpy()
    plt.figure()
    librosa.display.specshow(spec.squeeze())
    plt.colorbar()
    plt.title(title, fontsize=20)
    plt.savefig(os.path.join(image_folder, title + ".png"), dpi=100)


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
