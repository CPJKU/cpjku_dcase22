import scipy
from sacred.config_helpers import DynamicIngredient, CMD
import torch
import scipy.io.wavfile as wavfile
import os


def add_configs(ex):
    # config for cp_resnet designed for the limited number of parameters and MACs
    @ex.named_config
    def cp_mini_resnet():
        models = {
            "net": DynamicIngredient("models.cp.cp_resnet.model_ing", n_blocks=(2, 1, 1))
        }
