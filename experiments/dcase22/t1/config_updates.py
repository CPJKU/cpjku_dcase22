import scipy
from sacred.config_helpers import DynamicIngredient, CMD
import torch
import scipy.io.wavfile as wavfile
import os


def add_configs(ex):
    # reassembled 10-second recordings
    @ex.named_config
    def dcase22_reassembled():
            basedataset = DynamicIngredient("datasets.dcase22.dcase22t1.dataset",
                                            name="d22t1_r",
                                            base_dir="/share/rk6/shared/dcase22_reassembled/",
                                            audio_path="/share/rk6/shared/dcase22_reassembled/TAU-urban-acoustic-scenes-2022-mobile-development/",
                                            meta_csv="/share/rk6/shared/dcase22_reassembled/TAU-urban-acoustic-scenes-2022-mobile-development/meta.csv",
                                            train_files_csv="/share/rk6/shared/dcase22_reassembled/TAU-urban-acoustic-scenes-2022-mobile-development/evaluation_setup/fold1_train.csv",
                                            test_files_csv="/share/rk6/shared/dcase22_reassembled/TAU-urban-acoustic-scenes-2022-mobile-development/evaluation_setup/fold1_evaluate.csv"
                                            )

    # config for cp_resnet designed for the limited number of parameters and MACs
    @ex.named_config
    def cp_mini_resnet():
        models = {
            "net": DynamicIngredient("models.cp.cp_resnet.model_ing", n_blocks=(2, 1, 1))
        }
