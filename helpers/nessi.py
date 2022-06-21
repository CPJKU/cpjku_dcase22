
# Parameter Calculator imported from :
# https://github.com/AlbertoAncilotto/NeSsi/blob/main/nessi.py

import numpy as np
import torchinfo

MAX_MACC = 30e6  # 30M MACC
MAX_PARAMS = 128e3  # 128K params ### 128e4 in the github document


def get_torch_size(model, input_size):
    model_profile = torchinfo.summary(model, input_size=input_size)
    return model_profile.total_mult_adds, model_profile.total_params


def get_model_size(model, input_size=None):
    macc, params = get_torch_size(model, input_size)
    validate(macc, params)
    return macc, params


def validate(macc, params):
    print('Model statistics:')
    print('MACC:\t \t %.3f' % (macc / 1e6), 'M')
    print('Params:\t \t %.3f' % (params / 1e3), 'K\n')
    if macc > MAX_MACC:
        print('[Warning] Multiply accumulate count', macc, 'is more than the allowed maximum of', int(MAX_MACC))
    if params > MAX_PARAMS:
        print('[Warning] parameter count', params, 'is more than the allowed maximum of', int(MAX_PARAMS))