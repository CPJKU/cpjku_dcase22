import torch
import numpy as np
import random

from helpers.utils import spawn_get


def worker_init_fn(wid):
    seed_sequence = np.random.SeedSequence(
        [torch.initial_seed(), wid]
    )

    to_seed = spawn_get(seed_sequence, 2, dtype=int)
    torch.random.manual_seed(to_seed)

    np_seed = spawn_get(seed_sequence, 2, dtype=np.ndarray)
    np.random.seed(np_seed)

    py_seed = spawn_get(seed_sequence, 2, dtype=int)
    random.seed(py_seed)
