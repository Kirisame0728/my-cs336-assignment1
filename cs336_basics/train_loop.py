import numpy as np
import torch
from numpy import dtype


def data_loading(x ,batch_size, context_length, device):
    start_idx = np.random.randint(0, len(x)-context_length, size=batch_size).reshape(-1, 1)
    offsets = np.arange(context_length).reshape(1, -1)
    idx = start_idx + offsets
    inputs = x[idx]
    targets = x[idx + 1]
    inputs = torch.from_numpy(inputs).long().to(device)
    targets = torch.from_numpy(targets).long().to(device)

    return inputs, targets
