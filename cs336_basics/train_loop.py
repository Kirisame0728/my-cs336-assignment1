import numpy as np
import torch


def data_loading(x ,batch_size, context_length, device):
    start_idx = np.random.randint(0, len(x)-context_length, size=batch_size).reshape(-1, 1)
    offsets = np.arange(context_length).reshape(1, -1)
    idx = start_idx + offsets
    inputs = x[idx]
    targets = x[idx + 1]
    inputs = torch.from_numpy(inputs).long().to(device)
    targets = torch.from_numpy(targets).long().to(device)

    return inputs, targets

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]