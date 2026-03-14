import torch

def cross_entropy(inputs, targets):
    max_logits = inputs.max(dim=-1, keepdim=True).values
    shifted_inputs = inputs - max_logits
    denom = torch.logsumexp(shifted_inputs, dim=-1, keepdim=True)
    target = shifted_inputs[torch.arange(targets.shape[0]), targets]
    return torch.mean(denom-target)

