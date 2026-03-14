import torch
from typing import Optional
from collections.abc import Callable, Iterable
import math

def cross_entropy(inputs, targets):
    max_logits = inputs.max(dim=-1, keepdim=True).values
    shifted_inputs = inputs - max_logits
    denom = torch.logsumexp(shifted_inputs, dim=-1, keepdim=True)
    target = shifted_inputs[torch.arange(targets.shape[0]), targets]
    return torch.mean(denom-target)

class AdamWOpt(torch.optim.Optimizer):
    def __init__(self, params, weight_decay, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m, v = state["m"], state["v"]
                beta1, beta2 = betas

                # Update state count
                state["step"] += 1
                t = state["step"]

                grad = p.grad.data
                # m = beta1 * m + (1 - beta1) * grad
                # v = beta2 * v + (1 - beta2) * grad ** 2
                # state["m"] = m
                # state["v"] = v
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -=  alpha_t * (m / (torch.sqrt(v) + eps))
                p.data -= lr * weight_decay * p.data
        return loss

def learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return t * alpha_max / T_w
    elif t >= T_w and t <= T_c:
        return alpha_min + 0.5 * (1 + math.cos(math.pi * (t-T_w) / (T_c - T_w))) * (alpha_max - alpha_min)
    else:
        return alpha_min

def gradient_clipping(parameters, max_norm):
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in parameters if p.grad is not None))
    if norm > max_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad[:] *= max_norm / (1e-6 + norm)




