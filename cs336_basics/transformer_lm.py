import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        self.W = nn.Parameter(torch.empty((self.out_features, self.in_features), device=device, dtype=dtype) )
        nn.init.trunc_normal_(self.W)

    def forward(self, x):
        return x @ self.W.T

