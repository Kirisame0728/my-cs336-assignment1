import torch
from torch import nn
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty((self.out_features, self.in_features), device=device, dtype=dtype) )
        nn.init.trunc_normal_(self.W)
        self.init_parameters()

    def init_parameters(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return x @ self.W.T

class Embedding(nn.Module):
    def __init__(self,num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_matrix = nn.Parameter(torch.empty((self.num_embeddings, self.embedding_dim), device=device, dtype=dtype))
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.trunc_normal_(self.embedding_matrix, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gain

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = nn.Parameter(torch.empty((self.d_ff, self.d_model), device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty((self.d_model, self.d_ff), device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty((self.d_ff, self.d_model), device=device, dtype=dtype))
        self.init_parameters()

    def init_parameters(self):
        std_1 = math.sqrt(2.0 / (self.d_model + self.d_ff))
        std_2 = math.sqrt(2.0 / (self.d_ff + self.d_model))
        std_3 = math.sqrt(2.0 / (self.d_model + self.d_ff))

        nn.init.trunc_normal_(self.W1, mean=0.0, std=std_1, a=-3 * std_1, b=3 * std_1)
        nn.init.trunc_normal_(self.W2, mean=0.0, std=std_2, a=-3 * std_2, b=3 * std_2)
        nn.init.trunc_normal_(self.W3, mean=0.0, std=std_3, a=-3 * std_3, b=3 * std_3)

    def forward(self, x):
        SiLU = torch.sigmoid(x @ self.W1.T) * (x @ self.W1.T)
        GLU = SiLU * (x @ self.W3.T)
        return GLU @ self.W2.T

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float,d_k: int,max_seq_len: int,device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        pair_idx = torch.arange(self.d_k // 2, device=device, dtype=torch.float32)
        angle = 1.0 / pow(self.theta, 2.0*(pair_idx)/self.d_k)
        positions = torch.arange(self.max_seq_len, device=device)
        thetas = positions[:, None] * angle[None, :]
        self.register_buffer("cos", torch.cos(thetas), persistent=False)
        self.register_buffer("sin", torch.sin(thetas), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor)-> torch.Tensor:
        cosines = self.cos[token_positions]
        sines = self.sin[token_positions]
        x_odd = x[..., ::2]
        x_even = x[..., 1::2]

        rot_odd = x_odd * cosines - x_even * sines
        rot_even = x_odd * sines + x_even * cosines

        out = torch.empty_like(x)
        out[..., ::2] = rot_odd
        out[..., 1::2] = rot_even

        return out


def softmax(v, i):
    shift_c = v.max(dim=i, keepdim=True).values
    v_shift = v - shift_c
    softmax_denom = torch.sum(torch.exp(v_shift), dim=i, keepdim=True)
    return torch.exp(v_shift) / softmax_denom

def scaled_dot_product_attention(queries, keys, values, mask=None):
    self_attn = queries @ keys.transpose(-2, -1) / math.sqrt(queries.shape[-1])
    if mask is not None:
        self_attn =  self_attn.masked_fill(~mask, float("-inf"))
    return softmax(self_attn, -1) @ values

