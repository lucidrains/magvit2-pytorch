import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

from einops import rearrange, repeat

# helper

def exists(v):
    return v is not None

# main class

class MagViT2(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
