import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

from vector_quantize_pytorch import LFQ

from einops import rearrange, repeat, pack, unpack

# helper

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def is_odd(n):
    return not divisible_by(n, 2)

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# autoencoder - only best variant here offered, with causal conv 3d

class CausalConv3d(Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size,
        pad_mode = 'reflect',
        **kwargs
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        assert len(kernel_size) == 3

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.get('dilation', 1)
        stride = kwargs.get('stride', 1)

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode = self.pad_mode)
        return self.conv(x)

class CausalConvTranspose1d(Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        time_kernel_size,
        *,
        stride,
        **kwargs
    ):
        super().__init__()
        self.upsample_factor = stride
        self.conv = nn.ConvTranspose1d(chan_in, chan_out, time_kernel_size, stride, **kwargs)

    def forward(self, x):
        assert x.ndim == 5
        x = rearrange(x, 'b c t h w -> b h w c t')
        x, ps = pack_one(x, '* c t')
        n = x.shape[-1]

        out = self.conv(x)
        out = out[..., :(n * self.upsample_factor)]

        out = unpack_one(out, ps, '* c t')
        out = rearrange(out ,'b h w c t -> b c t h w')
        return out

# main class

class MagViT2(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
