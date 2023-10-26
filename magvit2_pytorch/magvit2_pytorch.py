import copy
from pathlib import Path
from math import log2, ceil, sqrt
from functools import wraps, partial

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.autograd import grad as torch_grad

import torchvision
from torchvision.models import VGG16_Weights

from collections import namedtuple

from vector_quantize_pytorch import LFQ

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.typing import Union, Tuple, Optional

from magvit2_pytorch.attend import Attend
from magvit2_pytorch.version import __version__

from kornia.filters import filter3d

import pickle

# helper

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def identity(t, *args, **kwargs):
    return t

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

# tensor helpers

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pick_video_frame(video, frame_indices):
    batch, device = video.shape[0], video.device
    video = rearrange(video, 'b c f ... -> b f c ...')
    batch_indices = torch.arange(batch, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')
    images = video[batch_indices, frame_indices]
    images = rearrange(images, 'b 1 c ... -> b c ...')
    return images

# gan related

def gradient_penalty(images, output):
    batch_size = images.shape[0]

    gradients = torch_grad(
        outputs = output,
        inputs = images,
        grad_outputs = torch.ones(output.size(), device = images.device),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

@beartype
def grad_layer_wrt_loss(
    loss: Tensor,
    layer: nn.Parameter
):
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach()

# helper decorators

def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, 'vgg')
        if has_vgg:
            vgg = self.vgg
            delattr(self, 'vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self.vgg = vgg

        return out
    return inner

# helper classes

def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    return nn.Sequential(*modules)

class Residual(Module):
    @beartype
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# token shifting

class TokenShift(Module):
    @beartype
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x, x_shift = x.chunk(2, dim = 1)
        x_shift = pad_at_dim(x_shift, (1, -1), dim = 2) # shift time dimension
        x = torch.cat((x, x_shift), dim = 1)
        return self.fn(x, **kwargs)

# rmsnorm

class RMSNorm(Module):
    def __init__(
        self,
        dim,
        channel_first = False,
        images = False
    ):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))

    def forward(self, x):
        return F.normalize(x, dim = (1 if self.channel_first else -1)) * self.scale * self.gamma

# attention

class Attention(Module):
    def __init__(
        self,
        *,
        dim,
        causal = False,
        dim_head = 32,
        heads = 8,
        flash = False,
        dropout = 0.,
        num_memory_kv = 4
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.to_qkv = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        )

        assert num_memory_kv > 0
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_memory_kv, dim_head))

        self.attend = Attend(
            causal = causal,
            dropout = dropout,
            flash = flash
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False)
        )

    def forward(self, x, mask = None):
        q, k, v = self.to_qkv(x)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = q.shape[0]), self.mem_kv)
        k = torch.cat((mk, k), dim = -2)
        v = torch.cat((mv, v), dim = -2)

        out = self.attend(q, k, v, mask = mask)
        return self.to_out(out)

class LinearAttention(Module):
    """
    using the specific linear attention proposed in https://arxiv.org/abs/2106.09681
    """

    def __init__(
        self,
        *,
        dim,
        dim_head = 32,
        heads = 8,
        flash = False,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.to_qkv = Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h d n', qkv = 3, h = heads)
        )

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.attend = Attend(
            scale = 1.,
            causal = False,
            dropout = dropout,
            flash = flash
        )

        self.to_out = Sequential(
            Rearrange('b h d n -> b n (h d)'),
            nn.Linear(dim_inner, dim)
        )

    def forward(self, x):
        q, k, v = self.to_qkv(x)

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        out = self.attend(q, k, v)

        return self.to_out(out)

class LinearSpaceAttention(LinearAttention):
    def forward(self, x, *args, **kwargs):
        x = rearrange(x, 'b c ... h w -> b ... h w c')
        x, batch_ps = pack_one(x, '* h w c')
        x, seq_ps = pack_one(x, 'b * c')

        x = super().forward(x, *args, **kwargs)

        x = unpack_one(x, seq_ps, 'b * c')
        x = unpack_one(x, batch_ps, '* h w c')
        return rearrange(x, 'b ... h w c -> b c ... h w')

class SpaceAttention(Attention):
    def forward(self, x, *args, **kwargs):
        x = rearrange(x, 'b c t h w -> b t h w c')
        x, batch_ps = pack_one(x, '* h w c')
        x, seq_ps = pack_one(x, 'b * c')

        x = super().forward(x, *args, **kwargs)

        x = unpack_one(x, seq_ps, 'b * c')
        x = unpack_one(x, batch_ps, '* h w c')
        return rearrange(x, 'b t h w c -> b c t h w')

class TimeAttention(Attention):
    def forward(self, x, *args, **kwargs):
        x = rearrange(x, 'b c t h w -> b h w t c')
        x, batch_ps = pack_one(x, '* t c')

        x = super().forward(x, *args, **kwargs)

        x = unpack_one(x, batch_ps, '* t c')
        return rearrange(x, 'b h w t c -> b c t h w')

def FeedForward(dim, mult = 4, images = False):
    conv_klass = nn.Conv2d if images else nn.Conv3d
    norm_klass = partial(RMSNorm, channel_first = True, images = images)
    dim_inner = dim * mult

    return Sequential(
        norm_klass(dim, channel_first = True, images = images),
        conv_klass(dim, dim_inner, 1),
        nn.GELU(),
        norm_klass(dim_inner, channel_first = True),
        conv_klass(dim_inner, dim, 1)
    )

# discriminator with anti-aliased downsampling (blurpool Zhang et al.)

class Blur(Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(
        self,
        x,
        space_only = False,
        time_only = False
    ):
        assert not (space_only and time_only)

        f = self.f

        if space_only:
            f = einsum('i, j -> i j', f, f)
            f = rearrange(f, '... -> 1 1 ...')
        elif time_only:
            f = rearrange(f, 'f -> 1 f 1 1')
        else:
            f = einsum('i, j, k -> i j k', f, f, f)
            f = rearrange(f, '... -> 1 ...')

        is_images = x.ndim == 4

        if is_images:
            x = rearrange(x, 'b c h w -> b c 1 h w')

        out = filter3d(x, f, normalized = True)

        if is_images:
            out = rearrange(out, 'b c 1 h w -> b c h w')

        return out

class DiscriminatorBlock(Module):
    def __init__(
        self,
        input_channels,
        filters,
        downsample = True,
        antialiased_downsample = True
    ):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding = 1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding = 1),
            leaky_relu()
        )

        self.maybe_blur = Blur() if antialiased_downsample else None

        self.downsample = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
            nn.Conv2d(filters * 4, filters, 1)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)

        x = self.net(x)

        if exists(self.downsample):
            if exists(self.maybe_blur):
                x = self.maybe_blur(x, space_only = True)

            x = self.downsample(x)

        x = (x + res) * (2 ** -0.5)
        return x

class Discriminator(Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels = 3,
        max_dim = 512,
        attn_heads = 8,
        attn_dim_head = 32,
        attn_flash = True,
        ff_mult = 4,
        antialiased_downsample = True
    ):
        super().__init__()
        image_size = pair(image_size)
        min_image_resolution = min(image_size)

        num_layers = int(log2(min_image_resolution) - 2)

        blocks = []

        layer_dims = [channels] + [(dim * 4) * (2 ** i) for i in range(num_layers + 1)]
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        blocks = []
        attn_blocks = []

        image_resolution = min_image_resolution

        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            block = DiscriminatorBlock(
                in_chan,
                out_chan,
                downsample = is_not_last,
                antialiased_downsample = antialiased_downsample
            )

            attn_block = Sequential(
                Residual(LinearSpaceAttention(
                    dim = out_chan,
                    heads = attn_heads,
                    dim_head = attn_dim_head,
                    flash = attn_flash
                )),
                Residual(FeedForward(
                    dim = out_chan,
                    mult = ff_mult,
                    images = True
                ))
            )

            blocks.append(ModuleList([
                block,
                attn_block
            ]))

            image_resolution //= 2

        self.blocks = ModuleList(blocks)

        dim_last = layer_dims[-1]

        downsample_factor = 2 ** num_layers
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        self.to_logits = Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding = 1),
            leaky_relu(),
            Rearrange('b ... -> b (...)'),
            nn.Linear(latent_dim, 1),
            Rearrange('b 1 -> b')
        )

    def forward(self, x):

        for block, attn_block in self.blocks:
            x = block(x)
            x = attn_block(x)

        return self.to_logits(x)

# modulatable conv from Karras et al. Stylegan2
# for conditioning on latents

class Conv3DMod(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        spatial_kernel,
        time_kernel,
        causal = True,
        dim_out = None,
        demod = True,
        eps = 1e-8,
        pad_mode = 'constant'
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.eps = eps

        assert is_odd(spatial_kernel) and is_odd(time_kernel)

        self.spatial_kernel = spatial_kernel
        self.time_kernel = time_kernel

        time_padding = (time_kernel - 1, 0) if causal else ((time_kernel // 2,) * 2)

        self.pad_mode = pad_mode
        self.padding = (*((spatial_kernel // 2,) * 4), *time_padding)
        self.weights = nn.Parameter(torch.randn((dim_out, dim, time_kernel, spatial_kernel, spatial_kernel)))

        self.demod = demod

        nn.init.kaiming_normal_(self.weights, a = 0, mode = 'fan_in', nonlinearity = 'selu')

    @beartype
    def forward(
        self,
        fmap,
        cond: Tensor
    ):
        """
        notation

        b - batch
        n - convs
        o - output
        i - input
        k - kernel
        """

        b = fmap.shape[0]

        # prepare weights for modulation

        weights = self.weights

        # do the modulation, demodulation, as done in stylegan2

        cond = rearrange(cond, 'b i -> b 1 i 1 1 1')

        weights = weights * (cond + 1)

        if self.demod:
            inv_norm = reduce(weights ** 2, 'b o i k0 k1 k2 -> b o 1 1 1 1', 'sum').clamp(min = self.eps).rsqrt()
            weights = weights * inv_norm

        fmap = rearrange(fmap, 'b c t h w -> 1 (b c) t h w')

        weights = rearrange(weights, 'b o ... -> (b o) ...')

        fmap = F.pad(fmap, self.padding, mode = self.pad_mode)
        fmap = F.conv3d(fmap, weights, groups = b)

        return rearrange(fmap, '1 (b o) ... -> b o ...', b = b)

# strided conv downsamples

class SpatialDownsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        kernel_size = 3,
        antialias = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.maybe_blur = Blur() if antialias else identity
        self.conv = nn.Conv2d(dim, dim_out, kernel_size, stride = 2, padding = kernel_size // 2)

    def forward(self, x):
        x = self.maybe_blur(x, space_only = True)

        x = rearrange(x, 'b c t h w -> b t c h w')
        x, ps = pack_one(x, '* c h w')

        out = self.conv(x)

        out = unpack_one(out, ps, '* c h w')
        out = rearrange(out, 'b t c h w -> b c t h w')
        return out

class TimeDownsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        kernel_size = 3,
        antialias = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.maybe_blur = Blur() if antialias else identity
        self.conv = nn.Conv1d(dim, dim_out, kernel_size, stride = 2, padding = kernel_size // 2)

    def forward(self, x):
        x = self.maybe_blur(x, time_only = True)

        x = rearrange(x, 'b c t h w -> b h w c t')
        x, ps = pack_one(x, '* c t')

        out = self.conv(x)

        out = unpack_one(out, ps, '* c t')
        out = rearrange(out, 'b h w c t -> b c t h w')
        return out

# depth to space upsamples

class SpatialUpsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange('b (c p1 p2) h w -> b c (h p1) (w p2)', p1 = 2, p2 = 2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b t c h w')
        x, ps = pack_one(x, '* c h w')

        out = self.net(x)

        out = unpack_one(out, ps, '* c h w')
        out = rearrange(out, 'b t c h w -> b c t h w')
        return out

class TimeUpsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv1d(dim, dim_out * 2, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange('b (c p) t -> b c (t p)', p = 2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, t = conv.weight.shape
        conv_weight = torch.empty(o // 2, i, t)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 2) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b h w c t')
        x, ps = pack_one(x, '* c t')

        out = self.net(x)

        out = unpack_one(out, ps, '* c t')
        out = rearrange(out, 'b h w c t -> b c t h w')
        return out

# autoencoder - only best variant here offered, with causal conv 3d

class CausalConv3d(Module):
    @beartype
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        pad_mode = 'reflect',
        **kwargs
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop('dilation', 1)
        stride = kwargs.pop('stride', 1)

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride = stride, dilation = dilation, **kwargs)

    def forward(self, x):
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else 'constant'

        x = F.pad(x, self.time_causal_padding, mode = pad_mode)
        return self.conv(x)

@beartype
def ResidualUnit(
    dim,
    kernel_size: Union[int, Tuple[int, int, int]],
    pad_mode: str = 'reflect'
):
    return Residual(Sequential(
        CausalConv3d(dim, dim, kernel_size, pad_mode = pad_mode),
        nn.ELU(),
        nn.Conv3d(dim, dim, 1),
        nn.ELU()
    ))

@beartype
class ResidualUnitMod(Module):
    def __init__(
        self,
        dim,
        kernel_size: Union[int, Tuple[int, int, int]],
        *,
        dim_cond,
        pad_mode: str = 'constant',
        demod = True
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        assert height_kernel_size == width_kernel_size

        self.to_cond = nn.Linear(dim_cond, dim)

        self.conv = Conv3DMod(
            dim = dim,
            spatial_kernel = height_kernel_size,
            time_kernel = time_kernel_size,
            causal = True,
            demod = demod,
            pad_mode = pad_mode
        )

        self.conv_out = nn.Conv3d(dim, dim, 1)

    @beartype
    def forward(
        self,
        x,
        cond: Tensor,
    ):
        res = x
        cond = self.to_cond(cond)

        x = self.conv(x, cond = cond)
        x = F.elu(x)
        x = self.conv_out(x)
        x = F.elu(x)
        return x + res

class CausalConvTranspose3d(Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        *,
        time_stride,
        **kwargs
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        self.upsample_factor = time_stride

        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        stride = (time_stride, 1, 1)
        padding = (0, height_pad, width_pad)

        self.conv = nn.ConvTranspose3d(chan_in, chan_out, kernel_size, stride, padding = padding, **kwargs)

    def forward(self, x):
        assert x.ndim == 5
        t = x.shape[2]

        out = self.conv(x)

        out = out[..., :(t * self.upsample_factor), :, :]
        return out

# video tokenizer class

LossBreakdown = namedtuple('LossBreakdown', [
    'recon_loss',
    'lfq_aux_loss',
    'lfq_per_sample_entropy_loss',
    'lfq_batch_entropy_loss',
    'lfq_commitment_loss',
    'perceptual_loss',
    'gen_loss'
])

DiscrLossBreakdown = namedtuple('DiscrLossBreakdown', [
    'discr_loss',
    'gradient_penalty'
])

class VideoTokenizer(Module):
    @beartype
    def __init__(
        self,
        *,
        image_size,
        layers: Tuple[Union[str, Tuple[str, int]], ...] = (
            'residual',
            'residual',
            'residual'
        ),
        residual_conv_kernel_size = 3,
        num_codebooks = 1,
        codebook_size = 8192,
        channels = 3,
        init_dim = 64,
        dim_cond = None,
        dim_cond_expansion_factor = 4.,
        input_conv_kernel_size: Tuple[int, int, int] = (7, 7, 7),
        output_conv_kernel_size: Tuple[int, int, int] = (3, 3, 3),
        pad_mode: str = 'reflect',
        lfq_entropy_loss_weight = 0.1,
        lfq_commitment_loss_weight = 1.,
        lfq_diversity_gamma = 1.,
        lfq_aux_loss_weight = 1.,
        attn_dim_head = 32,
        attn_heads = 8,
        attn_dropout = 0.,
        vgg: Optional[Module] = None,
        vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,
        perceptual_loss_weight = 1.,
        antialiased_downsample = True,
        discr_kwargs: Optional[dict] = None,
        use_gan = True,
        adversarial_loss_weight = 1.,
        grad_penalty_loss_weight = 10.,
        flash_attn = True
    ):
        super().__init__()

        # for autosaving the config

        _locals = locals()
        _locals.pop('self', None)
        _locals.pop('__class__', None)
        self._configs = pickle.dumps(_locals)

        # image size

        self.image_size = image_size

        # encoder

        self.conv_in = CausalConv3d(channels, init_dim, input_conv_kernel_size, pad_mode = pad_mode)

        self.encoder_layers = ModuleList([])
        self.decoder_layers = ModuleList([])

        self.conv_out = CausalConv3d(init_dim, channels, output_conv_kernel_size, pad_mode = pad_mode)

        dim = init_dim
        time_downsample_factor = 1
        has_cond = False

        for layer_def in layers:
            layer_type, *layer_params = cast_tuple(layer_def)

            if layer_type == 'residual':
                encoder_layer = ResidualUnit(dim, residual_conv_kernel_size)
                decoder_layer = ResidualUnit(dim, residual_conv_kernel_size)
                dim_out = dim

            elif layer_type == 'cond_residual':
                assert exists(dim_cond), 'dim_cond must be passed into VideoTokenizer, if tokenizer is to be conditioned'

                has_cond = True

                encoder_layer = ResidualUnitMod(dim, residual_conv_kernel_size, dim_cond = int(dim_cond * dim_cond_expansion_factor))
                decoder_layer = ResidualUnitMod(dim, residual_conv_kernel_size, dim_cond = int(dim_cond * dim_cond_expansion_factor))

            elif layer_type == 'compress_space':
                dim_out, = layer_params
                encoder_layer = SpatialDownsample2x(dim, dim_out, antialias = antialiased_downsample)
                decoder_layer = SpatialUpsample2x(dim_out, dim)

            elif layer_type == 'compress_time':
                dim_out, = layer_params
                encoder_layer = TimeDownsample2x(dim, dim_out, antialias = antialiased_downsample)
                decoder_layer = TimeUpsample2x(dim_out, dim)

                time_downsample_factor *= 2

            elif layer_type == 'attend_space':
                attn_kwargs = dict(
                    dim = dim,
                    dim_head = attn_dim_head,
                    heads = attn_heads,
                    dropout = attn_dropout,
                    flash = flash_attn
                )

                encoder_layer = Sequential(
                    Residual(SpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim))
                )

                decoder_layer = Sequential(
                    Residual(SpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim))
                )

            elif layer_type == 'linear_attend_space':
                attn_kwargs = dict(
                    dim = dim,
                    dim_head = attn_dim_head,
                    heads = attn_heads,
                    dropout = attn_dropout,
                    flash = flash_attn
                )

                encoder_layer = Sequential(
                    Residual(LinearSpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim))
                )

                decoder_layer = Sequential(
                    Residual(LinearSpaceAttention(**attn_kwargs)),
                    Residual(FeedForward(dim))
                )

            elif layer_type == 'attend_time':
                attn_kwargs = dict(
                    dim = dim,
                    dim_head = attn_dim_head,
                    heads = attn_heads,
                    dropout = attn_dropout,
                    causal = True,
                    flash = flash_attn
                )

                encoder_layer = Sequential(
                    Residual(TokenShift(TimeAttention(**attn_kwargs))),
                    Residual(TokenShift(FeedForward(dim)))
                )

                decoder_layer = Sequential(
                    Residual(TokenShift(TimeAttention(**attn_kwargs))),
                    Residual(TokenShift(FeedForward(dim)))
                )

            else:
                raise ValueError(f'unknown layer type {layer_type}')

            self.encoder_layers.append(encoder_layer)
            self.decoder_layers.insert(0, decoder_layer)

            dim = dim_out

        self.time_downsample_factor = time_downsample_factor
        self.time_padding = time_downsample_factor - 1

        # use a MLP stem for conditioning, if needed

        self.has_cond = has_cond
        self.encoder_cond_in = nn.Identity()
        self.decoder_cond_in = nn.Identity()

        if has_cond:
            self.dim_cond = dim_cond

            self.encoder_cond_in = Sequential(
                nn.Linear(dim_cond, int(dim_cond * dim_cond_expansion_factor)),
                nn.SiLU()
            )

            self.decoder_cond_in = Sequential(
                nn.Linear(dim_cond, int(dim_cond * dim_cond_expansion_factor)),
                nn.SiLU()
            )

        # lookup free quantizer(s) - multiple codebooks is possible
        # each codebook will get its own entropy regularization

        self.quantizers = LFQ(
            dim = dim,
            codebook_size = codebook_size,
            num_codebooks = num_codebooks,
            entropy_loss_weight = lfq_entropy_loss_weight,
            commitment_loss_weight = lfq_commitment_loss_weight,
            diversity_gamma = lfq_diversity_gamma
        )

        self.lfq_aux_loss_weight = lfq_aux_loss_weight

        # dummy loss

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # perceptual loss related

        use_vgg = channels == 3 and perceptual_loss_weight > 0.

        self.vgg = None
        self.perceptual_loss_weight = perceptual_loss_weight

        if use_vgg:
            if not exists(vgg):
                vgg = torchvision.models.vgg16(
                    weights = vgg_weights
                )

                vgg.classifier = Sequential(*vgg.classifier[:-2])

            self.vgg = vgg

        self.use_vgg = use_vgg

        # discriminator

        discr_kwargs = default(discr_kwargs, dict(
            dim = dim,
            image_size = image_size,
            max_dim = 512
        ))

        self.discr = Discriminator(**discr_kwargs)

        self.adversarial_loss_weight = adversarial_loss_weight
        self.grad_penalty_loss_weight = grad_penalty_loss_weight

        self.has_gan = use_gan and adversarial_loss_weight > 0.

    @classmethod
    def init_and_load_from(cls, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

        config = pickle.loads(pkg['config'])
        tokenizer = cls(**config)
        tokenizer.load(path, strict = strict)
        return tokenizer

    def parameters(self):
        return [
            *self.conv_in.parameters(),
            *self.conv_out.parameters(),
            *self.encoder_layers.parameters(),
            *self.decoder_layers.parameters(),
            *self.encoder_cond_in.parameters(),
            *self.decoder_cond_in.parameters(),
        ]

    def discr_parameters(self):
        return self.discr.parameters()

    def copy_for_eval(self):
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())

        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy.vgg

        vae_copy.eval()
        return vae_copy.to(device)

    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists(), f'{str(path)} already exists'

        pkg = dict(
            model_state_dict = self.state_dict(),
            version = __version__,
            config = self._configs
        )

        torch.save(pkg, str(path))

    def load(self, path, strict = True):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path))
        state_dict = pkg.get('model_state_dict')
        version = pkg.get('version')

        assert exists(state_dict)

        if exists(version):
            print(f'loading checkpointed tokenizer from version {version}')

        self.load_state_dict(state_dict, strict = strict)

    @beartype
    def encode(
        self,
        video: Tensor,
        quantize = False,
        cond: Optional[Tensor] = None
    ):
        # conditioning, if needed

        assert (not self.has_cond) or exists(cond), '`cond` must be passed into tokenizer forward method since conditionable layers were specified'

        if exists(cond):
            assert cond.shape == (video.shape[0], self.dim_cond)

            cond = self.encoder_cond_in(cond)
            cond_kwargs = dict(cond = cond)

        # initial conv

        x = self.conv_in(video)

        # encoder layers

        for fn in self.encoder_layers:

            layer_kwargs = dict()
            if isinstance(fn, (ResidualUnitMod,)):
                layer_kwargs = cond_kwargs

            x = fn(x, **layer_kwargs)

        maybe_quantize = identity if not quantize else self.quantizers

        return maybe_quantize(x)

    @beartype
    def decode_from_code_indices(
        self,
        codes: Tensor,
        cond: Optional[Tensor] = None
    ):
        quantized = self.quantizers.indices_to_codes(codes)
        out = self.decode(quantized, cond = cond)
        return out[:, :, self.time_padding:]

    @beartype
    def decode(
        self,
        quantized: Tensor,
        cond: Optional[Tensor] = None
    ):
        batch = quantized.shape[0]

        # conditioning, if needed

        assert (not self.has_cond) or exists(cond), '`cond` must be passed into tokenizer forward method since conditionable layers were specified'

        if exists(cond):
            assert cond.shape == (batch, self.dim_cond)

            cond = self.decoder_cond_in(cond)
            cond_kwargs = dict(cond = cond)

        # decoder layers

        x = quantized

        for fn in self.decoder_layers:

            layer_kwargs = dict()
            if isinstance(fn, (ResidualUnitMod,)):
                layer_kwargs = cond_kwargs

            x = fn(x, **layer_kwargs)

        # to pixels

        return self.conv_out(x)

    @beartype
    def forward(
        self,
        video_or_images: Tensor,
        cond: Optional[Tensor] = None,
        return_loss = False,
        return_codes = False,
        return_recon = False,
        return_discr_loss = False,
        apply_gradient_penalty = True
    ):
        assert (return_loss + return_codes + return_discr_loss) <= 1
        assert video_or_images.ndim in {4, 5}

        assert video_or_images.shape[-2:] == (self.image_size, self.image_size)

        # accept images for image pretraining (curriculum learning from images to video)

        if video_or_images.ndim == 4:
            video = rearrange(video_or_images, 'b c ... -> b c 1 ...')
        else:
            video = video_or_images

        batch, frames = video.shape[0], video.shape[2]

        assert divisible_by(frames - 1, self.time_downsample_factor), f'number of frames {frames} minus the first frame ({frames - 1}) must be divisible by the total downsample factor across time {self.time_downsample_factor}'

        # pad the time, accounting for total time downsample factor, so that images can be trained independently

        padded_video = pad_at_dim(video, (self.time_padding, 0), value = 0., dim = 2)

        # encoder

        x = self.encode(padded_video, cond = cond)

        # lookup free quantization

        (quantized, codes, aux_losses), lfq_loss_breakdown = self.quantizers(x, return_loss_breakdown = True)

        if return_codes and not return_recon:
            return codes

        # decoder

        padded_recon_video = self.decode(quantized, cond = cond)

        recon_video = padded_recon_video[:, :, self.time_padding:]

        if return_codes:
            return recon_video, codes

        # reconstruction loss

        if not (return_loss or return_discr_loss):
            return recon_video

        recon_loss = F.mse_loss(video, recon_video)

        # gan discriminator loss

        if return_discr_loss:
            assert self.has_gan
            assert exists(self.discr)

            frame_indices = torch.randn((batch, frames)).topk(1, dim = -1).indices

            real = pick_video_frame(video, frame_indices)

            if apply_gradient_penalty:
                real = real.requires_grad_()

            fake = pick_video_frame(recon_video, frame_indices)

            real_logits = self.discr(real)
            fake_logits = self.discr(fake)

            discr_loss = hinge_discr_loss(fake_logits, real_logits)

            if apply_gradient_penalty:
                gradient_penalty_loss = gradient_penalty(real, real_logits)
            else:
                gradient_penalty_loss = self.zero

            total_loss = discr_loss + gradient_penalty_loss * self.grad_penalty_loss_weight

            return total_loss, DiscrLossBreakdown(discr_loss, gradient_penalty_loss)

        # perceptual loss

        if self.use_vgg:
            frame_indices = torch.randn((batch, frames)).topk(1, dim = -1).indices

            input_vgg_input = pick_video_frame(video, frame_indices)
            recon_vgg_input = pick_video_frame(recon_video, frame_indices)

            input_vgg_feats = self.vgg(input_vgg_input)
            recon_vgg_feats = self.vgg(recon_vgg_input)

            perceptual_loss = F.mse_loss(input_vgg_feats, recon_vgg_feats)
        else:
            perceptual_loss = self.zero

        if self.has_gan:
            frame_indices = torch.randn((batch, frames)).topk(1, dim = -1).indices
            recon_video_frames = pick_video_frame(recon_video, frame_indices)

            fake_logits = self.discr(recon_video_frames)
            gen_loss = hinge_gen_loss(fake_logits)

            last_dec_layer = self.conv_out.conv.weight

            norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
            norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

            adaptive_weight = norm_grad_wrt_perceptual_loss / norm_grad_wrt_gen_loss.clamp(min = 1e-5)
            adaptive_weight.clamp_(max = 1e4)
        else:
            gen_loss = self.zero
            adaptive_weight = 0.

        total_loss = recon_loss \
            + aux_losses * self.lfq_aux_loss_weight \
            + perceptual_loss * self.perceptual_loss_weight \
            + gen_loss * adaptive_weight * self.adversarial_loss_weight

        return total_loss, LossBreakdown(recon_loss, aux_losses, *lfq_loss_breakdown, perceptual_loss, gen_loss)

# main class

class MagViT2(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
