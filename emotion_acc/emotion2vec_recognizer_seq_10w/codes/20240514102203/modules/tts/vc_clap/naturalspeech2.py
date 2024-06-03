import math
import torch
from torch import nn
from functools import partial
from torch.nn import functional as F
from torch import nn, einsum, Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat

import sys
sys.path.append('')

from modules.commons.layers import Embedding
from modules.tts.vc_clap.naturalspeech2_pytorch.attend import Attend
from modules.tts.vc_clap.speaker_encoder import ECAPA_TDNN_Encoder

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

# constants
mlist = nn.ModuleList

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(num, den):
    return (num % den) == 0

def identity(t, *args, **kwargs):
    return t

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

# tensor helpers

def pad_or_curtail_to_length(t, length):
    if t.shape[-1] == length:
        return t

    if t.shape[-1] > length:
        return t[..., :length]

    return F.pad(t, (0, length - t.shape[-1]))

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def generate_mask_from_repeats(repeats):
    repeats = repeats.int()
    device = repeats.device

    lengths = repeats.sum(dim = -1)
    max_length = lengths.amax().item()
    cumsum = repeats.cumsum(dim = -1)
    cumsum_exclusive = F.pad(cumsum, (1, -1), value = 0.)

    seq = torch.arange(max_length, device = device)
    seq = repeat(seq, '... j -> ... i j', i = repeats.shape[-1])

    cumsum = rearrange(cumsum, '... i -> ... i 1')
    cumsum_exclusive = rearrange(cumsum_exclusive, '... i -> ... i 1')

    lengths = rearrange(lengths, 'b -> b 1 1')
    mask = (seq < cumsum) & (seq >= cumsum_exclusive) & (seq < lengths)
    return mask

# sinusoidal positional embeds

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kernel_size, = self.kernel_size
        dilation, = self.dilation
        stride, = self.stride

        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        causal_padded_x = F.pad(x, (self.causal_padding, 0), value = 0.)
        return super().forward(causal_padded_x)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel = 3,
        groups = 8,
        dropout = 0.
    ):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, kernel, padding = kernel // 2)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kernel_size, = self.kernel_size
        dilation, = self.dilation
        stride, = self.stride

        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        causal_padded_x = F.pad(x, (self.causal_padding, 0), value = 0.)
        return super().forward(causal_padded_x)

class WavenetResBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dilation,
        kernel_size = 3,
        skip_conv = False,
        dim_cond_mult = None
    ):
        super().__init__()

        self.cond = exists(dim_cond_mult)
        self.to_time_cond = None

        if self.cond:
            self.to_time_cond = nn.Linear(dim * dim_cond_mult, dim * 2)

        self.conv = CausalConv1d(dim, dim, kernel_size, dilation = dilation)
        self.res_conv = CausalConv1d(dim, dim, 1)
        self.skip_conv = CausalConv1d(dim, dim, 1) if skip_conv else None

    def forward(self, x, t = None):

        if self.cond:
            assert exists(t)
            t = self.to_time_cond(t)
            t = rearrange(t, 'b c -> b c 1')
            t_gamma, t_beta = t.chunk(2, dim = -2)

        res = self.res_conv(x)

        x = self.conv(x)

        if self.cond:
            x = x * t_gamma + t_beta

        x = x.tanh() * x.sigmoid()

        x = x + res

        skip = None
        if exists(self.skip_conv):
            skip = self.skip_conv(x)

        return x, skip

class WavenetStack(nn.Module):
    def __init__(
        self,
        dim,
        *,
        layers,
        kernel_size = 3,
        has_skip = False,
        dim_cond_mult = None
    ):
        super().__init__()
        dilations = 2 ** torch.arange(layers)

        self.has_skip = has_skip
        self.blocks = mlist([])

        for dilation in dilations.tolist():
            block = WavenetResBlock(
                dim = dim,
                kernel_size = kernel_size,
                dilation = dilation,
                skip_conv = has_skip,
                dim_cond_mult = dim_cond_mult
            )

            self.blocks.append(block)

    def forward(self, x, t):
        residuals = []
        skips = []

        if isinstance(x, Tensor):
            x = (x,) * len(self.blocks)

        for block_input, block in zip(x, self.blocks):
            residual, skip = block(block_input, t)

            residuals.append(residual)
            skips.append(skip)

        if self.has_skip:
            return torch.stack(skips)

        return residuals

class Wavenet(nn.Module):
    def __init__(
        self,
        dim,
        *,
        stacks,
        layers,
        init_conv_kernel = 3,
        dim_cond_mult = None
    ):
        super().__init__()
        self.init_conv = CausalConv1d(dim, dim, init_conv_kernel)
        self.stacks = mlist([])

        for ind in range(stacks):
            is_last = ind == (stacks - 1)

            stack = WavenetStack(
                dim,
                layers = layers,
                dim_cond_mult = dim_cond_mult,
                has_skip = is_last
            )

            self.stacks.append(stack)

        self.final_conv = CausalConv1d(dim, dim, 1)

    def forward(self, x, t = None):

        x = self.init_conv(x)

        for stack in self.stacks:
            x = stack(x, t)

        return self.final_conv(x.sum(dim = 0))

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel,
        *,
        dropout = 0.,
        groups = 8,
        num_convs = 2
    ):
        super().__init__()

        blocks = []
        for ind in range(num_convs):
            is_first = ind == 0
            dim_in = dim if is_first else dim_out
            block = Block(
                dim_in,
                dim_out,
                kernel,
                groups = groups,
                dropout = dropout
            )
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        x = rearrange(x, 'b n c -> b c n')
        h = self.blocks(x)
        out = h + self.res_conv(x)
        return rearrange(out, 'b c n -> b n c')

def FeedForward(dim, mult = 4, causal_conv = False):
    dim_inner = int(dim * mult * 2 / 3)

    conv = None
    if causal_conv:
        conv = nn.Sequential(
            Rearrange('b n d -> b d n'),
            CausalConv1d(dim_inner, dim_inner, 3),
            Rearrange('b d n -> b n d'),
        )

    return Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        conv,
        nn.Linear(dim_inner, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_context = None,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        use_flash = False,
        cross_attn_include_queries = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attend = Attend(causal = causal, dropout = dropout, use_flash = use_flash)
        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(self, x, context = None, mask = None):
        h, has_context = self.heads, exists(context)

        context = default(context, x)

        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim = -2)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class RMSNorm_Rep(nn.Module):
    def __init__(self, dim, scale = True, dim_cond = None):
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None

        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None
        self.eps = 1e-12
    
    def normalize(self, input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12) -> Tensor:
        denom = input.norm(p, dim, keepdim=True).clamp_min(eps).expand_as(input)
        return input / denom

    def forward(self, x, cond = None):
        gamma = default(self.gamma, 1)
        # out = F.normalize(x, dim = -1) * self.scale * gamma
        out = self.normalize(x.clone(), dim = -1) * self.scale * gamma
        return out

class RMSNorm(nn.Module):
    def __init__(self, dim, scale = True, dim_cond = None):
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None

        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x, cond = None):
        gamma = default(self.gamma, 1)
        out = F.normalize(x, dim = -1) * self.scale * gamma

        if not self.cond:
            return out

        assert exists(cond)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim = -1)
        gamma, beta = map(lambda t: rearrange(t, 'b d -> b 1 d'), (gamma, beta))
        return out * gamma + beta

# transformer encoder

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        causal = False,
        dim_head = 64,
        heads = 8,
        use_flash = False,
        dropout = 0.,
        ff_mult = 4,
        final_norm = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                RMSNorm_Rep(dim),
                Attention(
                    dim,
                    causal = causal,
                    dim_head = dim_head,
                    heads = heads,
                    dropout = dropout,
                    use_flash = use_flash
                ),
                RMSNorm_Rep(dim),
                FeedForward(
                    dim,
                    mult = ff_mult
                )
            ]))

        self.norm = RMSNorm_Rep(dim) if final_norm else nn.Identity()

    def forward(self, x, mask = None):
        for attn_norm, attn, ff_norm, ff in self.layers:
            x = attn(attn_norm(x), mask = mask) + x
            x = ff(ff_norm(x)) + x

        return self.norm(x)

class ConditionableTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        ff_causal_conv = False,
        dim_cond_mult = None,
        cross_attn = False,
        use_flash = False
    ):
        super().__init__()
        self.dim = dim
        self.layers = mlist([])

        cond = exists(dim_cond_mult)

        maybe_adaptive_norm_kwargs = dict(scale = not cond, dim_cond = dim * dim_cond_mult) if cond else dict()
        rmsnorm = partial(RMSNorm, **maybe_adaptive_norm_kwargs)

        for _ in range(depth):
            self.layers.append(mlist([
                rmsnorm(dim),
                Attention(dim = dim, dim_head = dim_head, heads = heads, use_flash = use_flash),
                rmsnorm(dim) if cross_attn else None,
                Attention(dim = dim, dim_head = dim_head, heads = heads, use_flash = use_flash) if cross_attn else None,
                rmsnorm(dim),
                FeedForward(dim = dim, mult = ff_mult, causal_conv = ff_causal_conv)
            ]))

        self.to_pred = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim, bias = False)
        )

    def forward(
        self,
        x,
        times = None,
        context = None
    ):
        t = times

        for attn_norm, attn, cross_attn_norm, cross_attn, ff_norm, ff in self.layers:
            res = x
            x = attn_norm(x, cond = t)
            x = attn(x) + res

            if exists(cross_attn):
                assert exists(context)
                res = x
                x = cross_attn_norm(x, cond = t)
                x = cross_attn(x, context = context) + res

            res = x
            x = ff_norm(x, cond = t)
            x = ff(x) + res

        return self.to_pred(x)

class CondRelTransformerEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout=0.0,
                 window_size=4,
                 block_length=None,
                 prenet=True,
                 pre_ln=True,
                 ):

        super().__init__()

        self.n_vocab = n_vocab
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.prenet = prenet
        if n_vocab > 0:
            self.emb = Embedding(n_vocab+1, in_channels, padding_idx=n_vocab)
            
        self.proj = nn.Conv1d(in_channels, hidden_channels,kernel_size=1)
        self.pre = ConvReluNorm(hidden_channels, hidden_channels, hidden_channels,
                                    kernel_size=5, n_layers=2, p_dropout=0)
        self.encoder = ConditionableTransformer(
            dim = hidden_channels,
            depth = n_layers,
            dim_head = 64,
            heads = 8,
            ff_mult = 4,
            ff_causal_conv = True,
            dim_cond_mult = None,
            use_flash = False,
            cross_attn = True
        )

    def forward(self, x, context, x_mask=None):
        if self.n_vocab > 0:
            x = self.emb(x) * math.sqrt(self.in_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        # x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.proj(x) * x_mask
        x = self.pre(x, x_mask)
        x = x.transpose(1,2)  # [B,T,C]
        x_mask = x_mask.transpose(1,2)
        x = self.encoder(x, context=context) * x_mask
        return x

class TextConditionalRelTransformerEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout=0.0,
                 window_size=4,
                 block_length=None,
                 prenet=True,
                 pre_ln=True,
                 ):

        super().__init__()

        self.n_vocab = n_vocab
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.prenet = prenet
        if n_vocab > 0:
            self.emb = Embedding(n_vocab, hidden_channels, padding_idx=0)
    
        self.pre = ConvReluNorm(hidden_channels, hidden_channels, hidden_channels,
                                    kernel_size=5, n_layers=3, p_dropout=0)
        self.encoder = ConditionableTransformer(
            dim = hidden_channels,
            depth = n_layers,
            dim_head = 64,
            heads = 8,
            ff_mult = 4,
            ff_causal_conv = True,
            dim_cond_mult = None,
            use_flash = False,
            cross_attn = True
        )

    def forward(self, x,context, x_mask=None):
        if self.n_vocab > 0:
            x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        # x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x, x_mask)
        x = x.transpose(1,2)  # [B,T,C]
        x_mask = x_mask.transpose(1,2)
        x = self.encoder(x, context=context) * x_mask
        return x

class DurationPitchPredictor(nn.Module):
    def __init__(
        self,
        dim = 512,
        depth = 10,
        out_dim = 1,
        kernel_size = 3,
        dim_context = None,
        heads = 8,
        dim_head = 64,
        dropout = 0.2,
        use_resnet_block = True,
        num_convs_per_resnet_block = 2,
        num_convolutions_per_block = 3,
        use_flash_attn = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        conv_klass = ConvBlock if not use_resnet_block else partial(ResnetBlock, num_convs = num_convs_per_resnet_block)

        for _ in range(depth):
            layer = nn.ModuleList([
                nn.Sequential(*[
                    conv_klass(dim, dim, kernel_size) for _ in range(num_convolutions_per_block)
                ]),
                RMSNorm(dim),
                Attention(
                    dim,
                    dim_context = dim_context,
                    heads = heads,
                    dim_head = dim_head,
                    dropout = dropout,
                    use_flash = use_flash_attn,
                    cross_attn_include_queries = True
                )
            ])

            self.layers.append(layer)
        
        if out_dim == 1:        
            self.to_pred = nn.Sequential(
                nn.Linear(dim, out_dim),
                Rearrange('... 1 -> ...'),
                nn.ReLU()
            )
        else:
            self.to_pred = nn.Sequential(
                nn.Linear(dim, out_dim),
                nn.ReLU()
            )
    def forward(
        self,
        x,
        encoded_prompts,
        prompt_mask = None,
    ):
        for conv, norm, attn in self.layers:
            x = conv(x)
            x = attn(norm(x), encoded_prompts, mask = prompt_mask) + x

        return self.to_pred(x)

class NSDiffNet(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth=6,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        wavenet_layers = 8,
        wavenet_stacks = 4,
        dim_cond_mult = 4,
        use_flash_attn = False,
        dim_prompt = None,
        num_latents_m = 32,   # number of latents to be perceiver resampled ('q-k-v' with 'm' queries in the paper)
        cond_drop_prob = 0.,
        condition_on_prompt= False
    ):
        super().__init__()
        self.dim = dim

        self.input_projection = nn.Conv1d(80, dim, 1)

        # time condition

        dim_time = dim * dim_cond_mult

        self.to_time_cond = Sequential(
            LearnedSinusoidalPosEmb(dim),
            nn.Linear(dim + 1, dim_time),
            nn.SiLU()
        )

        # prompt condition

        self.cond_drop_prob = cond_drop_prob # for classifier free guidance
        self.condition_on_prompt = condition_on_prompt
        self.to_prompt_cond = None

        if self.condition_on_prompt:
            self.null_prompt_cond = nn.Parameter(torch.randn(dim_time))
            self.null_prompt_tokens = nn.Parameter(torch.randn(num_latents_m, dim))
            self.prompt_tokens_proj =  nn.Linear(dim_prompt, dim, bias=False)

            nn.init.normal_(self.null_prompt_cond, std = 0.02)
            nn.init.normal_(self.null_prompt_tokens, std = 0.02)

            self.to_prompt_cond = Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.Linear(dim_prompt, dim_time),
                nn.SiLU()
            )

        # aligned conditioning from aligner + duration module

        self.null_cond = None
        self.cond_to_model_dim = None

        if self.condition_on_prompt:
            self.cond_to_model_dim = nn.Conv1d(dim_prompt, dim, 1)
            self.null_cond = nn.Parameter(torch.zeros(dim, 1))

        # conditioning includes time and optionally prompt
        # t如果不和prompt concat的话，就不需要乘2
        dim_cond_mult = dim_cond_mult * (2 if condition_on_prompt else 1)

        # wavenet

        self.wavenet = Wavenet(
            dim = dim,
            stacks = wavenet_stacks,
            layers = wavenet_layers,
            dim_cond_mult = dim_cond_mult
        )

        # transformer

        self.transformer = ConditionableTransformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            ff_mult = ff_mult,
            ff_causal_conv = True,
            dim_cond_mult = dim_cond_mult,
            use_flash = use_flash_attn,
            cross_attn = condition_on_prompt
        )
        self.output_projection = nn.Conv1d(dim, 80, 1)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x,
        times,
        prompt = None,
        prompt_mask = None,
        cond = None,
        cond_drop_prob = None
    ):
        x = x[:, 0]
        b = x.shape[0]
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        x = self.input_projection(x).transpose(1,2)

        # prepare prompt condition
        # prob should remove going forward

        t = self.to_time_cond(times)
        c = None

        if exists(self.to_prompt_cond):
            assert exists(prompt)
            prompt_cond_drop_mask = prob_mask_like((b,), cond_drop_prob, self.device)

            prompt_cond = self.to_prompt_cond(prompt)

            prompt_cond = torch.where(
                rearrange(prompt_cond_drop_mask, 'b -> b 1'),
                self.null_prompt_cond,
                prompt_cond,
            )

            t = torch.cat((t, prompt_cond), dim = -1)

            prompt_tokens = self.prompt_tokens_proj(prompt)
            c = torch.where(
                rearrange(prompt_cond_drop_mask, 'b -> b 1 1'),
                self.null_prompt_tokens,
                prompt_tokens
            )

        # rearrange to channel first

        x = rearrange(x, 'b n d -> b d n')

        # sum aligned condition to input sequence

        if exists(self.cond_to_model_dim):
            assert exists(cond)
            cond = self.cond_to_model_dim(cond)

            cond_drop_mask = prob_mask_like((b,), cond_drop_prob, self.device)

            cond = torch.where(
                rearrange(cond_drop_mask, 'b -> b 1 1'),
                self.null_cond,
                cond
            )

            # for now, conform the condition to the length of the latent features

            cond = pad_or_curtail_to_length(cond, x.shape[-1])

            x = x + cond

        # main wavenet body

        # x = self.wavenet(x, t)
        # x = rearrange(x, 'b d n -> b n d')

        # x = self.transformer(x, t, context = c)
        # x = self.output_projection(x.transpose(1,2)) # [B,80,T]
        # return x[:, None, :, :]
        x = self.transformer(x.transpose(1,2), t, context = c)
        x = self.wavenet(x.transpose(1,2), t)

        x = self.output_projection(x) # [B,80,T]
        return x[:, None, :, :]

class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, attention_channels, kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)  # equals V and k in the paper

    def forward(self, x):

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x_in))
        # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)

class MultiStream_Spk_Encoder(nn.Module):
    def __init__(self,spk_bottleneck_dim=128,hidden_size=192, depth=2,num_streams=2):
        super().__init__()
        mid_dim = spk_bottleneck_dim
        out_dim = hidden_size
        self.mel_encoder = ECAPA_TDNN_Encoder(emb_dim=spk_bottleneck_dim)
        self.transformer_streams = nn.ModuleList()
        for _ in range(num_streams):
            self.transformer_streams.append(Transformer(
                dim = mid_dim,
                depth = depth,
                dropout = 0.0,
                use_flash=False,
            ))
        self.post_transformer = Transformer(
            dim = mid_dim,
            depth = depth,
            dropout = 0.0,
            use_flash = False
        )
        self.pooling = AttentiveStatsPool(mid_dim, attention_channels=32, global_context_att=False)
        self.bn = nn.BatchNorm1d(mid_dim * 2)
        self.linear = nn.Linear(mid_dim * 2, out_dim)


    def forward(self,mel):
        # mel  [B,T,80]
        mask = mel.eq(0.0).all(dim=-1)
        spk_embed = self.mel_encoder(mel).transpose(1,2)  # [B,T,C]
        out = spk_embed   
        for stream in self.transformer_streams:
            x = stream(x=spk_embed,mask=mask)
            out += x
        out = self.pooling(out.transpose(1,2))
        out = self.bn(out)
        out = self.linear(out)
        return out  # [B,C]

class WavLMSpeakerAdapter(nn.Module):
    def __init__(self,in_dim=768,spk_bottleneck_dim=64,hidden_size=192, depth=2,num_streams=2):
        super().__init__()
        mid_dim = spk_bottleneck_dim
        out_dim = hidden_size
        self.proj_in = nn.Linear(in_dim,mid_dim,bias=False)
        self.transformer_streams = nn.ModuleList()
        for _ in range(num_streams):
            self.transformer_streams.append(Transformer(
                dim = mid_dim,
                depth = depth,
                dropout = 0.0,
                use_flash=False,
            ))
        self.post_transformer = Transformer(
            dim = mid_dim,
            depth = depth,
            dropout = 0.0,
            use_flash = False
        )
        self.pooling = AttentiveStatsPool(mid_dim, attention_channels=32, global_context_att=False)
        self.bn = nn.BatchNorm1d(mid_dim * 2)
        self.linear = nn.Linear(mid_dim * 2, out_dim)


    def forward(self,spk_embed):
        # spk_embed  [B,T,768]
        mask = spk_embed.eq(0).all(dim=-1)
        spk_embed = self.proj_in(spk_embed)
        out = spk_embed
        for stream in self.transformer_streams:
            x = stream(x=spk_embed,mask=mask)
            out += x
        out = self.pooling(out.transpose(1,2))
        out = self.bn(out)
        out = self.linear(out)
        return out  # [B,C]


# net = WavLMSpeakerAdapter()
# x = torch.rand((16,126,768))
# y = net(x)
# print(y.shape)

# net = CondRelTransformerEncoder(100,
#                  16,
#                  192,
#                  192,
#                  768,
#                  8,
#                  4,
#                  5,
#                  p_dropout=0.0,
#                  prenet=True,
#                  pre_ln=True)
# input = torch.arange(0, 16*124).reshape(16, 124) % 101
# context = torch.randn((16,32,192))
# x_mask = (input<100).float()[:, :,None]
# x_mask = x_mask.transpose(1,2)
# print(input.shape,x_mask.shape)
# output = net(input,context,x_mask)
# print(output.shape)

# net = DurationPitchPredictor(dim=192,depth=5,out_dim=2)
# input = torch.randn((16,124,192))
# context = torch.randn((16,32,192))
# output = net(input,context)
# print(output.shape)


# net = NSDiffNet(
#     dim = 256,
#     depth = 6,
#     dim_prompt = 192,
#     cond_drop_prob = 0.25,
#     num_latents_m = 32,    ### style_token的数量     
#     condition_on_prompt = True)

# x = torch.rand((16,1,80,126))
# t = torch.randint(0, 10, (16,)).long()
# print(t.shape)
# prompt = torch.rand((16,32,192))   # [B,N,C]
# cond = torch.rand((16,192,126))  # [B,C,T]
# y = net(x=x,times=t,prompt=prompt,cond=cond)
# print(y.shape)