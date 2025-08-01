from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any
from .nn_utils import *

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


class TimestepEmbedding(nn.Module):
    def __init__(self, time_embed_dim):
        super().__init__()
        self.time_embed_dim = time_embed_dim

    def forward(self, timesteps):
        # Time steps must be a 1D vector
        return positional_encoding(timesteps, self.time_embed_dim)


class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        if use_conv:
            self.ops = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.ops = nn.AvgPool2d(2)

    def forward(self, x):
        return self.ops(x)


class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1) if use_conv else None

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.conv is not None:
            x = self.conv(x)
        return x


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(TimestepBlock):
    """
    From OpenAI diffusion repository
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        in_channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        group_norm_num=8,
        SE_Net=False,
        SE_R=16,
        SE_Pos="pre",
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(group_norm_num, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels, 2 * out_channels if use_scale_shift_norm else out_channels
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(group_norm_num, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1)),
        )
        self.se = SE_Block(out_channels, SE_R) if SE_Net else nn.Identity()
        self.SE_Pos = SE_Pos

        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3 if use_conv else 1,
                padding=1 if use_conv else 0,
            )

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        # apply SE block
        if self.SE_Pos == "pre":
            h = self.se(h)

        out = self.skip_connection(x) + h

        if self.SE_Pos == "post":
            out = self.se(out)

        return out


class CrossAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        attn_heads,
        attn_dims,
        cond_dim=None,
        dropout=0.0,
        use_xformer=False,
        attn_dropout=0.0,
        residual=True,
    ):
        """In principle, we could use torch.nn.MultiheadedAttention but it appears the
        built in attention module has linear projections that require q,k,v to have the same initial embedding size.
        Here, the context may be passed in with a different number of channels than x so we manually write it.
        """
        super().__init__()
        self.residual = residual
        self.xformer = XFORMERS_IS_AVAILBLE and use_xformer
        if self.xformer:
            print("Setting up Multiheaded attention with XFORMERS library")
        self.attention_op: Optional[Any] = None

        inner_dim = attn_heads * attn_dims
        self.scale = attn_dims**-0.5
        self.attn_heads = attn_heads

        cond_dim = cond_dim if cond_dim is not None else query_dim
        self.layer_normx = nn.LayerNorm(query_dim)
        self.layer_normc = nn.LayerNorm(cond_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cond_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, inner_dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.final_layer_norm = nn.LayerNorm(
            query_dim
        )  # Adding a final LayerNorm after attention

        ## PE?

    def forward(self, x, cond=None):
        x_orig = x
        x = self.layer_normx(x)
        cond = x if cond is None else self.layer_normc(cond)

        h = self.attn_heads
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        if self.xformer:
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )
        else:
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
            attn = sim.softmax(dim=-1)
            attn = self.attn_dropout(attn)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)

        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        out = self.to_out(out)
        if self.residual:
            out = out + x_orig
        out = self.final_layer_norm(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        query_dim,
        attn_heads,
        attn_dims,
        cond_dim=None,
        dropout=0.0,
        use_checkpoint=True,
        use_xformer=False,
    ):
        super().__init__()
        self.sattn = CrossAttention(
            query_dim, attn_heads, attn_dims, dropout=dropout, use_xformer=use_xformer
        )
        self.xattn = CrossAttention(
            query_dim, attn_heads, attn_dims, cond_dim, dropout, use_xformer
        )
        self.norm = nn.LayerNorm(query_dim)
        self.ff = nn.Sequential(
            nn.Linear(query_dim, attn_dims * attn_heads),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dims * attn_heads, query_dim),
        )
        self.checkpoint = use_checkpoint

    def forward(self, x, cond=None):
        if cond is None:
            return checkpoint(self._forward, (x,), self.parameters(), self.checkpoint)
        else:
            return checkpoint(
                self._forward, (x, cond), self.parameters(), self.checkpoint
            )

    def _forward(self, x, cond=None):
        x = self.sattn(x) + x
        x = self.xattn(x, cond) + x
        x = self.ff(self.norm(x)) + x
        return x


class SpatialTransformer(CrossAttnBlock):
    def __init__(
        self,
        in_channels,
        attn_heads,
        attn_dims,
        depth,
        group_norm_num=8,
        cond_dim=None,
        dropout=0.0,
        use_checkpoint=True,
        use_linear=True,
        use_xformer=False,
    ):
        """Implements self-attention or cross-attention depending on the cond input"""
        super().__init__()
        self.norm = nn.GroupNorm(group_norm_num, in_channels)
        if cond_dim is not None:
            self.cond_norm = nn.GroupNorm(group_norm_num, cond_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                Transformer(
                    in_channels,
                    attn_heads,
                    attn_dims,
                    cond_dim,
                    dropout,
                    use_checkpoint,
                    use_xformer,
                )
                for _ in range(depth)
            ]
        )

        self.use_linear = use_linear
        self.proj_in = (
            nn.Linear(in_channels, in_channels)
            if use_linear
            else nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        )
        self.proj_out = (
            zero_module(nn.Linear(in_channels, in_channels))
            if use_linear
            else zero_module(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            )
        )

    def forward(self, x, cond=None):
        x_in = x
        _, _, h, w = x.shape
        x = self.norm(x)

        if not self.use_linear:
            self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.use_linear:
            x = self.proj_in(x)

        # Assuming cond is image like also but with different c then we need to reshape
        if cond is not None:
            cond = rearrange(self.cond_norm(cond), "b c h w -> b (h w) c")
            # The number of channels in the conditioning vector will be reamapped to attn_heads*attn_dims
            # downstream in the attention call.

        for layer in self.transformer_blocks:
            x = layer(x, cond=cond)

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)

        return x + x_in


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads, use_xformer=False, name=None):
        super().__init__()
        self.n_heads = n_heads
        self.xformer = XFORMERS_IS_AVAILBLE and use_xformer
        if self.xformer:
            print(f"Setting up Multiheaded attention with XFORMERS library: {name}")
        self.attention_op: Optional[Any] = None

    def forward(self, qkv, rescale=None):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)

        if self.xformer:
            a = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )
        else:
            scale = 1 / math.sqrt(math.sqrt(ch))
            weight = torch.einsum(
                "bct,bcs->bts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            if rescale is not None:
                weight = torch.softmax(weight.float() * rescale, dim=-1).type(
                    weight.dtype
                )
            else:
                weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
            a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)


class CrossAttentionBlock(CrossAttnBlock):
    """
    An attention block that allows spatial positions to attend to each other.
    Defaults to self-attention
    """

    def __init__(
        self,
        query_dim,
        attn_dims,
        cross_attention=False,
        cond_dim=None,
        attn_heads=1,
        use_checkpoint=True,
        group_norm_num=8,
        use_xformer=False,
        dropout=0.0,
        attn_dropout=0.0,
    ):
        super().__init__()
        self.xformer = XFORMERS_IS_AVAILBLE and use_xformer
        self.attention_op: Optional[Any] = None
        self.num_heads = attn_heads
        self.use_checkpoint = use_checkpoint
        self.scale = attn_dims**-0.5
        inner_dim = attn_heads * attn_dims

        if not cross_attention:
            cond_dim = query_dim

        self.normq = nn.GroupNorm(group_norm_num, query_dim)
        # self.normkv = nn.GroupNorm(group_norm_num, cond_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cond_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.final_norm = nn.GroupNorm(group_norm_num, query_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # # Sinusoidal positional encoding
        # if use_positional_encoding:
        #     pos = torch.arange(0, hw_length, 1)
        #     self.positional_encodings = nn.Parameter(
        #         positional_encoding(pos, in_channels, fq=10000)
        #         .transpose(-2, -1)[None]
        #         .to("cuda")
        #     )
        # else:
        #     self.positional_encodings = None

    def forward(self, x, cond=None):
        return checkpoint(
            self._forward, (x, cond), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, cond=None):
        # B C H W
        _, _, H, W = x.shape
        x_orig = x

        x = self.normq(x)
        cond = x if cond is None else cond
        # cond = self.normkv(cond)
        h = self.num_heads
        q = self.to_q(rearrange(x, "b c H W -> b (H W) c"))
        k = self.to_k(rearrange(cond, "b c H W -> b (H W) c"))
        v = self.to_v(rearrange(cond, "b c H W -> b (H W) c"))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        if self.xformer:
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )
        else:
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
            attn = sim.softmax(dim=-1)
            attn = self.attn_dropout(attn)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)

        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = rearrange(out, "b (H W) c -> b c H W", H=H, W=W)
        out = out + x_orig
        # return self.final_norm(out)
        return out


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        query_dim,
        attn_heads=1,
        use_checkpoint=True,
        group_norm_num=8,
        use_xformer=False,
        use_positional_encoding=False,
        max_seq_length=1024,
    ):
        super().__init__()
        self.channels = query_dim
        self.num_heads = attn_heads
        self.use_checkpoint = use_checkpoint
        self.use_positional_encoding = use_positional_encoding

        # In hindsight, I want to change this to layernorm instead
        self.norm = nn.GroupNorm(group_norm_num, query_dim)
        self.qkv = nn.Conv1d(query_dim, query_dim * 3, 1)
        self.attention = QKVAttentionLegacy(
            self.num_heads, use_xformer, name="Spatial Atten."
        )
        self.proj_out = zero_module(nn.Conv1d(query_dim, query_dim, 1))

        # Sinusoidal positional encoding
        if use_positional_encoding:
            self.positional_encodings = nn.Parameter(
                self.get_positional_encodings(max_seq_length, query_dim),
                requires_grad=False,
            )
        else:
            self.positional_encodings = None

    def get_positional_encodings(self, max_seq_length, d_model):
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.transpose(0, 1).unsqueeze(0)  # Shape: (1, d_model, max_seq_length)

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)  # b c h*w
        qkv = self.norm(x)
        if self.use_positional_encoding:
            seq_length = qkv.size(2)
            qkv = qkv + self.positional_encodings[:, :, :seq_length]

        qkv = self.qkv(qkv)
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

    # def _forward(self, x):
    #     b, c, *spatial = x.shape
    #     x = x.reshape(b, c, -1)  # (B, C, H*W)
    #     orig_seq_length = x.shape[-1]  # Store the original spatial sequence length

    #     # Ensure minimum sequence length for `xformers`
    #     min_seq_length = 8  # Adjust if needed
    #     if orig_seq_length < min_seq_length:
    #         pad_size = min_seq_length - orig_seq_length
    #         x = torch.cat(
    #             [
    #                 x,
    #                 torch.zeros(
    #                     (x.shape[0], x.shape[1], pad_size),
    #                     device=x.device,
    #                     dtype=x.dtype,
    #                 ),
    #             ],
    #             dim=-1,
    #         )

    #     qkv = self.norm(x)  # Normalize padded `x`

    #     if self.use_positional_encoding:
    #         seq_length = qkv.size(2)
    #         qkv = qkv + self.positional_encodings[:, :, :seq_length]

    #     qkv = self.qkv(qkv)  # Project QKV after padding
    #     h = self.attention(qkv)  # Apply attention

    #     # Trim back to original sequence length to maintain shape consistency
    #     h = h[:, :, :orig_seq_length]

    #     h = self.proj_out(h)
    #     return (x[:, :, :orig_seq_length] + h).reshape(
    #         b, c, *spatial
    #     )  # Trim final output


class ResBlockNoTime(nn.Module):
    def __init__(
        self,
        in_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_checkpoint=False,
        group_norm_num=8,
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            nn.GroupNorm(group_norm_num, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(group_norm_num, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1)),
        )

        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3 if use_conv else 1,
                padding=1 if use_conv else 0,
            )

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class SpectralAttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        dim,
        dim_head,
        heads,
        use_checkpoint=True,
        group_norm_num=8,
        use_xformer=False,
    ):
        super().__init__()
        # dim this case will be the h*w flattened image
        self.dim = dim
        self.num_heads = heads
        self.use_checkpoint = use_checkpoint
        self.norm = nn.GroupNorm(group_norm_num, dim)  # 1 = instance norm

        self.qkv = nn.Conv1d(dim, dim * 3, 1)
        self.attention = QKVAttentionLegacy(
            self.num_heads, use_xformer, name="Spectral Atten."
        )
        self.proj_out = zero_module(nn.Conv1d(dim, dim, 1))

        self.pos_emb = nn.Sequential(
            nn.Linear(dim, dim),
            GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)  # b c hw

        qkv = rearrange(F.normalize(x, p=2, dim=1), "b c hw -> b hw c")
        qkv = self.qkv(qkv)
        h = self.attention(qkv)
        h = self.proj_out(h)
        h = rearrange(h, "b hw c -> b c hw").reshape(b, c, *spatial)  # b c h w

        q = qkv[:, : self.dim, :]  # b hw c
        qemb = self.pos_emb(q.transpose(-1, -2)).reshape(b, c, *spatial)
        return h + qemb
