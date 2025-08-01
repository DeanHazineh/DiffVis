from functools import partial
import torch.nn as nn
import torch
import numpy as np

from diffvis.nn_utils import *
from diffvis.nn_blocks import *


def convert_module_to_f16(module):
    """
    Converts module parameters and buffers to float16, excluding GroupNorm32, nn.GroupNorm, and nn.LayerNorm layers.

    Args:
        module (nn.Module): The module to convert.
    """
    # Convert parameters to float16
    for name, param in module.named_parameters(recurse=False):
        param.data = param.data.half()
        if param.grad is not None:
            param.grad.data = param.grad.data.half()

    # Convert buffers to float16
    for name, buffer in module.named_buffers(recurse=False):
        buffer.data = buffer.data.half()


class UNetModelX(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_res_blocks,
        res_use_conv=False,
        time_embed_dim=1024,
        use_scale_shift_norm=False,
        block_in_channels=(64, 128, 256),
        attn=(True, True, True),
        sattn=None,
        attn_on_upsample=True,
        attn_head_dim=32,
        sattn_head_dim=32,
        group_norm_num=32,
        sgroup_norm_num=8,
        ds_use_conv=False,
        dropout=0.0,
        use_checkpoint=True,
        use_crossattention=False,
        use_spatial_transformer=False,
        spatial_transformer_depth=1,
        spatial_transformer_linear=True,
        use_xformer=True,
        context_dim=0,  # crossattention
        input_size=[64, 64],
        attn_positional_encoding=False,
        sattn_positional_encoding=False,
        SE_Net=False,
        SE_R=4,
        SE_Pos="pre",
        fuse_xcond_t=False,
        double_res=False,
    ):
        super().__init__()
        assert len(attn) == len(
            block_in_channels
        ), "Attention block listings should be the same length as block channels."
        if sattn is None:
            sattn = [False for i in range(len(attn))]

        if use_crossattention and context_dim == 0:
            raise ValueError(
                "cond_dim must tbe specified if use_crossattention is True. You forgot to specify the context dimension"
            )

        num_blocks = len(block_in_channels)
        # Time embedding
        self.fuse_xcond_t = fuse_xcond_t
        self.time_embedding = nn.Sequential(
            TimestepEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Define partial modules for resblocks and attention
        use_resblock = partial(
            ResBlock,
            emb_channels=time_embed_dim,
            dropout=dropout,
            use_scale_shift_norm=use_scale_shift_norm,
            use_conv=res_use_conv,
            use_checkpoint=use_checkpoint,
            group_norm_num=group_norm_num,
            SE_Net=SE_Net,
            SE_R=SE_R,
            SE_Pos=SE_Pos,
        )

        if use_crossattention or use_spatial_transformer:
            use_attention = partial(
                SpatialTransformer,
                attn_dims=attn_head_dim,
                depth=spatial_transformer_depth,
                group_norm_num=group_norm_num,
                cond_dim=(context_dim if use_crossattention else None),
                dropout=dropout,
                use_checkpoint=use_checkpoint,
                use_linear=spatial_transformer_linear,
                use_xformer=use_xformer,
            )
        else:  # Use vanilla attention operations
            use_attention = partial(
                AttentionBlock,
                use_checkpoint=use_checkpoint,
                group_norm_num=group_norm_num,
                use_xformer=use_xformer,
                use_positional_encoding=attn_positional_encoding,
                max_seq_length=int(np.prod(input_size)),
            )

        # use_satten = partial(MSAB, num_blocks=1)
        # use_satten = partial(
        #     SpectralAttentionBlock,
        #     use_checkpoint=use_checkpoint,
        #     group_norm_num=sgroup_norm_num,
        #     use_xformer=use_xformer,
        #     use_positional_encoding=sattn_positional_encoding,
        # )

        # Initial conv layer to increase the channel dimensionality to the first block size
        self.input_blocks = nn.ModuleList()
        chs = block_in_channels[0]
        hw_length = int(input_size[0] * input_size[1])
        in_chs = []
        self.input_blocks.append(
            AbstractSequential(nn.Conv2d(in_channels, chs, kernel_size=3, padding=1))
        )
        in_chs.append(chs)

        # Create first half of the Unet
        for block_num, ch in enumerate(block_in_channels):
            for _ in range(num_res_blocks):
                layers = [use_resblock(chs, out_channels=ch)]

                if double_res:
                    layers.append(use_resblock(ch, out_channels=ch))

                if attn[block_num]:
                    print(
                        f"Block {block_num} - ch: {ch}, attn_heads: {ch // attn_head_dim}"
                    )
                    layers.append(
                        use_attention(
                            query_dim=ch,
                            # hw_length=hw_length,
                            attn_heads=ch // attn_head_dim,
                        )
                    )
                if sattn[block_num]:
                    layers.append(
                        use_satten(
                            in_channels=ch,
                            dim_head=sattn_head_dim,
                            heads=ch // sattn_head_dim,
                        )
                    )
                self.input_blocks.append(AbstractSequential(*layers))
                in_chs.append(ch)
                chs = ch

            if block_num != num_blocks - 1:
                self.input_blocks.append(
                    AbstractSequential(Downsample(ch, ds_use_conv))
                )
                in_chs.append(ch)
                hw_length = hw_length // 4

        # Create the Middle of the Unet
        modules = [
            use_resblock(chs, out_channels=chs),
            use_attention(
                query_dim=ch,
                # hw_length=hw_length,
                attn_heads=ch // attn_head_dim,
            ),
        ]
        if sattn[-1]:
            modules.append(
                # use_satten(
                #     in_channels=ch,
                #     hw_length=hw_length,
                #     attn_heads=hw_length // sattn_head_dim,
                # )
                use_satten(
                    in_channels=ch,
                    dim_head=sattn_head_dim,
                    heads=ch // sattn_head_dim,
                )
            )
        modules.append(use_resblock(chs, out_channels=chs))
        self.middle_block = AbstractSequential(*modules)

        # Create the end of the Unet
        self.output_blocks = nn.ModuleList()
        for block_num, ch in list(enumerate(block_in_channels))[::-1]:
            for _ in range(num_res_blocks):
                layers = [use_resblock(chs + in_chs.pop(), out_channels=ch)]

                if double_res:
                    layers.append(use_resblock(ch, out_channels=ch))

                if attn[block_num] and attn_on_upsample:
                    layers.append(
                        use_attention(
                            query_dim=ch,
                            # hw_length=hw_length,
                            attn_heads=ch // attn_head_dim,
                        )
                    )
                if sattn[block_num] and attn_on_upsample:
                    # layers.append(
                    #     use_satten(
                    #         in_channels=ch,
                    #         hw_length=hw_length,
                    #         attn_heads=hw_length // sattn_head_dim,
                    #     )
                    # )
                    layers.append(
                        use_satten(
                            in_channels=ch,
                            dim_head=sattn_head_dim,
                            heads=ch // sattn_head_dim,
                        )
                    )
                self.output_blocks.append(AbstractSequential(*layers))
                chs = ch

            layers = [use_resblock(ch + in_chs.pop(), out_channels=ch)]
            if block_num != 0:
                layers.append(Upsample(ch, use_conv=ds_use_conv))
                hw_length = int(hw_length * 4)
            self.output_blocks.append(AbstractSequential(*layers))

        # Create the output projection
        self.out = nn.Sequential(
            nn.GroupNorm(group_norm_num, chs),
            nn.SiLU(),
            zero_module(nn.Conv2d(chs, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, context=None):
        embedded_time = self.time_embedding(timesteps)

        if self.fuse_xcond_t:
            embedded_time = embedded_time + context

        hx = []
        for module in self.input_blocks:
            x = module(x, embedded_time, context)
            hx.append(x)

        x = self.middle_block(x, embedded_time, context)

        for module in self.output_blocks:
            cat_in = torch.cat([x, hx.pop()], dim=1)
            x = module(cat_in, embedded_time, context)

        return self.out(x)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        # self.apply(convert_module_to_f16)
        # self.apply(convert_module_to_f16)
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        self.out.apply(convert_module_to_f16)
        self.time_embedding.apply(convert_module_to_f16)


if __name__ == "__main__":
    model = UNetModelX(
        input_size=[64, 64],
        in_channels=32,
        out_channels=31,
        num_res_blocks=1,
        res_use_conv=False,
        time_embed_dim=1024,
        use_scale_shift_norm=False,
        block_in_channels=[64, 128, 256, 512, 512],
        attn=[True, True, True, True, True],
        attn_on_upsample=True,
        attn_head_dim=32,
        group_norm_num=32,
        ds_use_conv=True,
        dropout=0.0,
        use_checkpoint=False,
        use_crossattention=False,
        use_spatial_transformer=False,
        context_dim=0,
        use_xformer=True,
    )
    model.to(device="cuda", dtype=torch.float16)
    model.convert_to_fp16()

    timesteps = torch.tensor([1000, 1000], dtype=torch.float16, device="cuda")
    x = torch.rand((2, 32, 64, 64), device="cuda", dtype=torch.float16)
    out = model(x, timesteps)
    print(out.shape)
    print(out)
