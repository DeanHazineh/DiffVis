from functools import partial
import torch.nn as nn
import torch

from diffvis.diffusion.nn_blocks import (
    SpectralAttentionBlock,
    AttentionBlock,
    ResBlockNoTime,
    Downsample,
    Upsample,
)
from diffvis.diffusion.nn_utils import zero_module


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_res_blocks,
        res_use_conv=False,
        block_in_channels=(64, 128, 256),
        attn=(True, True, True),
        sattn=None,
        attn_on_upsample=True,
        attn_head_dim=32,
        sattn_head_dim=32,
        group_norm_num=32,
        sgroup_norm_num=32,
        ds_use_conv=False,
        dropout=0.0,
        use_checkpoint=True,
        use_xformer=True,
        input_size=[64, 64],
        attn_positional_encoding=False,
        sattn_positional_encoding=False,
    ):
        super().__init__()
        assert len(attn) == len(
            block_in_channels
        ), "Attention block listings should be the same length as block channels."
        if sattn is None:
            sattn = [False for i in range(len(attn))]

        num_blocks = len(block_in_channels)

        # Define partial modules for resblocks and attention
        use_resblock = partial(
            ResBlockNoTime,
            dropout=dropout,
            use_conv=res_use_conv,
            use_checkpoint=use_checkpoint,
            group_norm_num=group_norm_num,
        )

        use_attention = partial(
            AttentionBlock,
            use_checkpoint=use_checkpoint,
            group_norm_num=group_norm_num,
            use_xformer=use_xformer,
            use_positional_encoding=attn_positional_encoding,
        )

        use_satten = partial(
            SpectralAttentionBlock,
            use_checkpoint=use_checkpoint,
            group_norm_num=sgroup_norm_num,
            use_xformer=use_xformer,
        )

        # Initial conv layer to increase the channel dimensionality to the first block size
        self.input_blocks = nn.ModuleList()
        chs = block_in_channels[0]
        hw_length = int(input_size[0] * input_size[1])
        in_chs = []
        self.input_blocks.append(
            nn.Sequential(nn.Conv2d(in_channels, chs, kernel_size=3, padding=1))
        )
        in_chs.append(chs)

        # Create first half of the Unet
        for block_num, ch in enumerate(block_in_channels):
            for _ in range(num_res_blocks):
                layers = [use_resblock(chs, out_channels=ch)]
                if attn[block_num]:
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
                            dim=hw_length,
                            dim_head=sattn_head_dim,
                            heads=hw_length // sattn_head_dim,
                        )
                    )
                self.input_blocks.append(nn.Sequential(*layers))
                in_chs.append(ch)
                chs = ch

            if block_num != num_blocks - 1:
                self.input_blocks.append(nn.Sequential(Downsample(ch, ds_use_conv)))
                in_chs.append(ch)
                hw_length = hw_length // 4

        # Create the Middle of the Unet
        modules = [use_resblock(chs, out_channels=chs)]
        if attn[-1]:
            modules.append(
                use_attention(
                    query_dim=ch,
                    # hw_length=hw_length,
                    attn_heads=ch // attn_head_dim,
                )
            )
        if sattn[-1]:
            modules.append(
                use_satten(
                    dim=hw_length,
                    dim_head=sattn_head_dim,
                    heads=hw_length // sattn_head_dim,
                )
            )
        modules.append(use_resblock(chs, out_channels=chs))
        self.middle_block = nn.Sequential(*modules)

        # Create the end of the Unet
        self.output_blocks = nn.ModuleList()
        for block_num, ch in list(enumerate(block_in_channels))[::-1]:
            for _ in range(num_res_blocks):
                layers = [use_resblock(chs + in_chs.pop(), out_channels=ch)]
                if attn[block_num] and attn_on_upsample:
                    layers.append(
                        use_attention(
                            query_dim=ch,
                            # hw_length=hw_length,
                            attn_heads=ch // attn_head_dim,
                        )
                    )
                if sattn[block_num] and attn_on_upsample:
                    use_satten(
                        dim=hw_length,
                        dim_head=sattn_head_dim,
                        heads=hw_length // sattn_head_dim,
                    )
                self.output_blocks.append(nn.Sequential(*layers))
                chs = ch

            layers = [use_resblock(ch + in_chs.pop(), out_channels=ch)]
            if block_num != 0:
                layers.append(Upsample(ch, use_conv=ds_use_conv))
                hw_length = int(hw_length * 4)
            self.output_blocks.append(nn.Sequential(*layers))

        # Create the output projection
        self.out = nn.Sequential(
            nn.GroupNorm(group_norm_num, chs),
            nn.SiLU(),
            zero_module(nn.Conv2d(chs, out_channels, 3, padding=1)),
        )

    def forward(self, x):
        hx = []
        for module in self.input_blocks:
            x = module(x)
            hx.append(x)

        x = self.middle_block(x)

        for module in self.output_blocks:
            cat_in = torch.cat([x, hx.pop()], dim=1)
            x = module(cat_in)

        return self.out(x)
