import torch.nn as nn
from functools import partial
from .nn_blocks import ResBlockNoTime, AttentionBlock, Downsample
from .nn_utils import zero_module


class ImageAdapter(nn.Module):
    def __init__(
        self,
        in_channels,
        block_in_channels=(64, 128, 256),
        attn=(True, True, True),
        embd_dim=1024,
        attn_head_dim=32,
        num_res_blocks=1,
        group_norm_num=16,
        ds_conv=False,
        use_checkpoint=False,
        use_xformer=True,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        # Initial transformation
        in_layers = [
            nn.Conv2d(in_channels, block_in_channels[0], kernel_size=3, padding=1)
        ]
        self.blocks.append(nn.Sequential(*in_layers))

        # Encoder Scheme
        use_resblock = partial(
            ResBlockNoTime,
            dropout=0.0,
            use_conv=True,
            use_checkpoint=use_checkpoint,
            group_norm_num=group_norm_num,
        )
        use_attn = partial(
            AttentionBlock,
            group_norm_num=group_norm_num,
            use_checkpoint=use_checkpoint,
            use_xformer=use_xformer,
            use_positional_encoding=False,
            hw_length=None,
        )
        use_ds = partial(Downsample, use_conv=ds_conv)

        chs = block_in_channels[0]
        for block_num, ch in enumerate(block_in_channels):
            for _ in range(num_res_blocks):
                layers = [use_resblock(chs, out_channels=ch)]
                if attn[block_num]:
                    layers.append(
                        use_attn(
                            in_channels=ch,
                            attn_heads=ch // attn_head_dim,
                        )
                    )
                chs = ch
                self.blocks.append(nn.Sequential(*layers))

            if block_num != len(block_in_channels) - 1:
                self.blocks.append(use_ds(chs))
            else:
                self.blocks.append(nn.AdaptiveAvgPool2d(1))

        self.out = nn.Sequential(
            nn.Linear(chs, chs),
            nn.SiLU(),
            zero_module(nn.Linear(chs, embd_dim)),
        )

    def forward(self, x):
        for module in self.blocks:
            x = module(x)

        x = self.out(x.view(*x.shape[:2]))
        return x
