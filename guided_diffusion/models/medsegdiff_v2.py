import math
from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from batchgenerators.augmentations.utils import pad_nd_image
from scipy.ndimage.filters import gaussian_filter
from torch.cuda.amp import autocast

# local imports
from ..fp16_util import convert_module_to_f16, convert_module_to_f32
from ..nn import (
    avg_pool_nd,
    checkpoint,
    conv_nd,
    layer_norm,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)
from ..utils import InitWeights_He, maybe_to_torch, no_op, sigmoid_helper, softmax_helper, to_cuda
from .base_blocks import (
    AttentionBlock,
    Downsample,
    MobileNetBlock,
    QKVAttention,
    QKVAttentionLegacy,
    ResBlock,
    TimestepEmbedSequential,
    Upsample,
)
from .condition_net import GenericUNet, SS_Former, SymmetryEnhancedAttention


class UNetMedSegDiffV2(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    This version corresponds to MedSegDiff-V2, including a 2.5D conditioning U-Net.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
        bottleneck_ch = ch
        bottleneck_size = image_size // ds
        self.ss_former = SS_Former(
            bottleneck_ch, bottleneck_ch, height=bottleneck_size, width=bottleneck_size
        )
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        self.hwm = GenericUNet(self.in_channels - 1, 16, bottleneck_ch, 4, anchor_out=True)
        self.encoder_2_5d = GenericUNet(
            1,
            8,
            128,
            4,
            conv_op=nn.Conv3d,
            norm_op=nn.GroupNorm,
            norm_op_kwargs={"num_groups": 8, "eps": 1e-05, "affine": True},
            num_conv_per_stage=1,
        )
        self.decoder_2d = GenericUNet(
            128,
            8,
            bottleneck_ch,
            4,
            conv_op=nn.Conv2d,
            norm_op=nn.GroupNorm,
            norm_op_kwargs={"num_groups": 8, "eps": 1e-05, "affine": True},
            num_conv_per_stage=1,
        )
        self.sea = SymmetryEnhancedAttention(128)
        self.cal_head = nn.Sequential(
            normalization(bottleneck_ch), nn.SiLU(), conv_nd(dims, bottleneck_ch, 1, 1)
        )

    def forward(self, x, timesteps, y=None, x_2_5d=None):
        assert x_2_5d is not None, "MedSegDiff-V2 requires a 2.5D input `x_2_5d`"
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            emb = emb + self.label_emb(y)
        _, features_3d = self.encoder_2_5d(x_2_5d.type(self.dtype))
        sea_features_3d = self.sea(features_3d)
        center_slice_idx = sea_features_3d.shape[2] // 2
        sea_features_2d = sea_features_3d[:, :, center_slice_idx, :, :]
        semantic_cond, _ = self.decoder_2d(sea_features_2d)
        anchor_cond, _ = self.hwm(x[:, :-1, ...])
        cal = self.cal_head(anchor_cond)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.ss_former(h, anchor_cond, semantic_cond)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        main_output = self.out(h)
        return (main_output, cal)
