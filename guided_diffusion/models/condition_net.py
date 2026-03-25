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
    ConvNormNonlin,
    Downsample,
    ResBlock,
    StackedConvLayers,
    TimestepEmbedSequential,
    Upsample,
)


class FFParser(nn.Module):

    def __init__(self, dim, h=128, w=65):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape
        assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.reshape(B, C, H, W)
        return x


class GenericUNet(nn.Module):

    def __init__(
        self,
        input_channels,
        base_num_features,
        num_classes,
        num_pool,
        num_conv_per_stage=2,
        feat_map_mul_on_downscale=2,
        conv_op=nn.Conv2d,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=None,
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
        deep_supervision=False,
        dropout_in_localization=False,
        final_nonlin=lambda x: x,
        weightInitializer=InitWeights_He(0.01),
        pool_op_kernel_sizes=None,
        conv_kernel_sizes=None,
        upscale_logits=False,
        convolutional_pooling=True,
        convolutional_upsampling=True,
        max_num_features=None,
        basic_block=ConvNormNonlin,
        seg_output_use_bias=False,
        highway=False,
        anchor_out=False,
    ):
        super(GenericUNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.nonlin = nonlin
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.highway = highway
        self.anchor_out = anchor_out
        self.conv_kwargs = {"stride": 1, "dilation": 1, "bias": True}
        self.norm_op_kwargs = {"eps": 1e-05, "affine": True}
        if norm_op_kwargs is not None:
            self.norm_op_kwargs = norm_op_kwargs
        self.dropout_op_kwargs = {"p": 0, "inplace": True}
        self.nonlin_kwargs = {"negative_slope": 0.01, "inplace": True}
        self.conv_blocks_context = nn.ModuleList()
        self.conv_blocks_localization = nn.ModuleList()
        self.td = nn.ModuleList()
        self.tu = nn.ModuleList()
        self.seg_outputs = nn.ModuleList()
        output_features = base_num_features
        input_features = input_channels
        if pool_op_kernel_sizes is None:
            if self.conv_op == nn.Conv3d:
                pool_op_kernel_sizes = [(1, 2, 2)] * num_pool
            else:
                pool_op_kernel_sizes = [2] * num_pool
        if self.conv_op == nn.Conv2d:
            conv_kernel_size = 3
            conv_padding = 1
        elif self.conv_op == nn.Conv3d:
            conv_kernel_size = (3, 3, 3)
            conv_padding = (1, 1, 1)
        else:
            raise ValueError(f"Unsupported conv_op: {self.conv_op}")
        for d in range(num_pool):
            self.conv_kwargs["kernel_size"] = conv_kernel_size
            self.conv_kwargs["padding"] = conv_padding
            self.conv_blocks_context.append(
                StackedConvLayers(
                    input_features,
                    output_features,
                    num_conv_per_stage,
                    self.conv_op,
                    self.conv_kwargs,
                    self.norm_op,
                    self.norm_op_kwargs,
                    self.dropout_op,
                    self.dropout_op_kwargs,
                    self.nonlin,
                    self.nonlin_kwargs,
                    basic_block=basic_block,
                )
            )
            if self.convolutional_pooling:
                pool_op = self.conv_op(
                    output_features,
                    output_features,
                    pool_op_kernel_sizes[d],
                    pool_op_kernel_sizes[d],
                    bias=False,
                )
            else:
                pool_op = (nn.MaxPool2d if self.conv_op == nn.Conv2d else nn.MaxPool3d)(
                    pool_op_kernel_sizes[d]
                )
            self.td.append(pool_op)
            input_features = output_features
            output_features = int(round(output_features * feat_map_mul_on_downscale))
        self.conv_kwargs["kernel_size"] = conv_kernel_size
        self.conv_kwargs["padding"] = conv_padding
        self.conv_blocks_context.append(
            nn.Sequential(
                StackedConvLayers(
                    input_features,
                    output_features,
                    num_conv_per_stage,
                    self.conv_op,
                    self.conv_kwargs,
                    self.norm_op,
                    self.norm_op_kwargs,
                    self.dropout_op,
                    self.dropout_op_kwargs,
                    self.nonlin,
                    self.nonlin_kwargs,
                    basic_block=basic_block,
                )
            )
        )
        for u in range(num_pool):
            if self.convolutional_upsampling:
                if self.conv_op == nn.Conv2d:
                    transpconv = nn.ConvTranspose2d(
                        output_features, output_features // 2, 2, 2, bias=False
                    )
                elif self.conv_op == nn.Conv3d:
                    transpconv = nn.ConvTranspose3d(
                        output_features, output_features // 2, (1, 2, 2), (1, 2, 2), bias=False
                    )
                else:
                    raise ValueError(f"Unsupported conv_op: {self.conv_op}")
                self.tu.append(transpconv)
            elif self.conv_op == nn.Conv3d:
                self.tu.append(nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear"))
            else:
                self.tu.append(
                    nn.Upsample(
                        scale_factor=2,
                        mode="bilinear" if self.conv_op == nn.Conv2d else "trilinear",
                    )
                )
            self.conv_blocks_localization.append(
                nn.Sequential(
                    StackedConvLayers(
                        output_features,
                        output_features // 2,
                        num_conv_per_stage - 1,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                        basic_block=basic_block,
                    ),
                    StackedConvLayers(
                        output_features // 2,
                        output_features // 2,
                        1,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                        basic_block=basic_block,
                    ),
                )
            )
            output_features //= 2
        self.final_conv = (
            nn.Conv2d(output_features, num_classes, 1)
            if conv_op == nn.Conv2d
            else nn.Conv3d(output_features, num_classes, 1)
        )
        self.final_nonlin = final_nonlin

    def forward(self, x):
        skips = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if x.shape[-1] < 2:
                x = th.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            x = self.td[d](x)
        x = self.conv_blocks_context[-1](x)
        bottleneck_features = x
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            skip = skips.pop()
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat((x, skip), dim=1)
            x = self.conv_blocks_localization[u](x)
        seg_output = self.final_conv(x)
        return (self.final_nonlin(seg_output), bottleneck_features)


class NBP_Filter(nn.Module):

    def __init__(self, channel, h, w):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(channel, h, w, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        return x


class SS_Former(nn.Module):

    def __init__(self, in_channels, out_channels, height, width, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.norm = normalization(in_channels)
        self.q_conv = conv_nd(2, in_channels, in_channels, 1)
        self.k_conv = conv_nd(2, in_channels, in_channels, 1)
        self.v_conv = conv_nd(2, in_channels, in_channels, 1)
        self.nbp_filter = NBP_Filter(in_channels, h=height, w=width // 2 + 1)
        self.proj_out = zero_module(conv_nd(2, in_channels, out_channels, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(),
            nn.Linear(in_channels * 4, in_channels),
        )

    def forward(self, x, anchor_cond, semantic_cond):
        b, c, h, w = x.shape
        if anchor_cond.shape[-2:] != x.shape[-2:]:
            anchor_cond = F.interpolate(
                anchor_cond, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        if semantic_cond.shape[-2:] != x.shape[-2:]:
            semantic_cond = F.interpolate(
                semantic_cond, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        q = self.q_conv(x)
        k = self.k_conv(semantic_cond)
        v = self.v_conv(anchor_cond)
        q_fft = self.nbp_filter(q)
        k_fft = self.nbp_filter(k)
        scale = 1 / math.sqrt(c)
        weight = torch.einsum("bchw,bchw->bhw", q_fft * scale, k_fft * scale)
        weight = torch.softmax(weight.view(b, -1), dim=-1).view(b, h, w)
        attn = torch.einsum("bhw,bchw->bchw", weight, v)
        attn = attn.permute(0, 2, 3, 1)
        mlp_out = self.mlp(attn)
        mlp_out = mlp_out.permute(0, 3, 1, 2)
        return self.proj_out(mlp_out) + x


class SymmetryEnhancedAttention(nn.Module):
    """
    Symmetry Enhanced Attention (SEA) module.
    As described in the MedSegDiff-V2 diagram.
    This is an exemplary implementation based on the diagram.
    It assumes 3D features and performs self-attention and symmetry-attention.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def _forward(self, x):
        b, c, d, height, width = x.shape
        x_flat = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x_flat))
        self_h = self.attention(qkv)
        x_flipped = th.flip(x, dims=[2])
        x_flipped_flat = x_flipped.reshape(b, c, -1)
        qkv_flipped = self.qkv(self.norm(x_flipped_flat))
        q, _, _ = th.chunk(qkv, 3, dim=1)
        _, k_flipped, v_flipped = th.chunk(qkv_flipped, 3, dim=1)
        qkv_sym = th.cat([q, k_flipped, v_flipped], dim=1)
        sym_h = self.attention(qkv_sym)
        h = self_h + sym_h
        h = self.proj_out(h)
        return (x_flat + h).reshape(b, c, d, height, width)

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)
