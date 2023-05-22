import math
import os

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import Upsample, resize

import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList,
                         load_state_dict)
from mmcv.utils import to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils.embed import PatchEmbed, PatchMerging


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.head_embed_dims = embed_dims // num_heads
        # self.scale = qk_scale or head_embed_dims ** -0.5
        # define a parameter table of relative position bias
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((window_size[0] - 1) * (window_size[1] - 1),
        #                 num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        # rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        # rel_position_index = rel_index_coords + rel_index_coords.T
        # rel_position_index = rel_position_index.flip(1).contiguous()
        # self.register_buffer('relative_position_index', rel_position_index)
        self.dynamic_scale = nn.Parameter(torch.zeros(Wh*Ww, Wh*Ww//4))
        self.fc_q = nn.Linear(embed_dims, embed_dims)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.outer_bias = nn.Parameter(torch.zeros(num_heads, Wh*Ww, Wh*Ww//4))
        self.inner_bias = nn.Parameter(torch.zeros(num_heads, Wh*Ww, Wh*Ww//4))
        self.softmax = nn.Softmax(dim=-1)

        self.conv_downsampling = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=2,
                                           stride=2, groups=embed_dims)

    def init_weights(self):
        # trunc_normal_(self.relative_position_bias_table, std=0.02)
        pass

    def forward(self, x, skip_x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        assert x.shape == skip_x.shape, 'x.shape != skip_x.shape in WindowMSA'
        # print(x.shape)
        # print(self.embed_dims)
        # os.system("pause")
        q = self.fc_q(skip_x)
        q = q.reshape(B, N, self.num_heads, self.head_embed_dims).permute(0, 2, 1, 3)

        x = x.permute(0, 2, 1)
        x = x.reshape(B, C, 12, 12)
        x = self.conv_downsampling(x)
        N = int(N // 4)
        # print(x.shape)
        x = x.reshape(B, C, N)
        x = x.reshape(B, self.num_heads, self.head_embed_dims, N).permute(0, 1, 3, 2)
        # print(x.shape)
        # make torchscript happy (cannot use tensor as tuple)
        # q, k, v = qkv[0], qkv[1], qkv[2]

        # q = q * self.scale
        k = v = x
        attn = (q @ k.transpose(-2, -1))
        # print(attn.shape)
        # os.system("pause")
        attn = attn * self.dynamic_scale
        # relative_position_bias = self.relative_position_bias_table[
        #     self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1],
        #     self.window_size[0] * self.window_size[1],
        #     -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(
        #     2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn + self.inner_bias
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(0)
            mask = mask[:,:,:,:,::4]
            nW = mask.shape[1]
            attn = attn.view(B // nW, nW, self.num_heads, N*4,
                             N) + mask
            attn = attn.view(-1, self.num_heads, N*4, N)
        attn = self.softmax(attn)
        attn = attn + self.outer_bias
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N*4, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, skip_query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        assert query.shape == skip_query.shape, 'skip query should has the same shape with query'
        query = query.view(B, H, W, C)
        skip_query = skip_query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        skip_query = F.pad(skip_query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))
            shifted_skip_query = torch.roll(
                skip_query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            shifted_skip_query = skip_query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        skip_query_windows = self.window_partition(shifted_skip_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size ** 2, C)
        skip_query_windows = skip_query_windows.view(-1, self.window_size ** 2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, skip_query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(SwinBlock, self).__init__(init_cfg=init_cfg)

        self.with_cp = with_cp
        self.skip_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

        self.norm3 = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x, skip_x, hw_shape):

        def _inner_forward(x, skip_x):
            identity = x
            # print(x.shape)
            skip_x = self.skip_norm(skip_x)
            x = self.norm1(x)
            x = self.attn(x, skip_x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            x = self.norm3(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x, skip_x)

        return x


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,

                 is_upsample=False,
                 is_concat=True,

                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)

        self.is_upsample = is_upsample
        self.conv = ConvModule(
            in_channels=embed_dims * 2,
            out_channels=embed_dims * 2,
            kernel_size=1,
            stride=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=act_cfg)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x, skip_x, hw_shape):
        for block in self.blocks:
            x = block(x, skip_x, hw_shape)

        if self.is_upsample:
            x = torch.cat([x, skip_x], dim=2)
            up_hw_shape = [hw_shape[0] * 2, hw_shape[1] * 2]
            # print(x.shape)
            # os.system("pause")
            B, HW, C = x.shape
            x = x.view(B, hw_shape[0], hw_shape[1], C)
            x = x.permute(0, 3, 1, 2)
            '''
            x_up = resize(
                input=self.conv(x_up),
                # size=inputs[0].shape[2:],
                size=up_hw_shape,
                # mode=self.interpolate_mode,
                # align_corners=self.align_corners
            )
            '''
            # x_up = self.conv(x_up)

            x = self.conv(x)

            x = self.ps(x)
            x = x.permute(0, 2, 3, 1).view(B, up_hw_shape[0] * up_hw_shape[1], C // 4)
            return x
        else:
            # x_cat = torch.cat([x, skip_x], dim=2)
            x = torch.cat([x, skip_x], dim=2)
            B, HW, C = x.shape
            x = x.view(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2)
            x = self.conv(x)
            x = x.permute(0, 2, 3, 1).view(B, hw_shape[0] * hw_shape[1], C)
            return x


@HEADS.register_module()
class MusterHead_lsla_depth(BaseDecodeHead):

    def __init__(self,
                 embed_dims=768,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        num_layers = len(depths)

        assert strides[3] == patch_size, 'Use non-overlapping patch embed.'

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                is_upsample = True
                is_concat = True
            else:
                is_upsample = False
                is_concat = False

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=int(mlp_ratio * in_channels),
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],

                is_upsample=is_upsample,
                is_concat=is_concat,

                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)
            if is_upsample:
                in_channels = in_channels // 2

        self.num_features = [int(embed_dims * 2 ** i) for i in range(num_layers)]
        self.mlp_ratio = mlp_ratio

        # self.ffns = ModuleList()
        # for i in range(num_layers):
        #     mlp = FFN(
        #         embed_dims=in_channels,
        #         feedforward_channels=int(self.mlp_ratio * in_channels),
        #         num_fcs=2,
        #         ffn_drop=drop_rate,
        #         dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
        #         act_cfg=act_cfg,
        #         add_identity=True,
        #         init_cfg=None)
        #     self.ffns.append(mlp)
        #     in_channels *= 2
        self.outffn = FFN(
            embed_dims=self.channels,
            feedforward_channels=int(self.mlp_ratio * self.channels),
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)

        B, C, H, W = inputs[3].shape
        hw_shape = (H, W)
        index = 0
        # print(inputs[3].shape)
        # for ffn in self.ffns:
        #     ind = inputs[index]
        #     ind = ind.permute(0, 2, 3, 1)
        #     B, H, W, C = ind.shape
        #     hw_shape = (H, W)
        #     ind = ind.view(B, H * W, C)
        #     ind = ffn(ind)
        #     inputs[index] = ind
        #     index += 1
        for i, input in enumerate(inputs):
            inputs[i] = inputs[i].permute(0, 2, 3, 1)
            inputs[i] = inputs[i].reshape(inputs[i].shape[0], inputs[i].shape[1] * inputs[i].shape[2],
                                          inputs[i].shape[3])
        # print(inputs[3].shape)
        x = inputs[3]
        for i, stage in enumerate(self.stages):
            C //= 2
            x = stage(x, inputs[3 - i], hw_shape)
            hw_shape = (hw_shape[0] * 2, hw_shape[1] * 2)

        out = x
        # out = torch.cat([out, inputs[0]], dim=2)
        out = self.outffn(out)
        out = out.view(B, hw_shape[0] // 2, hw_shape[1] // 2, C * 4).permute(0, 3, 1, 2)

        out = self.cls_seg(out)

        return out
