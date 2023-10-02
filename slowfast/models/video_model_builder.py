#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch
import torch.nn as nn

import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY

# Qitong on Jan. 3rd, 2022.
import torch.nn.functional as F
import math

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
}

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
}


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


@MODEL_REGISTRY.register()
class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )
        if cfg.MODEL.FREEZE_STAGES > 0:
            self.freeze_param(cfg.MODEL.FREEZE_STAGES)

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        # print(pool_size)
        # print({len(pool_size), self.num_pathways})
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if cfg.DETECTION.ENABLE:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        1,
                        1,
                    ],
                    [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1],
                ],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            head = head_helper.ResNetBasicHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None],
                # if cfg.MULTIGRID.SHORT_CYCLE
                # else [
                #     [
                #         cfg.DATA.NUM_FRAMES
                #         // cfg.SLOWFAST.ALPHA
                #         // pool_size[0][0],
                #         cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                #         cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                #     ],
                #     [
                #         cfg.DATA.NUM_FRAMES // pool_size[1][0],
                #         cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                #         cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                #     ],ResNet
                # ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                test_noact=cfg.TEST.NO_ACT,
            )
            self.head_name = "head{}".format(cfg.TASK)
            self.add_module(self.head_name, head)
        
    def freeze_param(self, freeze_stages):
        assert freeze_stages <= 5
        stages = [self.s1, self.s2, self.s3, self.s4, self.s5]
        for stage in stages[:freeze_stages]:
            for p in stage.parameters():
                p.requires_grad = False

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)

        head = getattr(self, self.head_name)
        if self.enable_detection:
            x, loss = head(x, bboxes)
        else:
            x, loss = head(x)
        return x, loss


@MODEL_REGISTRY.register()
class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        # Qitong on Jan. 3rd, 2022.
        self.PiTS = cfg.MODEL.PiTS
        self.cur_epoch = -1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )
        if cfg.MODEL.FREEZE_STAGES > 0:
            self.freeze_param(cfg.MODEL.FREEZE_STAGES)

    def get_cur_epoch(self, cur_epoch):
        self.cur_epoch = cur_epoch

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()
        self.contrastive = cfg.MODEL.CONTRASTIVE
        self.attention = cfg.MODEL.ATT
        self.unsup = cfg.MODEL.UNSUP
        self.max_epoch = cfg.SOLVER.MAX_EPOCH
        self.pi_ts = cfg.MODEL.PiTS

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if self.enable_detection:
            head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        elif self.PiTS:
            head = head_helper.ResNetPiHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                batch_size=cfg.TRAIN.BATCH_SIZE,
                dis_mode=cfg.MODEL.PiKL,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                test_noact=cfg.TEST.NO_ACT,
            )
        elif self.contrastive and not self.attention and not self.unsup:
            head = head_helper.ResNetContrastiveHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                batch_size=cfg.TRAIN.BATCH_SIZE,
                temp=cfg.MODEL.TEMP,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                test_noact=cfg.TEST.NO_ACT,
            )
        elif self.contrastive and not self.attention and self.unsup:
            head = head_helper.ResNetUnsupervisedContrastiveHead(
                dim_in=[width_per_group * 32],
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                batch_size=cfg.TRAIN.BATCH_SIZE,
                temp=cfg.MODEL.TEMP,
                test_noact=cfg.TEST.NO_ACT,
            )
        elif self.contrastive and self.attention:
            head = head_helper.ResNetContrastiveAttHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                batch_size=cfg.TRAIN.BATCH_SIZE,
                temp=cfg.MODEL.TEMP,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                test_noact=cfg.TEST.NO_ACT,
            )
        else:
            head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                test_noact=cfg.TEST.NO_ACT,
            )
        self.head_name = "head{}".format(cfg.TASK)
        self.add_module(self.head_name, head)
    
    def freeze_param(self, freeze_stages):
        assert freeze_stages <= 5
        stages = [self.s1, self.s2, self.s3, self.s4, self.s5]
        for stage in stages[:freeze_stages]:
            for p in stage.parameters():
                p.requires_grad = False

    def forward(self, x, cur_epoch=-1, labels=None):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)

        head = getattr(self, self.head_name)
        if self.enable_detection:
            x, loss = head(x, bboxes)
        else:
            # Qitong on Jan. 3rd, 2022.
            if self.pi_ts:
                x, loss = head(x, float(self.cur_epoch) / self.max_epoch)
            else:
                x, loss = head(x)
        # print(loss)
        return x, loss


@MODEL_REGISTRY.register()
class ResNet_TS(nn.Module):
    """
    Qitong on Jan. 3rd, 2022.
    Momentum reference: https://github.com/salesforce/ALBEF/tree/9e9a5e952f72374c15cea02d3c34013554c86513
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet_TS, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )
        if cfg.MODEL.FREEZE_STAGES > 0:
            self.freeze_param(cfg.MODEL.FREEZE_STAGES)

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()
        self.contrastive = cfg.MODEL.CONTRASTIVE
        self.attention = cfg.MODEL.ATT
        self.unsup = cfg.MODEL.UNSUP
        self.batch_size = cfg.TRAIN.BATCH_SIZE

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s1_m = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        self.s2_m = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        for pathway in range(self.num_pathways):
            pool_m = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool_m".format(pathway), pool_m)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s3_m = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s4_m = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        self.s5_m = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if self.enable_detection:
            head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                test_noact=cfg.TEST.NO_ACT,
            )
            head_m = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                test_noact=cfg.TEST.NO_ACT,
            )
        self.head_name = "head{}".format(cfg.TASK)
        self.add_module(self.head_name, head)
        self.head_name_m = "head{}_m".format(cfg.TASK)
        self.add_module(self.head_name_m, head_m)

        self.model_pairs = [[self.s1, self.s1_m],
                            [self.s2, self.s2_m],
                            [self.s3, self.s3_m],
                            [self.s4, self.s4_m],
                            [self.s5, self.s5_m],
                            ]
        for pathway in range(self.num_pathways):
            self.model_pairs.append(
                [getattr(self, "pathway{}_pool".format(pathway)), getattr(self, "pathway{}_pool_m".format(pathway))]
            )
        self.model_pairs.append(
            [getattr(self, self.head_name), getattr(self, self.head_name_m)]
        )

        # self.copy_params()
        self.momentum = cfg.MODEL.MOMENTUM
        self.max_epoch = cfg.SOLVER.MAX_EPOCH
        self.disKL_mode = cfg.MODEL.PiKL
        self.pos_consistency_K = cfg.MODEL.MT_UNSUP_K
        self.neg_consistency_N = cfg.MODEL.MT_UNSUP_N
        self.KNN_D = cfg.MODEL.D
        self.lambda_mt = cfg.MODEL.L_MT
        self.lambda_pos = cfg.MODEL.L_POS
        self.lambda_con = cfg.MODEL.L_CON
        self.feat = cfg.MODEL.FEAT
        self.con_logit_l2 = cfg.MODEL.CON_LOGIT_L2
        if self.pos_consistency_K > -1:
            self.rank_list = torch.ones(self.pos_consistency_K)
            for idx in range(self.rank_list.size()[0]):
                # self.rank_list[idx] = self.rank_list[idx] * math.log(2, idx + 2)
                idx_ele = (1.0 - float(idx + 1) / self.rank_list.size()[0])
                # self.rank_list[idx] = 1
                self.rank_list[idx] = self.rank_list[idx] * idx_ele
                # self.rank_list[idx] = self.rank_list[idx] * math.exp(-5 * (1 - idx_ele) ** 2)
                # self.rank_list[idx] = self.rank_list[idx] * idx_ele ** 4
        # contrastive
        self.contrastive = cfg.MODEL.CONTRASTIVE
        self.queue_size = cfg.MODEL.QUEUE
        self.num_gpus = cfg.NUM_GPUS
        if self.contrastive:
            if self.feat:
                self.head_con = head_helper.ResNetMeanTeacherContrastiveHead(
                    dim_in=[width_per_group * 32],
                    pool_size=[None, None]
                    if cfg.MULTIGRID.SHORT_CYCLE
                    else [
                        [
                            cfg.DATA.NUM_FRAMES // pool_size[0][0],
                            cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                            cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                        ]
                    ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                    dropout_rate=cfg.MODEL.DROPOUT_RATE,
                    test_noact=cfg.TEST.NO_ACT,
                    con_last_dim=cfg.MODEL.CON_LAST_DIM
                )
                self.head_con_m = head_helper.ResNetMeanTeacherContrastiveHead(
                    dim_in=[width_per_group * 32],
                    pool_size=[None, None]
                    if cfg.MULTIGRID.SHORT_CYCLE
                    else [
                        [
                            cfg.DATA.NUM_FRAMES // pool_size[0][0],
                            cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                            cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                        ]
                    ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                    dropout_rate=cfg.MODEL.CON_DROPOUT_RATE,
                    test_noact=cfg.TEST.NO_ACT,
                    con_last_dim=cfg.MODEL.CON_LAST_DIM
                )
                self.model_pairs.append(
                    [self.head_con, self.head_con_m]
                )
            self.temp = cfg.MODEL.TEMP
        """
        TODO from Jan. 20th, 2022:
        What if pred-logits sup-consistency loss with contrastive? 
        """
        if self.queue_size > -1:
            # create the queue
            if self.feat:
                if self.contrastive:
                    # feature contrastive loss
                    self.register_buffer("tpv_feat_queue", torch.randn(cfg.MODEL.CON_LAST_DIM, self.queue_size))
                    self.tpv_feat_queue = nn.functional.normalize(self.tpv_feat_queue, dim=0)
                    self.tpv_label_queue = nn.functional.normalize(self.tpv_label_queue, dim=0)
                else:
                    # feature positive consistency loss
                    self.register_buffer("tpv_pos_feat_queue", torch.randn(2048, self.queue_size))
                    self.tpv_pos_feat_queue = nn.functional.normalize(self.tpv_pos_feat_queue, dim=0)
            else:
                self.register_buffer("tpv_pred_queue", torch.zeros(cfg.MODEL.NUM_CLASSES, self.queue_size))
                self.register_buffer("tpv_label_queue", torch.zeros(cfg.MODEL.NUM_CLASSES, self.queue_size))
                if self.contrastive:
                    self.register_buffer("tpv_pred_norm_queue", torch.zeros(cfg.MODEL.NUM_CLASSES, self.queue_size))
                    self.tpv_pred_norm_queue = nn.functional.normalize(self.tpv_pred_norm_queue, dim=0)
                    self.tpv_label_queue = nn.functional.normalize(self.tpv_label_queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def freeze_param(self, freeze_stages):
        assert freeze_stages <= 5
        stages = [self.s1, self.s2, self.s3, self.s4, self.s5]
        for stage in stages[:freeze_stages]:
            for p in stage.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)


    @torch.no_grad()
    def _dequeue_and_enqueue_logits(self, tpv_pred, tpv_label):
        """logits queue: for prediction logits positive consistency loss"""
        # gather keys before updating queue
        tpv_preds = concat_all_gather(tpv_pred)
        tpv_labels = concat_all_gather(tpv_label)

        batch_size = tpv_pred.shape[0] * self.num_gpus
        # print(batch_size)

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.tpv_pred_queue[:, ptr:ptr + batch_size] = tpv_preds.T
        self.tpv_label_queue[:, ptr:ptr + batch_size] = tpv_labels.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_logits_norm_labels(self, tpv_pred, tpv_pred_norm, tpv_label_norm):
        """logits queue: for prediction logits contrastive & positive consistency loss"""
        # gather keys before updating queue
        tpv_preds = concat_all_gather(tpv_pred)
        tpv_pred_norms = concat_all_gather(tpv_pred_norm)
        tpv_label_norms = concat_all_gather(tpv_label_norm)

        batch_size = tpv_pred.shape[0] * self.num_gpus
        # print(batch_size)

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.tpv_pred_queue[:, ptr:ptr + batch_size] = tpv_preds.T
        self.tpv_pred_norm_queue[:, ptr:ptr + batch_size] = tpv_pred_norms.T
        self.tpv_label_queue[:, ptr:ptr + batch_size] = tpv_label_norms.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_2(self, tpv_feat, tpv_pred, tpv_label):
        # print('yes-2!')
        # gather keys before updating queue
        tpv_feats = concat_all_gather(tpv_feat)
        tpv_preds = concat_all_gather(tpv_pred)
        tpv_labels = concat_all_gather(tpv_label)

        batch_size = tpv_pred.shape[0] * self.num_gpus
        # print(batch_size)

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.tpv_feat_queue[:, ptr:ptr + batch_size] = tpv_feats.T
        self.tpv_pred_queue[:, ptr:ptr + batch_size] = tpv_preds.T
        self.tpv_label_queue[:, ptr:ptr + batch_size] = tpv_labels.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_feat(self, tpv_feat):
        """feat queue: for feature positive consistency loss"""
        # gather keys before updating queue
        tpv_feats = concat_all_gather(tpv_feat)

        batch_size = tpv_feat.shape[0] * self.num_gpus

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.tpv_pos_feat_queue[:, ptr:ptr + batch_size] = tpv_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def _get_mean_teacher_positive_supervised_loss(self, fpv_preds, fpv_labels):
        """
        Qitong on Jan. 7th, 2022.

        Args:
            fpv_preds: predictions of FPVs, calculating consistency loss.
            fpv_labels: labels of FPVs, finds the FPV & TPV pairs with the same class.

        Returns: positive consistency loss

        """

        tpv_pred_all = self.tpv_pred_queue.clone().detach()
        tpv_label_all = self.tpv_label_queue.clone().detach()
        M_fpv = torch.stack([fpv_label.unsqueeze(0).expand_as(tpv_label_all.T) for fpv_label in fpv_labels])
        idx_matrixs = (~(M_fpv.type(torch.BoolTensor) ^ tpv_label_all.T.unsqueeze(0).expand_as(M_fpv)
                         .type(torch.BoolTensor))).to(fpv_preds.device)
        loss_list = torch.stack([torch.sum((tpv_pred_all.T - fpv_pred.expand_as(tpv_pred_all.T) * idx_m).pow(2))
                                 for fpv_pred, idx_m in zip(fpv_preds, idx_matrixs)])
        sum_list = torch.stack([torch.sum(idx_m) for idx_m in idx_matrixs]) + 1e-20
        return (loss_list / sum_list).mean()

    def _get_mean_teacher_positive_knn_loss(self, fpv_preds):
        """
        Qitong on Jan. 7th, 2022.

        Args:
            fpv_preds: predictions of FPVs, calculating consistency loss.

        Returns: positive consistency loss

        """
        # print('yes')
        tpv_pred_all = self.tpv_pred_queue.clone().detach()
        loss = 0.0
        for fpv_pred in fpv_preds:
            # dist = torch.stack([F.kl_div(F.log_softmax(tpv_pred.unsqueeze(0), dim=1),
            #                              F.softmax(fpv_pred.unsqueeze(0), dim=1), reduction='sum')
            #                     for tpv_pred in tpv_pred_all.T], dim=0)
            # dist = torch.stack([F.kl_div(F.log_softmax(fpv_pred.unsqueeze(0), dim=1),
            #                              F.softmax(tpv_pred.unsqueeze(0), dim=1), reduction='sum')
            #                     for tpv_pred in tpv_pred_all.T], dim=0)
            dist = torch.stack([(fpv_pred - tpv_pred).pow(2).mean()
                                for tpv_pred in tpv_pred_all.T], dim=0)
            knn = dist.topk(self.pos_consistency_K, largest=False)

            dist_temp = torch.stack([F.cosine_similarity(fpv_pred.unsqueeze(0), tpv_pred.unsqueeze(0))
                                     for tpv_pred in tpv_pred_all.T], dim=0).squeeze()
            # print(dist_temp.size(), dist.size())
            knn1 = dist_temp.topk(self.pos_consistency_K, largest=True)
            knn2 = dist_temp.topk(self.neg_consistency_N, largest=False)
            print(knn1.values)
            print(-torch.log(knn1.values.sum() / (knn1.values.sum() + knn2.values.sum() + 1e-20)))
            print(knn2.values)
            if self.KNN_D:
                loss += (torch.exp(-0.2 * knn.values.clone().detach() ** 2) * knn.values).mean()
                # print(233)
            else:
                # print(self.rank_list)
                # loss += (knn.values * self.rank_list.to(fpv_preds.device)).mean()
                loss += knn.values.mean()

        return loss / self.batch_size

    def _get_mean_teacher_positive_knn_feat_loss(self, fpv_feats):
        """
        Qitong on Jan. 18th, 2022.

        Args:
            fpv_feats: features of FPVs, calculating consistency loss.

        Returns: positive feature consistency loss

        """
        # print('yes feat!')
        # print(fpv_feats[0].size())
        tpv_feat_all = self.tpv_pos_feat_queue.clone().detach()
        loss = 0.0
        for fpv_feat in fpv_feats:
            if self.neg_consistency_N > -1:
                dist = torch.stack([F.cosine_similarity(fpv_feat.unsqueeze(0), tpv_feat.unsqueeze(0))
                                         for tpv_feat in tpv_feat_all.T], dim=0).squeeze()
                knn_pos = dist.topk(self.pos_consistency_K, largest=True)
                knn_neg = dist.topk(self.neg_consistency_N, largest=False)
                loss += -torch.log(knn_pos.values.sum() / (knn_pos.values.sum() + knn_neg.values.sum() + 1e-20))
            else:
                # dist = torch.stack([F.kl_div(F.log_softmax(tpv_feat.unsqueeze(0), dim=1),
                #                              F.softmax(fpv_feat.unsqueeze(0), dim=1), reduction='sum')
                #                     for tpv_feat in tpv_feat_all.T], dim=0)
                dist = torch.stack([(fpv_feat - tpv_feat).pow(2).mean()
                                    for tpv_feat in tpv_feat_all.T], dim=0)
                knn = dist.topk(self.pos_consistency_K, largest=False)
                if self.KNN_D:
                    loss += (torch.exp(-0.2 * knn.values.clone().detach() ** 2) * knn.values).mean()
                else:
                    # loss += (knn.values * self.rank_list.to(fpv_preds.device)).mean()
                    loss += knn.values.mean()

        return loss / self.batch_size

    def _get_multilabel_contrastive_loss(self, fpv_feat, tpv_feat, labels_norm):
        """
        Qitong on Jan. 17th, 2022.

        Args: ... ...

        Returns: multilabel contrastive loss

        Previous Notes on Jan. 5th, 2022.
        SLOW Backbone only!
        """
        tpv_feat_all = torch.cat([tpv_feat.T, self.tpv_feat_queue.clone().detach()], dim=1)
        sim_preds = fpv_feat @ tpv_feat_all / self.temp
        # labels_norm = labels / (labels.norm(dim=1)[:, None] + 1e-20)
        labels_all = torch.cat([labels_norm.T, self.tpv_label_queue.clone().detach()], dim=1)
        # sim_targets = torch.mm(labels_norm, labels_norm.transpose(0, 1))
        sim_targets = labels_norm @ labels_all
        # print(sim_targets)
        # sim_targets.fill_diagonal_(1)
        # assert torch.min(sim_targets) >= 0.0 and torch.max(sim_targets) <= 1.0
        loss_contrastive = -torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        # loss_contrastive = torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        # print(labels.size(), labels_all.size(), sim_targets.size(), torch.min(sim_targets), torch.max(sim_targets))
        return loss_contrastive

    def _get_multilabel_logits_contrastive_loss(self, fpv_pred_norm, tpv_pred_norm, labels_norm):
        """
        Qitong on Jan. 20th, 2022.

        Args: ... ...

        Returns: multilabel prediction logits contrastive loss

        Previous Notes on Jan. 5th, 2022.
        SLOW Backbone only!
        """
        tpv_pred_norm_all = torch.cat([tpv_pred_norm.T, self.tpv_pred_queue.clone().detach()], dim=1)
        sim_preds = fpv_pred_norm @ tpv_pred_norm_all
        labels_all = torch.cat([labels_norm.T, self.tpv_label_queue.clone().detach()], dim=1)
        sim_targets = labels_norm @ labels_all
        if self.con_logit_l2:
            loss_contrastive = (sim_preds - sim_targets).pow(2).mean()
        else:
            loss_contrastive = -torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        # loss_contrastive = torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        # print(labels.size(), labels_all.size(), sim_targets.size(), torch.min(sim_targets), torch.max(sim_targets))
        return loss_contrastive

    def forward(self, x, cur_epoch, labels=None, bboxes=None):
        if self.training:
            """FPV predictions"""
            fpv_x = [x[0][:int(self.batch_size / 2)]]
            tpv_x = [x[0][int(self.batch_size / 2):]]
            fpv_x = self.s1(fpv_x)
            fpv_x = self.s2(fpv_x)
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                fpv_x[pathway] = pool(fpv_x[pathway])
            fpv_x = self.s3(fpv_x)
            fpv_x = self.s4(fpv_x)
            fpv_x = self.s5(fpv_x)
            head = getattr(self, self.head_name)
            if self.enable_detection:
                fpv_pred, _ = head(fpv_x, bboxes)
            else:
                fpv_pred, _ = head(fpv_x)

            with torch.no_grad():
                """Momentum & TPV predictions"""
                self._momentum_update()
                tpv_x = self.s1_m(tpv_x)
                tpv_x = self.s2_m(tpv_x)
                for pathway in range(self.num_pathways):
                    pool_m = getattr(self, "pathway{}_pool_m".format(pathway))
                    tpv_x[pathway] = pool_m(tpv_x[pathway])
                tpv_x = self.s3_m(tpv_x)
                tpv_x = self.s4_m(tpv_x)
                tpv_x = self.s5_m(tpv_x)
                head_m = getattr(self, self.head_name_m)
                if self.enable_detection:
                    tpv_pred, _ = head_m(tpv_x, bboxes)
                else:
                    tpv_pred, _ = head_m(tpv_x)

            """epoch schedule function"""
            epoch_ele = float(cur_epoch) / self.max_epoch
            epoch_func = math.exp(-5 * (1 - epoch_ele) ** 2)

            """general mean-teacher loss"""
            loss = {}
            if self.disKL_mode:
                # KL mode
                loss_consistency = epoch_func * \
                                   F.kl_div(F.log_softmax(tpv_pred, dim=1), F.softmax(fpv_pred, dim=1), reduction='sum')
            else:
                # L2 distance mode
                loss_consistency = epoch_func * (fpv_pred - tpv_pred).pow(2).mean()
            loss['mean-con'] = loss_consistency * self.lambda_mt

            """contrastive loss"""
            if self.contrastive:
                if self.feat:
                    # feature-based contrastive
                    fpv_x_norm = F.normalize(self.head_con(fpv_x).squeeze(), dim=-1)
                    tpv_x_norm = F.normalize(self.head_con_m(tpv_x).squeeze(), dim=-1)
                    labels_norm = F.normalize(labels, dim=-1)
                    loss_contrastive = self._get_multilabel_contrastive_loss(fpv_x_norm, tpv_x_norm, labels_norm)
                    loss['feat_contrastive'] = epoch_func * loss_contrastive * self.lambda_con
                else:
                    # prediction logits contrastive
                    fpv_pred_norm = F.normalize(fpv_pred, dim=-1)
                    tpv_pred_norm = F.normalize(tpv_pred, dim=1)
                    labels_norm = F.normalize(labels, dim=-1)
                    loss_contrastive = self._get_multilabel_logits_contrastive_loss(
                        fpv_pred_norm, tpv_pred_norm, labels_norm)
                    if self.con_logit_l2:
                        loss['l2_logits_contrastive'] = epoch_func * loss_contrastive * self.lambda_con
                    else:
                        loss['ylog(x)_logits_contrastive'] = epoch_func * loss_contrastive * self.lambda_con


            """positive consistency loss"""
            if self.queue_size > -1:
                if self.pos_consistency_K > -1:
                    # KNN based loss
                    if self.feat:
                        # loss['feat_knn_pos_consistency'] = epoch_func * self._get_mean_teacher_positive_knn_feat_loss(
                        #     F.normalize(fpv_x[0], dim=1)) * self.lambda_pos
                        fpv_feat = F.adaptive_avg_pool3d(fpv_x[0], (1, 1, 1)).squeeze()
                        loss_temp = epoch_func * self._get_mean_teacher_positive_knn_feat_loss(
                            F.normalize(fpv_feat, dim=-1)) * self.lambda_pos
                        if not math.isnan(loss_temp) and loss_temp > 0.0:
                            if self.neg_consistency_N > -1:
                                loss['feat_Qcon_knn_pos_consistency'] = loss_temp
                            else:
                                loss['feat_knn_pos_consistency'] = loss_temp
                    else:
                        loss['logits_knn_pos_consistency'] = epoch_func * \
                                           self._get_mean_teacher_positive_knn_loss(fpv_pred) * self.lambda_pos
                else:
                    assert not self.feat, "Supervised positive consistency loss does not support feature operations!!!"
                    # supervised loss
                    loss['logits_sup_pos_consistency'] = epoch_func * self._get_mean_teacher_positive_supervised_loss(
                        fpv_pred, labels) * self.lambda_pos

                """queue operations"""
                if self.feat:
                    if self.contrastive and self.queue_size > -1:
                        # TODO: feature contrastive & positive consistency loss
                        pass
                    elif self.queue_size > -1:
                        # feature positive consistency loss only
                        tpv_feat = F.adaptive_avg_pool3d(tpv_x[0], (1, 1, 1)).squeeze()
                        self._dequeue_and_enqueue_feat(F.normalize(tpv_feat, dim=1))
                        # print(233)
                else:
                    if self.contrastive and self.queue_size > -1:
                        # prediction logits contrastive & positive consistency loss
                        self._dequeue_and_enqueue_logits_norm_labels(
                            tpv_pred, F.normalize(tpv_pred, dim=-1), F.normalize(labels, dim=-1))
                    elif self.queue_size > -1:
                        # feature positive consistency loss only
                        self._dequeue_and_enqueue_logits(tpv_pred, labels)
                # elif self.contrastive and self.queue_size > -1:
                #     self._dequeue_and_enqueue_2(tpv_x_norm, tpv_pred, F.normalize(labels, dim=-1))
                # for k, v in loss.items():
                #     print(k, v)
                # print(fpv_feat.size(), self.tpv_pos_feat_queue.size())
        else:
            """inference, with FPV predictions only"""
            x = self.s1(x)
            x = self.s2(x)
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x[pathway] = pool(x[pathway])
            x = self.s3(x)
            x = self.s4(x)
            x = self.s5(x)

            head = getattr(self, self.head_name)
            if self.enable_detection:
                fpv_pred, loss = head(x, bboxes)
            else:
                fpv_pred, loss = head(x)

        return fpv_pred, loss

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output