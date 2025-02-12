#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm

from slowfast.models import head_helper, resnet_helper, stem_helper
from slowfast.models.build import MODEL_REGISTRY
from .kd_models import MultiTaskHead

from .video_model_builder import FuseFastToSlow
from slowfast.models.my_head import *

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


@MODEL_REGISTRY.register()
class SlowFast_LEMMA3S_PHRASE_EP_HOI_2S(nn.Module):
    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast_LEMMA3S_PHRASE_EP_HOI_2S, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
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

        self.s1_t = stem_helper.VideoModelStem(
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
        self.s1_fuse_t = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2_t = resnet_helper.ResStage(
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
        self.s2_fuse_t = FuseFastToSlow(
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

        self.s3_t = resnet_helper.ResStage(
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
        self.s3_fuse_t = FuseFastToSlow(
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

        self.s4_t = resnet_helper.ResStage(
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
        self.s4_fuse_t = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5_t = resnet_helper.ResStage(
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


        head = MultiTaskHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            num_classes=cfg.MODEL.NUM_CLASSES_LIST,
            pool_size=[None, None],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            test_noact=cfg.TEST.NO_ACT,
        )
        self.head_name = "head{}".format(cfg.TASK)
        self.add_module(self.head_name, head)

        head = MultiTaskHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            num_classes=[25, 64],
            pool_size=[None, None],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func="sigmoid",
            test_noact=cfg.TEST.NO_ACT,
        )
        self.add_module("head_lemma", head)
        self.cls_loss_lemma = nn.BCEWithLogitsLoss()

        self.batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        self.num_gpus = cfg.NUM_GPUS
        self.max_epoch = cfg.SOLVER.MAX_EPOCH
        self.lambda_tpv_cls = cfg.MODEL.L_CLS
        self.contrastive = cfg.MODEL.CONTRASTIVE
        self.dcl_thres = cfg.MODEL.DCL_THRES
        self.dcl_sigma = cfg.MODEL.DCL_SIGMA
        if cfg.MODEL.CONTRASTIVE:
            self.contrastive = True
            self.global_con_head = SlowFastPhraseGlobalConHead_DCL(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                dim_out=cfg.MODEL.CON_LAST_DIM,
                pool_size=[None, None],
                temp_pred=cfg.MODEL.TEMP,
                temp_gt=0.07,
                thres=self.dcl_thres,
                sigma=self.dcl_sigma
            )
            self.lambda_con = cfg.MODEL.L_CON
            self.lambda_con_f = cfg.MODEL.L_CON_V
            self.lambda_con_t = cfg.MODEL.L_CON_N
        self.model_pairs = [
            [self.s1, self.s1_t],
            [self.s2, self.s2_t],
            [self.s3, self.s3_t],
            [self.s4, self.s4_t],
            [self.s5, self.s5_t],
            [self.s1_fuse, self.s1_fuse_t],
            [self.s2_fuse, self.s2_fuse_t],
            [self.s3_fuse, self.s3_fuse_t],
            [self.s4_fuse, self.s4_fuse_t],
        ]

    @torch.no_grad()
    def copy_s1_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                # param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def copy_lemma_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs_moco:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.moco_m + param.data * (1. - self.moco_m)

    def freeze_param(self, freeze_stages):
        assert freeze_stages <= 5
        stages = [self.s1, self.s2, self.s3, self.s4, self.s5]
        for stage in stages[:freeze_stages]:
            for p in stage.parameters():
                p.requires_grad = False

    def _transfer_label(self, label, num_class):
        res = torch.zeros(label.size()[0], num_class).to(label.device)
        for idx in range(label.size()[0]):
            res[idx][label[idx]] = 1
        return res

    def forward(self, x_fpv, x_tpv=None, cur_epoch=-1,
                labels=None, labels_verb=None, labels_noun=None,
                same_idx=None, meta=None, phrase_model=None):
        if self.training:
            print(x_fpv[0].equal(x_tpv[0]), 'in0')
            print(x_fpv[1].equal(x_tpv[1]), 'in1')
            x_fpv = self.s1(x_fpv)
            x_fpv = self.s1_fuse(x_fpv)
            x_fpv = self.s2(x_fpv)
            x_fpv = self.s2_fuse(x_fpv)
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x_fpv[pathway] = pool(x_fpv[pathway])
            x_fpv = self.s3(x_fpv)
            x_fpv = self.s3_fuse(x_fpv)
            x_fpv = self.s4(x_fpv)
            x_fpv = self.s4_fuse(x_fpv)
            x_fpv = self.s5(x_fpv)

            # self._momentum_update()
            x_tpv = self.s1_t(x_tpv)
            x_tpv = self.s1_fuse_t(x_tpv)
            x_tpv = self.s2_t(x_tpv)
            x_tpv = self.s2_fuse_t(x_tpv)
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x_tpv[pathway] = pool(x_tpv[pathway])
            x_tpv = self.s3_t(x_tpv)
            x_tpv = self.s3_fuse_t(x_tpv)
            x_tpv = self.s4_t(x_tpv)
            x_tpv = self.s4_fuse_t(x_tpv)
            x_tpv = self.s5_t(x_tpv)

            feat_fpv = [x_fpv[0], x_fpv[1]]
            feat_tpv = [x_tpv[0], x_tpv[1]]
            print(x_fpv[0].size(), x_fpv[1].size(), x_tpv[0].size(), x_tpv[1].size())

            if self.contrastive:
                feat_fpv_ori = [feat_fpv[0].clone(), feat_fpv[1].clone()]
                feat_tpv_ori = [feat_tpv[0].clone(), feat_tpv[1].clone()]

            head = getattr(self, self.head_name)
            pred_fpv, _ = head(feat_fpv)

            my_loss = {}
            pred_tpv, _ = self.head_lemma(feat_tpv)
            print(pred_tpv[0].size(), pred_tpv[1].size(), labels_verb.size(), labels_noun.size())
            loss_cls_pred_tpv = self.cls_loss_lemma(pred_tpv[0], labels_verb) + \
                                self.cls_loss_lemma(pred_tpv[1], labels_noun)
            my_loss['loss_cls_pred_tpv'] = loss_cls_pred_tpv * self.lambda_tpv_cls

            """phrase"""
            label_matrix = F.normalize(torch.tensor(phrase_model.encode(meta["fpv_narration"])), dim=1) @\
                           F.normalize(torch.tensor(phrase_model.encode(meta["tpv_narration"])), dim=1).T
            label_matrix = label_matrix.cuda()
            label_matrix_fpv = F.normalize(torch.tensor(phrase_model.encode(meta["fpv_narration"])), dim=1) @ \
                               F.normalize(torch.tensor(phrase_model.encode(meta["fpv_narration"])), dim=1).T
            label_matrix_fpv = label_matrix_fpv.cuda()
            label_matrix_tpv = F.normalize(torch.tensor(phrase_model.encode(meta["tpv_narration"])), dim=1) @ \
                               F.normalize(torch.tensor(phrase_model.encode(meta["tpv_narration"])), dim=1).T
            label_matrix_tpv = label_matrix_tpv.cuda()
            fpv_narrs = F.normalize(torch.tensor(phrase_model.encode(meta["fpv_narration"])), dim=1).cuda()
            tpv_narrs = F.normalize(torch.tensor(phrase_model.encode(meta["tpv_narration"])), dim=1).cuda()

            """alignment"""
            if self.contrastive:
                loss_group = self.global_con_head(feat_fpv_ori, feat_tpv_ori,
                                                  label_matrix, label_matrix_fpv, label_matrix_tpv,
                                                  fpv_narrs, tpv_narrs)
                loss_gcon, loss_gcon_f, loss_gcon_t = loss_group
                if loss_gcon is not None:
                    my_loss['loss-dcl-os1-dis-tpl0.1-ee'] = loss_gcon * self.lambda_con
                if loss_gcon_f is not None:
                    my_loss['loss-dcl1-dis-mm-fpv'] = loss_gcon_f * self.lambda_con_f
                if loss_gcon_t is not None:
                    my_loss['loss-dcl1-dis-mm-tpv'] = loss_gcon_t * self.lambda_con_t

            """vis. tool"""
            for k, v in my_loss.items():
                print(k, v.item())
            if self.contrastive:
                print(self.contrastive, self.lambda_tpv_cls, self.lambda_con,
                      self.lambda_con_f, self.lambda_con_t, self.dcl_thres)
            else:
                print(self.contrastive, self.lambda_tpv_cls)
            return pred_fpv, my_loss
        else:
            x = self.s1(x_fpv)
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
            x, loss = head(x)
            return x, loss


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


@torch.no_grad()
def concat_all_gather_list(list):
    tensors_gather = [[1 for i in range(len(list))]
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(tensors_gather, list)

    output = [i for item in tensors_gather for i in item]
    return output