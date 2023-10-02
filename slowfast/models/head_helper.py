#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn
# Qitong on Dec. 18th, 2021.
import torch.nn.functional as F
from detectron2.layers import ROIAlign
# Qitong on Jan. 3rd, 2022.
import math


class ResNetRoIHead(nn.Module):
    """
    ResNe(X)t RoI head.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, bboxes):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out = t_pool(inputs[pathway])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out = roi_align(out, bboxes)

            s_pool = getattr(self, "s{}_spool".format(pathway))
            pool_out.append(s_pool(out))

        # B C H W.
        x = torch.cat(pool_out, 1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.projection(x)
        x = self.act(x)
        return x

class ResNetCAMCONRoIHead(nn.Module):
    """
        Qitong on Apr. 14th, 2022.
    """

    def __init__(
        self,
        dim_in,
        con_last_dim,
        pool_size,
        resolution,
        scale_factor,
        dropout_rate=0.0,
        aligned=True,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetCAMCONRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), con_last_dim, bias=True)

    def forward(self, inputs, bboxes):
        res = []
        for idx in range(bboxes.size()[1]):
            bbox = bboxes[:, idx, :]
            assert (
                    len(inputs) == self.num_pathways
            ), "Input tensor does not contain {} pathway".format(self.num_pathways)
            pool_out = []
            for pathway in range(self.num_pathways):
                t_pool = getattr(self, "s{}_tpool".format(pathway))
                out = t_pool(inputs[pathway])
                assert out.shape[2] == 1
                out = torch.squeeze(out, 2)

                roi_align = getattr(self, "s{}_roi".format(pathway))
                out = roi_align(out, bbox)

                s_pool = getattr(self, "s{}_spool".format(pathway))
                pool_out.append(s_pool(out))

            # B C H W.
            x = torch.cat(pool_out, 1).squeeze()

            # Perform dropout.
            if hasattr(self, "dropout"):
                x = self.dropout(x)

            # x = x.view(x.shape[0], -1)
            # print(x.size())
            res.append(x.unsqueeze(1))
            # res.append(self.projection(x))
        # print(torch.cat(res, 1).size())
        return self.projection(torch.cat(res, dim=1))

class ResNetCHRRoIHead(nn.Module):
    """
    ResNe(X)t CHR RoI head.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True,
    ):
        """
            Qitong on Mar. 5th, 2022.
        """
        super(ResNetCHRRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, bboxes):
        res_tensors = []
        for idx in range(bboxes.size()[1]):
            bbox = bboxes[:, idx, :]
            assert (
                    len(inputs) == self.num_pathways
            ), "Input tensor does not contain {} pathway".format(self.num_pathways)
            pool_out = []
            for pathway in range(self.num_pathways):
                t_pool = getattr(self, "s{}_tpool".format(pathway))
                out = t_pool(inputs[pathway])
                assert out.shape[2] == 1
                out = torch.squeeze(out, 2)

                roi_align = getattr(self, "s{}_roi".format(pathway))
                out = roi_align(out, bbox)

                s_pool = getattr(self, "s{}_spool".format(pathway))
                pool_out.append(s_pool(out))

            res_tensors.append(torch.cat(pool_out, 1).squeeze().unsqueeze(1))
        return torch.cat(res_tensors, 1)


class ResNetCAMCHRRoIHead(nn.Module):
    """
    ResNe(X)t CAM CHR RoI head.
    """

    def __init__(
        self,
        dim_in,
        pool_size,
        resolution,
        scale_factor,
        aligned=True,
    ):
        """
            Qitong on Apr. 8th, 2022.
        """
        super(ResNetCAMCHRRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

    def forward(self, inputs, bboxes):
        res_tensors = []
        for idx in range(bboxes.size()[1]):
            bbox = bboxes[:, idx, :]
            assert (
                    len(inputs) == self.num_pathways
            ), "Input tensor does not contain {} pathway".format(self.num_pathways)
            pool_out = []
            for pathway in range(self.num_pathways):
                t_pool = getattr(self, "s{}_tpool".format(pathway))
                out = t_pool(inputs[pathway])
                assert out.shape[2] == 1
                out = torch.squeeze(out, 2)

                roi_align = getattr(self, "s{}_roi".format(pathway))
                out = roi_align(out, bbox)

                s_pool = getattr(self, "s{}_spool".format(pathway))
                pool_out.append(s_pool(out))

            res_tensors.append(torch.cat(pool_out, 1).squeeze().unsqueeze(1))
        return torch.cat(res_tensors, 1)


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        test_noact=False,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.test_noact = test_noact

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            # print(pathway)
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        # print(self.num_pathways)
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        # print(233, x.size())
        x = self.projection(x)

        # Performs fully convolutional inference.
        if not self.training:
            if not self.test_noact:
                x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x, {}

class ResNetBasicHead_LateCon(nn.Module):
    """
        Qitong on Mar. 19th, 2021.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        num_gpus,
        lambda_con,
        dropout_rate=0.0,
        act_func="softmax",
        test_noact=False,
        con_last_dim=768,
        temp=0.07
    ):
        super(ResNetBasicHead_LateCon, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.test_noact = test_noact
        self.num_gpus = num_gpus
        self.lambda_con = lambda_con
        self.temp = temp

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)
        self.con_proj_fpv = nn.Linear(sum(dim_in), con_last_dim, bias=True)
        self.con_proj_tpv = nn.Linear(sum(dim_in), con_last_dim, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def _get_multilabel_kl_loss(self, fpv_feat, tpv_feat, label):
        """
        Qitong on Feb. 24th, 2022.
        """
        sim_preds = fpv_feat @ tpv_feat.T / self.temp
        sim_targets = label @ label.T
        # print('feat', F.softmax(sim_preds))
        loss_kl = F.kl_div(F.log_softmax(sim_preds.double(), dim=1),
                           F.softmax(sim_targets.double(), dim=1), reduction='sum')
        return loss_kl

    def forward(self, inputs, labels=None, epoch_func=-1.0):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            # print(pathway)
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        # print(self.num_pathways)
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x_ = self.dropout(x)
            x_ = self.projection(x_)
        else:
            x_ = self.projection(x)

        loss = {}
        # Late multi-view alignment
        if self.training:
            x = x.squeeze()
            mini_b = int(x.size()[0] / self.num_gpus)
            # print(mini_b, x.size())
            fpv_x, tpv_x = x[:mini_b], x[mini_b:]
            fpv_x_pre_norm = self.con_proj_fpv(fpv_x)
            fpv_x_norm = F.normalize(fpv_x_pre_norm, dim=-1)
            tpv_x_pre_norm = self.con_proj_tpv(tpv_x)
            tpv_x_norm = F.normalize(tpv_x_pre_norm, dim=-1)
            labels_norm = F.normalize(labels, p=1, dim=1)
            # print(fpv_x_norm.size(), tpv_x_norm.size(), labels_norm.size())
            loss_kl = self._get_multilabel_kl_loss(fpv_x_norm, tpv_x_norm, labels_norm)
            loss['kl-multilabel-contrastive-p1label-late-fusion'] = epoch_func * loss_kl * self.lambda_con
            for k, v in loss.items():
                print(k, v.item(), self.lambda_con, epoch_func, self.temp, 'late')

        # Performs fully convolutional inference.
        if not self.training:
            if not self.test_noact:
                x_ = self.act(x_)
            x_ = x_.mean([1, 2, 3])

        x_ = x_.view(x_.shape[0], -1)
        return x_, loss

class ResNetActHead(nn.Module):
    """
        Qitong on Mar. 19th, 2022.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        test_noact=False,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetActHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.test_noact = test_noact

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        x = torch.cat(inputs, 1)
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        # print(233, x.size())
        x = self.projection(x)

        # Performs fully convolutional inference.
        if not self.training:
            if not self.test_noact:
                x = self.act(x)
        x = x.view(x.shape[0], -1)
        return x, {}

class ResNetPiHead(nn.Module):
    """
    Qitong on Jan. 3rd, 2022.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        batch_size,
        dis_mode,
        dropout_rate=0.0,
        act_func="softmax",
        test_noact=False,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetPiHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.test_noact = test_noact

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)
        self.batch_size = batch_size
        self.disKL_mode = dis_mode

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, epoch_ele):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        # print(233, x.size())
        x = self.projection(x)

        # Performs fully convolutional inference.
        if not self.training:
            if not self.test_noact:
                x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        # PI model consistency
        if self.training:
            if self.disKL_mode:
                fpv_x = x[:int(self.batch_size / 2)].squeeze()
                tpv_x = x[int(self.batch_size / 2):].squeeze()
                loss_consistency = math.exp(-5 * (1 - epoch_ele) ** 2) * \
                                   F.kl_div(F.log_softmax(fpv_x, dim=1), F.softmax(tpv_x, dim=1), reduction='sum')
                # print(fpv_x.size(), tpv_x.size(), loss_consistency)
            else:
                fpv_x = x[:int(self.batch_size / 2)]
                tpv_x = x[int(self.batch_size / 2):]
                loss_consistency = math.exp(-5 * (1 - epoch_ele) ** 2) * (fpv_x - tpv_x).pow(2).mean()
        return_loss = {}
        if self.training:
            return_loss['pi-con'] = loss_consistency * 0.1
            # print(233)
        return x, return_loss

class ResNetContrastiveHead(nn.Module):
    """
    Qitong Wang on Dec. 18th, 2021.
    """
    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        batch_size,
        temp,
        dropout_rate=0.0,
        act_func="softmax",
        test_noact=False,
    ):
        super(ResNetContrastiveHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.test_noact = test_noact

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        # contrastive learning
        self.batch_size = batch_size
        self.temp = temp
        # self.temp = nn.Parameter(torch.ones([]) * temp)
        self.self_dim = 1024
        self.self_dim_2 = 256
        self.out_dim = 64
        self.con_proj_fpv = nn.Sequential(
            nn.Linear(sum(dim_in), self.self_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.self_dim, self.self_dim_2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.self_dim_2, self.out_dim, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Linear(self.self_dim, self.self_dim, bias=True),
        )
        self.con_proj_tpv = nn.Sequential(
            nn.Linear(sum(dim_in), self.self_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.self_dim, self.self_dim_2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.self_dim_2, self.out_dim, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Linear(self.self_dim, self.self_dim, bias=True),
        )
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        # print(inputs[0].get_device())
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        # if hasattr(self, "dropout"):
        #     x = self.dropout(x)
        # Perform contrastive learning
        if self.training:
            # x_ = self.con_proj(x)
            # fpv_x = F.normalize(x_[:int(self.batch_size / 2)].squeeze(), dim=-1)
            # fpv_x = F.normalize(self.fpv_mlp(x_[:int(self.batch_size / 2)].squeeze()), dim=-1)
            # tpv_x = F.normalize(x_[int(self.batch_size / 2):].squeeze(), dim=-1)
            # divide mlp
            fpv_x = F.normalize(self.con_proj_fpv(x[:int(self.batch_size / 2)].squeeze()), dim=-1)
            tpv_x = F.normalize(self.con_proj_tpv(x[int(self.batch_size / 2):].squeeze()), dim=-1)
            # print(233333)
            sim_preds = fpv_x @ tpv_x.T / self.temp
            sim_targets = torch.zeros(sim_preds.size()).to(x.device)
            sim_targets.fill_diagonal_(1)
            loss_con = -torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        x = self.projection(x)

        # Performs fully convolutional inference.
        if not self.training:
            if not self.test_noact:
                x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return_loss = {}
        if self.training:
            # print(loss_con)
            return_loss['con'] = loss_con * 0.05
        return x, return_loss

class ResNetTimeContrastiveHead(nn.Module):
    """
    Qitong Wang on Apr. 19th, 2022.
    """
    def __init__(
        self,
        dim_in,
        pool_size,
        dropout_rate=0.0,
        test_noact=False,
        con_last_dim=768,
        time_length=8
    ):
        super(ResNetTimeContrastiveHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.test_noact = test_noact

        for pathway in range(self.num_pathways):
            avg_pool = nn.AdaptiveAvgPool3d((time_length, 1, 1))
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        # contrastive learning
        self.self_dim = 768
        self.con_proj = nn.Sequential(
            nn.Linear(sum(dim_in), con_last_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(self.self_dim, con_last_dim),
        )
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        # print(inputs[0].size(), len(inputs))
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # print(x.size(), 'before')
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1)).squeeze()
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        # Perform projection
        x = self.con_proj(x)
        # print(x.size(), 'after')
        return x

class ResNetMeanTeacherContrastiveHead(nn.Module):
    """
    Qitong Wang on Jan. 5th, 2022.
    """
    def __init__(
        self,
        dim_in,
        pool_size,
        dropout_rate=0.0,
        test_noact=False,
        con_last_dim=768,
    ):
        super(ResNetMeanTeacherContrastiveHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.test_noact = test_noact

        for pathway in range(self.num_pathways):
            avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        # contrastive learning
        self.self_dim = 768
        self.con_proj = nn.Sequential(
            nn.Linear(sum(dim_in), con_last_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(self.self_dim, con_last_dim),
        )
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        # print(self.num_pathways, len(inputs))
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1)).squeeze()
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        # Perform projection
        x = self.con_proj(x)
        return x

class ResNetMeanTeacherContrastiveHead_MP(nn.Module):
    """
    Qitong Wang on Mar. 21st, 2022.
    """
    def __init__(
        self,
        dim_in,
        pool_size,
        dropout_rate=0.0,
        test_noact=False,
        con_last_dim=768,
    ):
        super(ResNetMeanTeacherContrastiveHead_MP, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.test_noact = test_noact

        for pathway in range(self.num_pathways):
            max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
            self.add_module("pathway{}_maxpool".format(pathway), max_pool)

        # contrastive learning
        self.self_dim = 768
        self.con_proj = nn.Sequential(
            nn.Linear(sum(dim_in), con_last_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(self.self_dim, con_last_dim),
        )
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        # print(self.num_pathways, len(inputs))
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_maxpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1)).squeeze()
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        # Perform projection
        x = self.con_proj(x)
        return x

class ResNetMeanTeacherContrastiveHead_NoPool(nn.Module):
    """
    Qitong Wang on Feb. 24th, 2022.
    """
    def __init__(
        self,
        dim_in,
        pool_size,
        dropout_rate=0.0,
        test_noact=False,
        con_last_dim=768,
    ):
        super(ResNetMeanTeacherContrastiveHead_NoPool, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.test_noact = test_noact

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        # contrastive learning
        # self.self_dim = 512
        self.con_proj = nn.Sequential(
            nn.Linear(sum(dim_in), con_last_dim),
            nn.ReLU(inplace=True),
            nn.Linear(con_last_dim, con_last_dim)
        )
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        # Perform projection
        x = self.con_proj(x)
        return x

class ResNetUnsupervisedContrastiveHead(nn.Module):
    """
    Qitong Wang on Jan. 1st, 2022.
    """
    def __init__(
        self,
        dim_in,
        pool_size,
        batch_size,
        temp,
        test_noact=False,
    ):
        super(ResNetUnsupervisedContrastiveHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.test_noact = test_noact

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        # contrastive learning
        self.batch_size = batch_size
        self.temp = temp
        # self.temp = nn.Parameter(torch.ones([]) * temp)
        self.self_dim = 1024
        self.out_dim = 768
        self.con_proj_fpv = nn.Sequential(
            nn.Linear(sum(dim_in), self.self_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.self_dim, self.out_dim, bias=True),
        )
        self.con_proj_tpv = nn.Sequential(
            nn.Linear(sum(dim_in), self.self_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.self_dim, self.out_dim, bias=True),
        )

    def forward(self, inputs):
        # print(inputs[0].get_device())
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform contrastive learning
        if self.training:
            # x_ = self.con_proj(x)
            # fpv_x = F.normalize(x_[:int(self.batch_size / 2)].squeeze(), dim=-1)
            # # fpv_x = F.normalize(self.fpv_mlp(x_[:int(self.batch_size / 2)].squeeze()), dim=-1)
            # tpv_x = F.normalize(x_[int(self.batch_size / 2):].squeeze(), dim=-1)
            # divide mlp
            fpv_x = F.normalize(self.con_proj_fpv(x[:int(self.batch_size / 2)].squeeze()), dim=-1)
            tpv_x = F.normalize(self.con_proj_tpv(x[int(self.batch_size / 2):].squeeze()), dim=-1)
            sim_preds = fpv_x @ tpv_x.T / self.temp
            sim_targets = torch.zeros(sim_preds.size()).to(x.device)
            sim_targets.fill_diagonal_(1)
            loss_con = -torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        return_loss = {}
        if self.training:
            # print(loss_con)
            return_loss['con'] = loss_con
        return None, return_loss

class ResNetContrastiveAttHead(nn.Module):
    """
    Qitong Wang on Dec. 18th, 2021.
    WARNING: slowfast not implemented!!
    """
    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        batch_size,
        temp,
        dropout_rate=0.0,
        act_func="softmax",
        test_noact=False,
    ):
        super(ResNetContrastiveAttHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.test_noact = test_noact

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        # contrastive learning
        self.batch_size = batch_size
        self.temp = temp
        self.self_dim = 512
        self.con_proj = nn.Sequential(
            nn.Linear(sum(dim_in), self.self_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(sum(dim_in), sum(dim_in)),
        )
        # attention
        self.att_mlp = nn.Sequential(
            nn.Linear(sum(dim_in), self.self_dim),
            nn.Linear(self.self_dim, self.self_dim),
        )
        if pool_size[pathway] is None:
            self.att_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.att_max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        else:
            self.att_avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.att_max_pool = nn.MaxPool3d(pool_size[pathway], stride=1)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        assert (
                len(inputs) == 1
        ), "Slowfast not implemented!!"
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform contrastive learning
        if self.training:
            att_x = F.relu(self.att_mlp(self.att_avg_pool(inputs[0]).squeeze())
                           + self.att_mlp(self.att_max_pool(inputs[0]).squeeze()))
            att_fpv_x = att_x[:int(self.batch_size / 2)]
            att_tpv_x = att_x[int(self.batch_size / 2):]
            x_ = self.con_proj(x)
            fpv_x = F.normalize(x_[:int(self.batch_size / 2)].squeeze() * att_fpv_x, dim=-1)
            tpv_x = F.normalize(x_[int(self.batch_size / 2):].squeeze() * att_tpv_x, dim=-1)
            sim_preds = fpv_x @ tpv_x.T / self.temp
            sim_targets = torch.zeros(sim_preds.size()).to(x.device)
            sim_targets.fill_diagonal_(1)
            loss_con = -torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
            # loss_att = torch.mean(torch.cosine_similarity(att_fpv_x, att_tpv_x))
            # loss_att = torch.sqrt(torch.mean((att_fpv_x - att_tpv_x) ** 2))
            loss_att = F.mse_loss(att_fpv_x, att_tpv_x, reduction='mean')
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convolutional inference.
        if not self.training:
            if not self.test_noact:
                x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return_loss = {}
        if self.training:
            return_loss['con'] = loss_con * 0.05
            return_loss['att'] = loss_att * 0.1
            # print(loss_con, loss_att)
        return x, return_loss