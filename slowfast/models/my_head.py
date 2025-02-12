#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from .CHR_Transformer_1 import CHR_Transformer_3
from .losses import *

class SlowFastSingleGlobalConHead(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        pool_size,
        temp,
        one_head
    ):
        super(SlowFastSingleGlobalConHead, self).__init__()

        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        self.one_head = one_head
        self.temp = temp
        if self.one_head:
            self.fpv_global_proj = nn.Linear(sum(dim_in), dim_out, bias=True)
            self.tpv_global_proj = nn.Linear(sum(dim_in), dim_out, bias=True)
        else:
            self.fpv_global_verb_proj = nn.Linear(sum(dim_in), dim_out, bias=True)
            self.tpv_global_verb_proj = nn.Linear(sum(dim_in), dim_out, bias=True)
            self.fpv_global_noun_proj = nn.Linear(sum(dim_in), dim_out, bias=True)
            self.tpv_global_noun_proj = nn.Linear(sum(dim_in), dim_out, bias=True)

    def _get_multilabel_kl_loss(self, fpv_feat, tpv_feat, label_v, label_n, label_lt_v, label_lt_n, same_idx):
        """
            Qitong on Aug. 7th, 2022.
        """
        sim_preds = fpv_feat @ tpv_feat.T / self.temp
        sim_labels = (label_v @ label_lt_v.T + label_n @ label_lt_n.T) / 2
        sim_targets = torch.zeros(sim_labels.size()).to(sim_labels.device)
        for i in range(sim_targets.size()[0]):
            if same_idx[i].item() == 2:
                sim_targets[i][i] = 1.0
            elif same_idx[i].item() == 1 or same_idx[i].item() == 3 or same_idx[i].item() == 4:
                pos_v = sim_labels[i][i].item()
                neg_v = (1.0 - pos_v) / (sim_targets.size()[0] - 1)
                for j in range(sim_targets.size()[0]):
                    sim_targets[i][j] = pos_v if i == j else neg_v
            elif same_idx[i].item() == 0:
                # uni. distribuition
                uni_v = 1.0 / sim_targets.size()[0]
                for j in range(sim_targets.size()[0]):
                    sim_targets[i][j] = uni_v
            else:
                raise NotImplementedError
        loss = -torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        return loss

    def _get_multilabel_div_kl_loss(self, fpv_feat, tpv_feat, label, label_lt, same_idx):
        """
            Qitong on Aug. 7th, 2022.
        """
        sim_preds = fpv_feat @ tpv_feat.T / self.temp
        sim_labels = label @ label_lt.T
        sim_targets = torch.zeros(sim_labels.size()).to(sim_labels.device)
        for i in range(sim_targets.size()[0]):
            if same_idx[i].item() == 2:
                sim_targets[i][i] = 1.0
            elif same_idx[i].item() == 1 or same_idx[i].item() == 3 or same_idx[i].item() == 4:
                pos_v = sim_labels[i][i].item()
                neg_v = (1.0 - pos_v) / (sim_targets.size()[0] - 1)
                for j in range(sim_targets.size()[0]):
                    sim_targets[i][j] = pos_v if i == j else neg_v
            elif same_idx[i].item() == 0:
                # uni. distribuition
                uni_v = 1.0 / sim_targets.size()[0]
                for j in range(sim_targets.size()[0]):
                    sim_targets[i][j] = uni_v
            else:
                raise NotImplementedError
        loss = -torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        return loss

    def forward(self, inputs_fpv, inputs_tpv,
                labels_norm_verb, labels_norm_noun,
                labels_norm_verb_lt, labels_norm_noun_lt,
                same_idx):
        assert (
            len(inputs_fpv) == self.num_pathways
        ), "FPVs tensor does not contain {} pathway".format(self.num_pathways)
        assert (
                len(inputs_tpv) == self.num_pathways
        ), "TPVs tensor does not contain {} pathway".format(self.num_pathways)
        pool_out_fpv = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            # print(inputs_fpv[pathway].shape, m(inputs_fpv[pathway]).shape, 'fpv')
            pool_out_fpv.append(m(inputs_fpv[pathway]))
        x_fpv = torch.cat(pool_out_fpv, 1)
        pool_out_tpv = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            # print(inputs_tpv[pathway].shape, m(inputs_tpv[pathway]).shape, 'tpv')
            pool_out_tpv.append(m(inputs_tpv[pathway]))
        x_tpv = torch.cat(pool_out_tpv, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).

        x_fpv = x_fpv.permute((0, 2, 3, 4, 1)).squeeze()
        x_tpv = x_tpv.permute((0, 2, 3, 4, 1)).squeeze()
        fpv_global_feat = F.normalize(x_fpv, dim=1)
        tpv_global_feat = F.normalize(x_tpv, dim=1)
        print(fpv_global_feat.size(), tpv_global_feat.size(), 233)
        if self.one_head:
            fpv_global_feat_pre_norm = self.fpv_global_proj(fpv_global_feat)
            fpv_global_feat_norm = F.normalize(fpv_global_feat_pre_norm, dim=1)
            tpv_global_feat_pre_norm = self.tpv_global_proj(tpv_global_feat)
            tpv_global_feat_norm = F.normalize(tpv_global_feat_pre_norm, dim=1)
            loss_ce = self._get_multilabel_kl_loss(
                fpv_global_feat_norm, tpv_global_feat_norm,
                labels_norm_verb, labels_norm_noun,
                labels_norm_verb_lt, labels_norm_noun_lt,
                same_idx=same_idx
            )
            return loss_ce, True
        else:
            fpv_global_feat_pre_norm_v = self.fpv_global_verb_proj(fpv_global_feat)
            fpv_global_feat_norm_v = F.normalize(fpv_global_feat_pre_norm_v, dim=1)
            tpv_global_feat_pre_norm_v = self.tpv_global_verb_proj(tpv_global_feat)
            tpv_global_feat_norm_v = F.normalize(tpv_global_feat_pre_norm_v, dim=1)
            fpv_global_feat_pre_norm_n = self.fpv_global_noun_proj(fpv_global_feat)
            fpv_global_feat_norm_n = F.normalize(fpv_global_feat_pre_norm_n, dim=1)
            tpv_global_feat_pre_norm_n = self.tpv_global_noun_proj(tpv_global_feat)
            tpv_global_feat_norm_n = F.normalize(tpv_global_feat_pre_norm_n, dim=1)
            loss_ce_v = self._get_multilabel_div_kl_loss(
                fpv_global_feat_norm_v, tpv_global_feat_norm_v,
                labels_norm_verb, labels_norm_verb_lt, same_idx,
            )
            loss_ce_n = self._get_multilabel_div_kl_loss(
                fpv_global_feat_norm_n, tpv_global_feat_norm_n,
                labels_norm_noun, labels_norm_noun_lt, same_idx,
            )
            return loss_ce_v + loss_ce_n, False

class SlowFastPhraseGlobalConHead(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        pool_size,
        temp_pred,
        temp_gt
    ):
        super(SlowFastPhraseGlobalConHead, self).__init__()

        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        self.temp_pred = temp_pred
        self.temp_gt = temp_gt
        self.fpv_global_proj = nn.Linear(sum(dim_in), dim_out, bias=True)
        self.tpv_global_proj = nn.Linear(sum(dim_in), dim_out, bias=True)

    def _get_phrase_loss(self, fpv_feat, tpv_feat, sim_targets_ori, l12):
        """
            Qitong on Aug. 7th, 2022.
        """
        sim_preds = fpv_feat @ tpv_feat.T
        if l12:
            loss = F.l1_loss(input=sim_preds, target=sim_targets_ori)
        else:
            loss = F.mse_loss(input=sim_preds, target=sim_targets_ori)
        return loss

    def _get_phrase_ce_loss(self, fpv_feat, tpv_feat, sim_targets_ori):
        """
            Qitong on Aug. 7th, 2022.
        """
        sim_preds = fpv_feat @ tpv_feat.T / self.temp_pred
        sim_targets = F.softmax(sim_targets_ori / self.temp_gt, dim=1)
        loss = -torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        return loss

    def forward(self, inputs_fpv, inputs_tpv, label_matrix):
        assert (
            len(inputs_fpv) == self.num_pathways
        ), "FPVs tensor does not contain {} pathway".format(self.num_pathways)
        assert (
                len(inputs_tpv) == self.num_pathways
        ), "TPVs tensor does not contain {} pathway".format(self.num_pathways)
        pool_out_fpv = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            # print(inputs_fpv[pathway].shape, m(inputs_fpv[pathway]).shape, 'fpv')
            pool_out_fpv.append(m(inputs_fpv[pathway]))
        x_fpv = torch.cat(pool_out_fpv, 1)
        pool_out_tpv = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            # print(inputs_tpv[pathway].shape, m(inputs_tpv[pathway]).shape, 'tpv')
            pool_out_tpv.append(m(inputs_tpv[pathway]))
        x_tpv = torch.cat(pool_out_tpv, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).

        x_fpv = x_fpv.permute((0, 2, 3, 4, 1)).squeeze()
        x_tpv = x_tpv.permute((0, 2, 3, 4, 1)).squeeze()
        fpv_global_feat = F.normalize(x_fpv, dim=1)
        tpv_global_feat = F.normalize(x_tpv, dim=1)
        fpv_global_feat_pre_norm = self.fpv_global_proj(fpv_global_feat)
        fpv_global_feat_norm = F.normalize(fpv_global_feat_pre_norm, dim=1)
        tpv_global_feat_pre_norm = self.tpv_global_proj(tpv_global_feat)
        tpv_global_feat_norm = F.normalize(tpv_global_feat_pre_norm, dim=1)
        loss_ce = self._get_phrase_ce_loss(fpv_global_feat_norm, tpv_global_feat_norm, label_matrix)
        return loss_ce

class SlowFastPhraseGlobalConHead_P(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        pool_size,
        temp_pred,
        temp_gt,
        thres=0.8
    ):
        super(SlowFastPhraseGlobalConHead_P, self).__init__()

        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        self.temp_pred = temp_pred
        self.temp_gt = temp_gt
        self.thres = thres
        self.fpv_global_proj = nn.Linear(sum(dim_in), dim_out, bias=True)
        self.tpv_global_proj = nn.Linear(sum(dim_in), dim_out, bias=True)

    def my_sigmoid(self, vec):
        # print(vec)
        return 1 / (1 + torch.exp(-3 * vec))

    def cos2preb(self, t):
        return t / 2 + 0.5

    def _get_phrase_loss(self, fpv_feat, tpv_feat, sim_targets_ori, l12):
        """
            Qitong on Aug. 7th, 2022.
        """
        sim_preds = fpv_feat @ tpv_feat.T
        # sim_preds = self.my_sigmoid(fpv_feat @ tpv_feat.T)
        if l12:
            loss = F.l1_loss(input=sim_preds, target=sim_targets_ori)
        else:
            loss = F.mse_loss(input=sim_preds, target=sim_targets_ori)
        return loss

    def _get_ce_loss(self, fpv_feat, tpv_feat, sim_targets_ori, temp_pred, temp_gt, bce=False, dig=False):
        """
            Qitong on Aug. 7th, 2022.
        """
        if bce:
            # print('gt', sim_targets_ori)
            sim_preds = fpv_feat @ tpv_feat.T
            if dig:
                loss = F.binary_cross_entropy(
                    input=self.my_sigmoid(torch.diagonal(sim_preds)),
                    target=self.cos2preb(torch.diagonal(sim_targets_ori))
                )
            else:
                loss = F.binary_cross_entropy(
                    input=self.my_sigmoid(sim_preds),
                    target=self.cos2preb(sim_targets_ori)
                )
        else:
            sim_preds = fpv_feat @ tpv_feat.T / temp_pred
            sim_targets = sim_targets_ori
            loss = -torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        return loss

    def _get_part_ce_loss(self, fpv_feat, tpv_feat, sim_targets_ori, temp_pred, temp_gt, thres):
        """
            Qitong on Aug. 7th, 2022.
        """

        batch_pesudo_pair_sim = torch.diagonal(sim_targets_ori)
        idx = torch.nonzero(
            torch.where(
                batch_pesudo_pair_sim > thres,
                torch.ones_like(batch_pesudo_pair_sim),
                torch.zeros_like(batch_pesudo_pair_sim)
            )
        ).squeeze()
        fpv_feat_slides = torch.index_select(fpv_feat, 0, idx)
        tpv_feat_slides = torch.index_select(tpv_feat, 0, idx)
        sim_preds_f = fpv_feat_slides @ tpv_feat.T / temp_pred
        sim_preds_t = tpv_feat_slides @ fpv_feat.T / temp_pred
        label_matrix_01 = torch.zeros(sim_preds_f.size()).to(sim_preds_f.device)
        label_matrix_01.fill_diagonal_(1)
        loss_f = -torch.sum(F.log_softmax(sim_preds_f, dim=1) * label_matrix_01, dim=1).mean()
        loss_t = -torch.sum(F.log_softmax(sim_preds_t, dim=1) * label_matrix_01, dim=1).mean()
        print(sim_preds_f.size(), sim_preds_t.size(), 'vanilla ce')
        if sim_preds_f.size()[0] == 0:
            return None
        else:
            return (loss_f + loss_t) / 2

    def _get_phrase_ce_loss(self, fpv_feat, tpv_feat, sim_targets_ori, temp_pred, temp_gt, bce=False, dig=False):
        """
            Qitong on Aug. 7th, 2022.
        """
        if bce:
            # print('gt', sim_targets_ori)
            sim_preds = fpv_feat @ tpv_feat.T
            # loss = F.binary_cross_entropy_with_logits(input=sim_preds, target=F.relu(sim_targets_ori))
            if dig:
                # loss = F.binary_cross_entropy(
                #     input=self.my_sigmoid(torch.diagonal(sim_preds)),
                #     target=self.cos2preb(torch.diagonal(sim_targets_ori))
                # )
                loss = F.binary_cross_entropy(
                    input=self.cos2preb(torch.diagonal(sim_preds)),
                    target=self.cos2preb(torch.diagonal(sim_targets_ori))
                )
            else:
                # loss = F.binary_cross_entropy(
                #     input=self.my_sigmoid(sim_preds),
                #     target=self.cos2preb(sim_targets_ori)
                # )
                loss = F.binary_cross_entropy(
                    input=self.cos2preb(sim_preds),
                    target=self.cos2preb(sim_targets_ori)
                )
        else:
            sim_preds = fpv_feat @ tpv_feat.T / temp_pred
            sim_targets = F.softmax(sim_targets_ori / temp_gt, dim=1)
            loss = -torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        return loss

    def _get_phrase_loss_od(self, fpv_feat, tpv_feat, sim_targets_ori, l12):
        """
            Qitong on Aug. 7th, 2022.
        """
        sim_preds = fpv_feat @ tpv_feat.T
        sim_preds_od = self._get_off_diag(sim_preds)
        sim_targets_ori_od = self._get_off_diag(sim_targets_ori)
        if l12:
            loss = F.l1_loss(input=sim_preds_od, target=sim_targets_ori_od)
        else:
            loss = F.mse_loss(input=sim_preds_od, target=sim_targets_ori_od)
        print(sim_preds_od.requires_grad, sim_targets_ori_od.requires_grad,
              sim_preds_od.size(), sim_targets_ori_od.size(), "grad check l12")
        return loss

    def _get_phrase_ce_loss_od(self, fpv_feat, tpv_feat, sim_targets_ori, temp_pred, temp_gt, bce=False):
        """
            Qitong on Aug. 7th, 2022.
        """
        if bce:
            sim_preds = self._get_off_diag(fpv_feat @ tpv_feat.T)
            sim_targets = self._get_off_diag(sim_targets_ori)
            loss = F.binary_cross_entropy_with_logits(input=sim_preds, target=F.relu(sim_targets))
        else:
            sim_preds = self._get_off_diag(fpv_feat @ tpv_feat.T) / temp_pred
            sim_targets = F.softmax(self._get_off_diag(sim_targets_ori) / temp_gt, dim=1)
            loss = -torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        return loss

    def _get_off_diag(self, m):
        n = m.size()[0]
        return m.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)

    def forward(self, inputs_fpv, inputs_tpv, label_matrix, label_matrix_fpv, label_matrix_tpv,
                fpv_narrs=None, tpv_narrs=None, label_matrix_l=None):
        assert (
            len(inputs_fpv) == self.num_pathways
        ), "FPVs tensor does not contain {} pathway".format(self.num_pathways)
        assert (
                len(inputs_tpv) == self.num_pathways
        ), "TPVs tensor does not contain {} pathway".format(self.num_pathways)
        pool_out_fpv = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            # print(inputs_fpv[pathway].shape, m(inputs_fpv[pathway]).shape, 'fpv')
            pool_out_fpv.append(m(inputs_fpv[pathway]))
        x_fpv = torch.cat(pool_out_fpv, 1)
        pool_out_tpv = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            # print(inputs_tpv[pathway].shape, m(inputs_tpv[pathway]).shape, 'tpv')
            pool_out_tpv.append(m(inputs_tpv[pathway]))
        x_tpv = torch.cat(pool_out_tpv, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).

        x_fpv = x_fpv.permute((0, 2, 3, 4, 1)).squeeze()
        x_tpv = x_tpv.permute((0, 2, 3, 4, 1)).squeeze()

        fpv_global_feat = F.normalize(x_fpv, dim=1)
        tpv_global_feat = F.normalize(x_tpv, dim=1)
        fpv_global_feat_pre_norm = self.fpv_global_proj(fpv_global_feat)
        fpv_global_feat_norm = F.normalize(fpv_global_feat_pre_norm, dim=1)
        tpv_global_feat_pre_norm = self.tpv_global_proj(tpv_global_feat)
        tpv_global_feat_norm = F.normalize(tpv_global_feat_pre_norm, dim=1)

        label_matrix_01 = torch.zeros(label_matrix.size()).to(label_matrix.device)
        label_matrix_01.fill_diagonal_(1)
        # if label_matrix_l is None:
        #     loss_gcon = self._get_phrase_loss(fpv_global_feat_norm, tpv_global_feat_norm, label_matrix, False)
        # else:
        #     alpha = 0.
        #     label_matrix_ = alpha * label_matrix_01 + (1 - alpha) * F.softmax(label_matrix / 1.0, dim=1)
            # loss_gcon = self._get_phrase_loss(fpv_global_feat_norm, tpv_global_feat_norm, label_matrix_, False)
            # loss_gcon = self._get_ce_loss(fpv_global_feat_norm, tpv_global_feat_norm, label_matrix_,
            #                               temp_pred=self.temp_pred, temp_gt=0.07, bce=False)
            # loss_gcon = self._get_part_ce_loss(fpv_global_feat_norm, tpv_global_feat_norm, label_matrix,
            #                                    temp_pred=self.temp_pred, temp_gt=0.07, thres=self.thres)
            # print('CE!! part', alpha, self.temp_pred)
        # loss_gcon_f = self._get_phrase_loss(fpv_global_feat_norm, fpv_global_feat_norm, label_matrix_fpv, False)
        # loss_gcon_t = self._get_phrase_loss(tpv_global_feat_norm, tpv_global_feat_norm, label_matrix_tpv, False)

        # loss_gcon = self._get_phrase_ce_loss(fpv_global_feat_norm, tpv_global_feat_norm, label_matrix,
        #                                      temp_pred=self.temp_pred, temp_gt=0.07, bce=True, dig=False)
        loss_gcon = self._get_part_ce_loss(fpv_global_feat_norm, tpv_global_feat_norm, label_matrix,
                                           temp_pred=self.temp_pred, temp_gt=0.07, thres=self.thres)
        print('CE!! part', self.temp_pred)
        label_matrix_01 = torch.zeros(label_matrix.size()).to(label_matrix.device)
        label_matrix_01.fill_diagonal_(1)
        loss_gcon_f = self._get_ce_loss(fpv_global_feat_norm, fpv_narrs, label_matrix_01,
                                        temp_pred=self.temp_pred, temp_gt=0.07, bce=False)
        loss_gcon_t = self._get_ce_loss(tpv_global_feat_norm, tpv_narrs, label_matrix_01,
                                        temp_pred=self.temp_pred, temp_gt=0.07, bce=False)
        return loss_gcon, loss_gcon_f, loss_gcon_t

class SlowFastPhraseGlobalConHead_DCL(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        pool_size,
        temp_pred,
        temp_gt,
        thres=0.8,
        sigma=0.5
    ):
        super(SlowFastPhraseGlobalConHead_DCL, self).__init__()

        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        self.temp_pred = temp_pred
        self.temp_gt = temp_gt
        self.thres = thres
        self.sigma = sigma
        self.fpv_global_proj = nn.Linear(sum(dim_in), dim_out, bias=True)
        self.tpv_global_proj = nn.Linear(sum(dim_in), dim_out, bias=True)
        # self.dcl_loss = DCL(temperature=self.temp_pred)
        # self.dcl_loss = DCLW(temperature=self.temp_pred)
        self.dcl_loss_mm = DCL(temperature=self.temp_pred)
        self.dcl_loss = DCL_OS(sigma=self.sigma, temperature=self.temp_pred)

    def _get_ce_loss(self, fpv_feat, tpv_feat, sim_targets_ori, temp_pred, temp_gt):
        """
            Qitong on Aug. 7th, 2022.
        """
        res_loss_1 = self.dcl_loss_mm(fpv_feat, tpv_feat)
        res_loss_2 = self.dcl_loss_mm(tpv_feat, fpv_feat)
        # if res_loss_1 < 0 or res_loss_2 < 0:
        #     return None
        return res_loss_1 + res_loss_2

    def _get_part_ce_loss(self, fpv_feat, tpv_feat, fpv_narrs, tpv_narrs,
                          sim_targets_ori, temp_pred, temp_gt, thres):
        """
            Qitong on Aug. 7th, 2022.
        """

        batch_pesudo_pair_sim = torch.diagonal(sim_targets_ori)
        idx = torch.nonzero(
            torch.where(
                batch_pesudo_pair_sim > thres,
                torch.ones_like(batch_pesudo_pair_sim),
                torch.zeros_like(batch_pesudo_pair_sim)
            )
        ).squeeze()
        fpv_feat_slides = torch.index_select(fpv_feat, 0, idx)
        tpv_feat_slides = torch.index_select(tpv_feat, 0, idx)
        fpv_narr_slides = torch.index_select(fpv_narrs, 0, idx)
        tpv_narr_slides = torch.index_select(tpv_narrs, 0, idx)
        print(fpv_feat_slides.size(), tpv_feat_slides.size(), fpv_narr_slides.size(), tpv_narr_slides.size(), 23333)
        if fpv_feat_slides.size()[0] < 2:
            return None
        else:
            # res_loss_1 = self.dcl_loss(fpv_feat_slides, tpv_feat)
            # res_loss_2 = self.dcl_loss(tpv_feat, fpv_feat_slides)
            res_loss_1 = self.dcl_loss(fpv_feat_slides, tpv_feat_slides, fpv_narr_slides, tpv_narr_slides)
            res_loss_2 = self.dcl_loss(tpv_feat_slides, fpv_feat_slides, tpv_narr_slides, fpv_narr_slides)
            # res_loss_1 = self.dcl_loss(fpv_feat_slides, tpv_feat_slides)
            # res_loss_2 = self.dcl_loss(tpv_feat_slides, fpv_feat_slides)
            # if res_loss_1 < 0 or res_loss_2 < 0:
            #     return None
            return res_loss_1 + res_loss_2

    def _get_part_triplet_loss(self, fpv_feat, tpv_feat, fpv_narrs, tpv_narrs,
                               sim_targets_ori, temp_pred, temp_gt, thres):
        """
            Qitong on Feb. 18th, 2023.
        """

        neg_fpv_feat = fpv_feat[torch.randperm(fpv_feat.size()[0])]
        return F.triplet_margin_loss(anchor=fpv_feat, positive=tpv_feat, negative=neg_fpv_feat, margin=0.1, p=2)

    def _get_phrase_ce_loss(self, fpv_feat, tpv_feat,
                            sim_targets_ori, temp_pred, temp_gt, bce=False, dig=False):
        """
            Qitong on Aug. 7th, 2022.
        """
        if bce:
            # print('gt', sim_targets_ori)
            sim_preds = fpv_feat @ tpv_feat.T
            # loss = F.binary_cross_entropy_with_logits(input=sim_preds, target=F.relu(sim_targets_ori))
            if dig:
                loss = F.binary_cross_entropy(
                    input=self.cos2preb(torch.diagonal(sim_preds)),
                    target=self.cos2preb(torch.diagonal(sim_targets_ori))
                )
            else:
                loss = F.binary_cross_entropy(
                    input=self.cos2preb(sim_preds),
                    target=self.cos2preb(sim_targets_ori)
                )
        else:
            sim_preds = fpv_feat @ tpv_feat.T / temp_pred
            sim_targets = F.softmax(sim_targets_ori / temp_gt, dim=1)
            loss = -torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        return loss

    def forward(self, inputs_fpv, inputs_tpv, label_matrix, label_matrix_fpv, label_matrix_tpv,
                fpv_narrs=None, tpv_narrs=None, label_matrix_l=None):
        assert (
            len(inputs_fpv) == self.num_pathways
        ), "FPVs tensor does not contain {} pathway".format(self.num_pathways)
        assert (
                len(inputs_tpv) == self.num_pathways
        ), "TPVs tensor does not contain {} pathway".format(self.num_pathways)
        pool_out_fpv = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            # print(inputs_fpv[pathway].shape, m(inputs_fpv[pathway]).shape, 'fpv')
            pool_out_fpv.append(m(inputs_fpv[pathway]))
        x_fpv = torch.cat(pool_out_fpv, 1)
        pool_out_tpv = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            # print(inputs_tpv[pathway].shape, m(inputs_tpv[pathway]).shape, 'tpv')
            pool_out_tpv.append(m(inputs_tpv[pathway]))
        x_tpv = torch.cat(pool_out_tpv, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).

        x_fpv = x_fpv.permute((0, 2, 3, 4, 1)).squeeze()
        x_tpv = x_tpv.permute((0, 2, 3, 4, 1)).squeeze()

        fpv_global_feat = F.normalize(x_fpv, dim=1)
        tpv_global_feat = F.normalize(x_tpv, dim=1)
        fpv_global_feat_pre_norm = self.fpv_global_proj(fpv_global_feat)
        fpv_global_feat_norm = F.normalize(fpv_global_feat_pre_norm, dim=1)
        tpv_global_feat_pre_norm = self.tpv_global_proj(tpv_global_feat)
        tpv_global_feat_norm = F.normalize(tpv_global_feat_pre_norm, dim=1)

        label_matrix_01 = torch.zeros(label_matrix.size()).to(label_matrix.device)
        label_matrix_01.fill_diagonal_(1)
        if label_matrix_l is None:
            print("not a??!!")
            # loss_gcon = self._get_part_ce_loss(fpv_global_feat_norm, tpv_global_feat_norm,
            #                                    fpv_narrs, tpv_narrs, label_matrix,
            #                                    temp_pred=self.temp_pred, temp_gt=0.07, thres=self.thres)
            loss_gcon = self._get_part_triplet_loss(fpv_global_feat_norm,
                                                    tpv_global_feat_norm,
                                                    fpv_narrs, tpv_narrs, label_matrix,
                                                    temp_pred=self.temp_pred, temp_gt=0.07, thres=self.thres)
        else:
            alpha = 0.0
            print("why a??!!", alpha, fpv_global_feat_norm.size(), tpv_global_feat_norm.size())
            label_matrix_ = alpha * label_matrix_l + (1 - alpha) * label_matrix
            # loss_gcon = self._get_phrase_loss(fpv_global_feat_norm, tpv_global_feat_norm, label_matrix_, False)
            # loss_gcon = self._get_ce_loss(fpv_global_feat_norm, tpv_global_feat_norm, label_matrix_,
            #                               temp_pred=self.temp_pred, temp_gt=0.07, bce=False)
            # loss_gcon = self._get_part_ce_loss(fpv_global_feat_norm, tpv_global_feat_norm,
            #                                    fpv_narrs, tpv_narrs, label_matrix_,
            #                                    temp_pred=self.temp_pred, temp_gt=0.07, thres=self.thres)
            # loss_gcon = self._get_part_triplet_loss(fpv_global_feat_norm,
            #                                         tpv_global_feat_norm,
            #                                         fpv_narrs, tpv_narrs, label_matrix,
            #                                         temp_pred=self.temp_pred, temp_gt=0.07, thres=self.thres)
        print('DCL!! part', self.temp_pred, self.sigma)
        label_matrix_01 = torch.zeros(label_matrix.size()).to(label_matrix.device)
        label_matrix_01.fill_diagonal_(1)
        loss_gcon_f = self._get_ce_loss(fpv_global_feat_norm, fpv_narrs, label_matrix_01,
                                        temp_pred=self.temp_pred, temp_gt=0.07)
        loss_gcon_t = self._get_ce_loss(tpv_global_feat_norm, tpv_narrs, label_matrix_01,
                                        temp_pred=self.temp_pred, temp_gt=0.07)
        loss_gcon = None
        # loss_gcon_f = None
        # loss_gcon_t = None
        return loss_gcon, loss_gcon_f, loss_gcon_t

class SlowFastPhraseGlobalConHead_DCL_DIV(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        pool_size,
        temp_pred,
        temp_gt,
        thres=0.8,
        sigma=0.5
    ):
        super(SlowFastPhraseGlobalConHead_DCL_DIV, self).__init__()

        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        self.temp_pred = temp_pred
        self.temp_gt = temp_gt
        self.thres = thres
        self.sigma = sigma
        self.fpv_global_proj = nn.Linear(sum(dim_in), dim_out, bias=True)
        self.tpv_global_proj = nn.Linear(sum(dim_in), dim_out, bias=True)
        self.fpv_global_proj_mm = nn.Linear(sum(dim_in), 384, bias=True)
        self.tpv_global_proj_mm = nn.Linear(sum(dim_in), 384, bias=True)
        # self.dcl_loss = DCL(temperature=self.temp_pred)
        # self.dcl_loss = DCLW(temperature=self.temp_pred)
        self.dcl_loss_mm = DCL(temperature=self.temp_pred)
        self.dcl_loss = DCL_OS(sigma=self.sigma, temperature=self.temp_pred)

    def _get_ce_loss(self, fpv_feat, tpv_feat, sim_targets_ori, temp_pred, temp_gt):
        """
            Qitong on Aug. 7th, 2022.
        """
        res_loss_1 = self.dcl_loss_mm(fpv_feat, tpv_feat)
        res_loss_2 = self.dcl_loss_mm(tpv_feat, fpv_feat)
        # if res_loss_1 < 0 or res_loss_2 < 0:
        #     return None
        return res_loss_1 + res_loss_2

    def _get_part_ce_loss(self, fpv_feat, tpv_feat, fpv_narrs, tpv_narrs,
                          sim_targets_ori, temp_pred, temp_gt, thres):
        """
            Qitong on Aug. 7th, 2022.
        """

        batch_pesudo_pair_sim = torch.diagonal(sim_targets_ori)
        idx = torch.nonzero(
            torch.where(
                batch_pesudo_pair_sim > thres,
                torch.ones_like(batch_pesudo_pair_sim),
                torch.zeros_like(batch_pesudo_pair_sim)
            )
        ).squeeze()
        fpv_feat_slides = torch.index_select(fpv_feat, 0, idx)
        tpv_feat_slides = torch.index_select(tpv_feat, 0, idx)
        fpv_narr_slides = torch.index_select(fpv_narrs, 0, idx)
        tpv_narr_slides = torch.index_select(tpv_narrs, 0, idx)
        print(fpv_feat_slides.size(), tpv_feat_slides.size(), fpv_narr_slides.size(), tpv_narr_slides.size(), 23333)
        if fpv_feat_slides.size()[0] < 2:
            return None
        else:
            # res_loss_1 = self.dcl_loss(fpv_feat_slides, tpv_feat)
            # res_loss_2 = self.dcl_loss(tpv_feat, fpv_feat_slides)
            res_loss_1 = self.dcl_loss(fpv_feat_slides, tpv_feat_slides, fpv_narr_slides, tpv_narr_slides)
            res_loss_2 = self.dcl_loss(tpv_feat_slides, fpv_feat_slides, tpv_narr_slides, fpv_narr_slides)
            # res_loss_1 = self.dcl_loss(fpv_feat_slides, tpv_feat_slides)
            # res_loss_2 = self.dcl_loss(tpv_feat_slides, fpv_feat_slides)
            # if res_loss_1 < 0 or res_loss_2 < 0:
            #     return None
            return res_loss_1 + res_loss_2

    def _get_phrase_ce_loss(self, fpv_feat, tpv_feat,
                            sim_targets_ori, temp_pred, temp_gt, bce=False, dig=False):
        """
            Qitong on Aug. 7th, 2022.
        """
        if bce:
            # print('gt', sim_targets_ori)
            sim_preds = fpv_feat @ tpv_feat.T
            # loss = F.binary_cross_entropy_with_logits(input=sim_preds, target=F.relu(sim_targets_ori))
            if dig:
                loss = F.binary_cross_entropy(
                    input=self.cos2preb(torch.diagonal(sim_preds)),
                    target=self.cos2preb(torch.diagonal(sim_targets_ori))
                )
            else:
                loss = F.binary_cross_entropy(
                    input=self.cos2preb(sim_preds),
                    target=self.cos2preb(sim_targets_ori)
                )
        else:
            sim_preds = fpv_feat @ tpv_feat.T / temp_pred
            sim_targets = F.softmax(sim_targets_ori / temp_gt, dim=1)
            loss = -torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        return loss

    def forward(self, inputs_fpv, inputs_tpv, label_matrix, label_matrix_fpv, label_matrix_tpv,
                fpv_narrs=None, tpv_narrs=None, label_matrix_l=None):
        assert (
            len(inputs_fpv) == self.num_pathways
        ), "FPVs tensor does not contain {} pathway".format(self.num_pathways)
        assert (
                len(inputs_tpv) == self.num_pathways
        ), "TPVs tensor does not contain {} pathway".format(self.num_pathways)
        pool_out_fpv = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            # print(inputs_fpv[pathway].shape, m(inputs_fpv[pathway]).shape, 'fpv')
            pool_out_fpv.append(m(inputs_fpv[pathway]))
        x_fpv = torch.cat(pool_out_fpv, 1)
        pool_out_tpv = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            # print(inputs_tpv[pathway].shape, m(inputs_tpv[pathway]).shape, 'tpv')
            pool_out_tpv.append(m(inputs_tpv[pathway]))
        x_tpv = torch.cat(pool_out_tpv, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).

        x_fpv = x_fpv.permute((0, 2, 3, 4, 1)).squeeze()
        x_tpv = x_tpv.permute((0, 2, 3, 4, 1)).squeeze()

        fpv_global_feat = F.normalize(x_fpv, dim=1)
        tpv_global_feat = F.normalize(x_tpv, dim=1)
        fpv_global_feat_pre_norm = self.fpv_global_proj(fpv_global_feat)
        fpv_global_feat_norm = F.normalize(fpv_global_feat_pre_norm, dim=1)
        tpv_global_feat_pre_norm = self.tpv_global_proj(tpv_global_feat)
        tpv_global_feat_norm = F.normalize(tpv_global_feat_pre_norm, dim=1)
        fpv_global_feat_pre_norm_mm = self.fpv_global_proj_mm(fpv_global_feat)
        fpv_global_feat_norm_mm = F.normalize(fpv_global_feat_pre_norm_mm, dim=1)
        tpv_global_feat_pre_norm_mm = self.tpv_global_proj_mm(tpv_global_feat)
        tpv_global_feat_norm_mm = F.normalize(tpv_global_feat_pre_norm_mm, dim=1)

        label_matrix_01 = torch.zeros(label_matrix.size()).to(label_matrix.device)
        label_matrix_01.fill_diagonal_(1)
        if label_matrix_l is None:
            print("not a??!!")
            loss_gcon = self._get_part_ce_loss(fpv_global_feat_norm, tpv_global_feat_norm,
                                               fpv_narrs, tpv_narrs, label_matrix,
                                               temp_pred=self.temp_pred, temp_gt=0.07, thres=self.thres)
        else:
            alpha = 0.0
            print("why a??!!", alpha)
            label_matrix_ = alpha * label_matrix_l + (1 - alpha) * label_matrix
            # loss_gcon = self._get_phrase_loss(fpv_global_feat_norm, tpv_global_feat_norm, label_matrix_, False)
            # loss_gcon = self._get_ce_loss(fpv_global_feat_norm, tpv_global_feat_norm, label_matrix_,
            #                               temp_pred=self.temp_pred, temp_gt=0.07, bce=False)
            loss_gcon = self._get_part_ce_loss(fpv_global_feat_norm, tpv_global_feat_norm,
                                               fpv_narrs, tpv_narrs, label_matrix_,
                                               temp_pred=self.temp_pred, temp_gt=0.07, thres=self.thres)
        print('DCL!! part', self.temp_pred, self.sigma, 'div',
              fpv_global_feat_norm.size(), fpv_global_feat_norm_mm.size(),
              tpv_global_feat_norm.size(), tpv_global_feat_norm_mm.size())
        label_matrix_01 = torch.zeros(label_matrix.size()).to(label_matrix.device)
        label_matrix_01.fill_diagonal_(1)
        loss_gcon_f = self._get_ce_loss(fpv_global_feat_norm_mm, fpv_narrs, label_matrix_01,
                                        temp_pred=0.05, temp_gt=0.07)
        loss_gcon_t = self._get_ce_loss(tpv_global_feat_norm_mm, tpv_narrs, label_matrix_01,
                                        temp_pred=0.05, temp_gt=0.07)
        return loss_gcon, loss_gcon_f, loss_gcon_t

class SlowFastPhraseGlobalConHead_P_DIV(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        pool_size,
        temp_pred,
        temp_gt
    ):
        super(SlowFastPhraseGlobalConHead_P_DIV, self).__init__()

        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        self.temp_pred = temp_pred
        self.temp_gt = temp_gt
        self.fpv_global_proj_s = nn.Linear(2048, dim_out, bias=True)
        self.tpv_global_proj_s = nn.Linear(2048, dim_out, bias=True)
        self.fpv_global_proj_f = nn.Linear(256, 128, bias=True)
        self.tpv_global_proj_f = nn.Linear(256, 128, bias=True)

    def my_sigmoid(self, vec):
        # print(vec)
        return 1 / (1 + torch.exp(-3 * vec))

    def _get_phrase_loss(self, fpv_feat, tpv_feat, sim_targets_ori, l12):
        """
            Qitong on Aug. 7th, 2022.
        """
        sim_preds = fpv_feat @ tpv_feat.T
        # sim_preds = self.my_sigmoid(fpv_feat @ tpv_feat.T)
        if l12:
            loss = F.l1_loss(input=sim_preds, target=sim_targets_ori)
        else:
            loss = F.mse_loss(input=sim_preds, target=sim_targets_ori)
        return loss

    def _get_phrase_ce_loss(self, fpv_feat, tpv_feat, sim_targets_ori, temp_pred, temp_gt, bce=False):
        """
            Qitong on Aug. 7th, 2022.
        """
        if bce:
            # print('gt', sim_targets_ori)
            sim_preds = fpv_feat @ tpv_feat.T
            # loss = F.binary_cross_entropy_with_logits(input=sim_preds, target=F.relu(sim_targets_ori))
            loss = F.binary_cross_entropy(input=self.my_sigmoid(sim_preds), target=F.relu(sim_targets_ori))
        else:
            sim_preds = fpv_feat @ tpv_feat.T / temp_pred
            sim_targets = F.softmax(sim_targets_ori / temp_gt, dim=1)
            loss = -torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        return loss

    def forward(self, inputs_fpv, inputs_tpv, label_matrix, label_matrix_fpv, label_matrix_tpv):
        assert (
            len(inputs_fpv) == self.num_pathways
        ), "FPVs tensor does not contain {} pathway".format(self.num_pathways)
        assert (
                len(inputs_tpv) == self.num_pathways
        ), "TPVs tensor does not contain {} pathway".format(self.num_pathways)
        pool_out_fpv = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            # print(inputs_fpv[pathway].shape, m(inputs_fpv[pathway]).shape, 'fpv')
            pool_out_fpv.append(m(inputs_fpv[pathway]))
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x_fpv_s = pool_out_fpv[0].permute((0, 2, 3, 4, 1)).squeeze()
        x_fpv_f = pool_out_fpv[1].permute((0, 2, 3, 4, 1)).squeeze()
        pool_out_tpv = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            # print(inputs_tpv[pathway].shape, m(inputs_tpv[pathway]).shape, 'tpv')
            pool_out_tpv.append(m(inputs_tpv[pathway]))
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x_tpv_s = pool_out_tpv[0].permute((0, 2, 3, 4, 1)).squeeze()
        x_tpv_f = pool_out_tpv[1].permute((0, 2, 3, 4, 1)).squeeze()
        print(x_fpv_s.size(), x_fpv_f.size(), x_tpv_s.size(), x_tpv_f.size(), 'div')


        fpv_global_feat_s = F.normalize(x_fpv_s, dim=1)
        tpv_global_feat_s = F.normalize(x_tpv_s, dim=1)
        fpv_global_feat_pre_norm_s = self.fpv_global_proj_s(fpv_global_feat_s)
        fpv_global_feat_norm_s = F.normalize(fpv_global_feat_pre_norm_s, dim=1)
        tpv_global_feat_pre_norm_s = self.tpv_global_proj_s(tpv_global_feat_s)
        tpv_global_feat_norm_s = F.normalize(tpv_global_feat_pre_norm_s, dim=1)
        loss_gcon_s = self._get_phrase_loss(fpv_global_feat_norm_s, tpv_global_feat_norm_s, label_matrix, False)

        fpv_global_feat_f = F.normalize(x_fpv_f, dim=1)
        tpv_global_feat_f = F.normalize(x_tpv_f, dim=1)
        fpv_global_feat_pre_norm_f = self.fpv_global_proj_f(fpv_global_feat_f)
        fpv_global_feat_norm_f = F.normalize(fpv_global_feat_pre_norm_f, dim=1)
        tpv_global_feat_pre_norm_f = self.tpv_global_proj_f(tpv_global_feat_f)
        tpv_global_feat_norm_f = F.normalize(tpv_global_feat_pre_norm_f, dim=1)
        loss_gcon_f = self._get_phrase_loss(fpv_global_feat_norm_f, tpv_global_feat_norm_f, label_matrix, False)

        loss_gcon = (loss_gcon_s + loss_gcon_f) / 2
        return loss_gcon, None, None

class SlowFastPhraseGlobalConHead_P_CST(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        pool_size,
        temp_pred,
        temp_gt
    ):
        super(SlowFastPhraseGlobalConHead_P_CST, self).__init__()

        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        self.temp_pred = temp_pred
        self.temp_gt = temp_gt
        self.fpv_global_proj = nn.Linear(sum(dim_in), dim_out, bias=True)
        self.tpv_global_proj = nn.Linear(sum(dim_in), dim_out, bias=True)

    def _get_phrase_loss(self, fpv_feat, tpv_feat, sim_targets_ori, l12):
        """
            Qitong on Aug. 7th, 2022.
        """
        sim_preds = fpv_feat @ tpv_feat.T
        if l12:
            loss = F.l1_loss(input=sim_preds, target=sim_targets_ori)
        else:
            loss = F.mse_loss(input=sim_preds, target=sim_targets_ori)
        return loss

    def my_sigmoid(self, vec):
        # print(vec)
        return 1 / (1 + torch.exp(-3 * vec))

    def _get_ce_loss(self, fpv_feat, tpv_feat, sim_targets_ori, temp_pred, temp_gt, bce=False):
        """
            Qitong on Aug. 7th, 2022.
        """
        if bce:
            # print('gt', sim_targets_ori)
            sim_preds = fpv_feat @ tpv_feat.T
            # loss = F.binary_cross_entropy_with_logits(input=sim_preds, target=F.relu(sim_targets_ori))
            loss = F.binary_cross_entropy(input=self.my_sigmoid(sim_preds), target=F.relu(sim_targets_ori))
        else:
            sim_preds = fpv_feat @ tpv_feat.T / temp_pred
            sim_targets = sim_targets_ori
            loss = -torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        return loss

    def _get_phrase_ce_loss(self, fpv_feat, tpv_feat, sim_targets_ori, temp_pred, temp_gt, bce=False):
        """
            Qitong on Aug. 7th, 2022.
        """
        if bce:
            # print('gt', sim_targets_ori)
            sim_preds = fpv_feat @ tpv_feat.T
            # loss = F.binary_cross_entropy_with_logits(input=sim_preds, target=F.relu(sim_targets_ori))
            loss = F.binary_cross_entropy(input=self.my_sigmoid(sim_preds), target=F.relu(sim_targets_ori))
        else:
            sim_preds = fpv_feat @ tpv_feat.T / temp_pred
            sim_targets = F.softmax(sim_targets_ori / temp_gt, dim=1)
            loss = -torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        return loss

    def _get_phrase_loss_od(self, fpv_feat, tpv_feat, sim_targets_ori, l12):
        """
            Qitong on Aug. 7th, 2022.
        """
        sim_preds = fpv_feat @ tpv_feat.T
        sim_preds_od = self._get_off_diag(sim_preds)
        sim_targets_ori_od = self._get_off_diag(sim_targets_ori)
        if l12:
            loss = F.l1_loss(input=sim_preds_od, target=sim_targets_ori_od)
        else:
            loss = F.mse_loss(input=sim_preds_od, target=sim_targets_ori_od)
        print(sim_preds_od.requires_grad, sim_targets_ori_od.requires_grad,
              sim_preds_od.size(), sim_targets_ori_od.size(), "grad check l12")
        return loss

    def _get_phrase_ce_loss_od(self, fpv_feat, tpv_feat, sim_targets_ori, temp_pred, temp_gt, bce=False):
        """
            Qitong on Aug. 7th, 2022.
        """
        if bce:
            sim_preds = self._get_off_diag(fpv_feat @ tpv_feat.T)
            sim_targets = self._get_off_diag(sim_targets_ori)
            loss = F.binary_cross_entropy_with_logits(input=sim_preds, target=F.relu(sim_targets))
        else:
            sim_preds = self._get_off_diag(fpv_feat @ tpv_feat.T) / temp_pred
            sim_targets = F.softmax(self._get_off_diag(sim_targets_ori) / temp_gt, dim=1)
            loss = -torch.sum(F.log_softmax(sim_preds, dim=1) * sim_targets, dim=1).mean()
        return loss

    def _get_off_diag(self, m):
        n = m.size()[0]
        return m.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)

    def forward(self, inputs_fpv_1, inputs_tpv_1, inputs_fpv_2, inputs_tpv_2,
                label_matrix, label_matrix_fpv, label_matrix_tpv,
                fpv_narrs=None, tpv_narrs=None):
        assert (
            len(inputs_fpv_1) == self.num_pathways
        ), "FPVs tensor does not contain {} pathway".format(self.num_pathways)
        assert (
                len(inputs_tpv_1) == self.num_pathways
        ), "TPVs tensor does not contain {} pathway".format(self.num_pathways)
        assert (
                len(inputs_fpv_2) == self.num_pathways
        ), "FPVs tensor does not contain {} pathway".format(self.num_pathways)
        assert (
                len(inputs_tpv_2) == self.num_pathways
        ), "TPVs tensor does not contain {} pathway".format(self.num_pathways)
        pool_out_fpv_1 = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out_fpv_1.append(m(inputs_fpv_1[pathway]))
        x_fpv_1 = torch.cat(pool_out_fpv_1, 1)
        pool_out_tpv_1 = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out_tpv_1.append(m(inputs_tpv_1[pathway]))
        x_tpv_1 = torch.cat(pool_out_tpv_1, 1)

        pool_out_fpv_2 = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out_fpv_2.append(m(inputs_fpv_2[pathway]))
        x_fpv_2 = torch.cat(pool_out_fpv_2, 1)
        pool_out_tpv_2 = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out_tpv_2.append(m(inputs_tpv_2[pathway]))
        x_tpv_2 = torch.cat(pool_out_tpv_2, 1)

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x_fpv_1 = x_fpv_1.permute((0, 2, 3, 4, 1)).squeeze()
        x_tpv_1 = x_tpv_1.permute((0, 2, 3, 4, 1)).squeeze()
        x_fpv_2 = x_fpv_2.permute((0, 2, 3, 4, 1)).squeeze()
        x_tpv_2 = x_tpv_2.permute((0, 2, 3, 4, 1)).squeeze()
        print(x_fpv_1.size(), x_fpv_2.size(), x_tpv_1.size(), x_tpv_2.size())

        fpv_global_feat_1 = F.normalize(x_fpv_1, dim=1)
        tpv_global_feat_1 = F.normalize(x_tpv_1, dim=1)
        fpv_global_feat_pre_norm_1 = self.fpv_global_proj(fpv_global_feat_1)
        fpv_global_feat_norm_1 = F.normalize(fpv_global_feat_pre_norm_1, dim=1)
        tpv_global_feat_pre_norm_1 = self.tpv_global_proj(tpv_global_feat_1)
        tpv_global_feat_norm_1 = F.normalize(tpv_global_feat_pre_norm_1, dim=1)

        fpv_global_feat_2 = F.normalize(x_fpv_2, dim=1)
        tpv_global_feat_2 = F.normalize(x_tpv_2, dim=1)
        fpv_global_feat_pre_norm_2 = self.fpv_global_proj(fpv_global_feat_2)
        fpv_global_feat_norm_2 = F.normalize(fpv_global_feat_pre_norm_2, dim=1)
        tpv_global_feat_pre_norm_2 = self.tpv_global_proj(tpv_global_feat_2)
        tpv_global_feat_norm_2 = F.normalize(tpv_global_feat_pre_norm_2, dim=1)

        loss_gcon_11 = self._get_phrase_loss(fpv_global_feat_norm_1, tpv_global_feat_norm_1, label_matrix, False)
        loss_gcon_12 = self._get_phrase_loss(fpv_global_feat_norm_1, tpv_global_feat_norm_2, label_matrix, False)
        loss_gcon_21 = self._get_phrase_loss(fpv_global_feat_norm_2, tpv_global_feat_norm_1, label_matrix, False)
        loss_gcon_22 = self._get_phrase_loss(fpv_global_feat_norm_2, tpv_global_feat_norm_2, label_matrix, False)

        # loss_gcon_11 = self._get_phrase_ce_loss(fpv_global_feat_norm_1, tpv_global_feat_norm_1, label_matrix,
        #                                        temp_pred=self.temp_pred, temp_gt=0.5, bce=True)
        # loss_gcon_12 = self._get_phrase_ce_loss(fpv_global_feat_norm_1, tpv_global_feat_norm_2, label_matrix,
        #                                        temp_pred=self.temp_pred, temp_gt=0.5, bce=True)
        # loss_gcon_21 = self._get_phrase_ce_loss(fpv_global_feat_norm_2, tpv_global_feat_norm_1, label_matrix,
        #                                         temp_pred=self.temp_pred, temp_gt=0.5, bce=True)
        # loss_gcon_22 = self._get_phrase_ce_loss(fpv_global_feat_norm_2, tpv_global_feat_norm_2, label_matrix,
        #                                         temp_pred=self.temp_pred, temp_gt=0.5, bce=True)

        label_matrix_01 = torch.zeros(label_matrix.size()).to(label_matrix.device)
        label_matrix_01.fill_diagonal_(1)
        loss_gcon_f1 = self._get_ce_loss(fpv_global_feat_norm_1, fpv_narrs, label_matrix_01,
                                         temp_pred=self.temp_pred, temp_gt=0.07, bce=False)
        loss_gcon_t1 = self._get_ce_loss(tpv_global_feat_norm_1, tpv_narrs, label_matrix_01,
                                         temp_pred=self.temp_pred, temp_gt=0.07, bce=False)
        loss_gcon_f2 = self._get_ce_loss(fpv_global_feat_norm_2, fpv_narrs, label_matrix_01,
                                         temp_pred=self.temp_pred, temp_gt=0.07, bce=False)
        loss_gcon_t2 = self._get_ce_loss(tpv_global_feat_norm_2, tpv_narrs, label_matrix_01,
                                         temp_pred=self.temp_pred, temp_gt=0.07, bce=False)

        return loss_gcon_11, loss_gcon_12, loss_gcon_21, loss_gcon_22, \
               loss_gcon_f1, loss_gcon_t1, loss_gcon_f2, loss_gcon_t2

def gen_targets(gts, B, T, size, mode):
    assert mode in ["soft", "hard"]
    targets = torch.zeros((B, T, size, size)).cuda()
    # print(gts)
    for idx, gt in enumerate(gts):
        boxes = gt[:, :4] * size
        boxes = boxes.floor().long()
        boxes = torch.clamp(boxes, min=0, max=size - 1)

        for i in range(boxes.shape[0]):
            iT = int(gt[i, 5])
            # print(iT)
            x1, y1, x2, y2 = [x.item() for x in boxes[i, :4]]
            # x1, y1, x2, y2 = boxes[i, 0].item()
            if mode == 'soft':
                targets[idx, iT, y1:y2 + 1, x1:x2 + 1] = torch.clamp(
                    targets[idx, iT, y1:y2 + 1, x1:x2 + 1],
                    min=gt[i, 4].item(),
                )
            else:
                targets[idx, iT, y1:y2 + 1, x1:x2 + 1] = 1.0

    # print(targets[0, 0], 'check inside~')

    return targets

def cal_cam_loss(x, cam_gt, heatmap_size, pos_weight=1.0, mode="soft", pathway=-1):
    x_fast = x[0] if len(x) == 1 else x[1]
    B, _, T, H, W = x_fast.shape
    # B x T x heatmap_size x heatmap_size
    hand_targets = cam_gt
    # print(hand_targets.size(), obj_targets.size(), 'hoi')
    if len(x) == 1:
        x = x[0]
    else:
        slow_T = x[0].shape[2]
        index = torch.linspace(0, T - 1, slow_T).long().cuda()
        slow_hand_targets = torch.index_select(hand_targets, 1, index)

        if pathway == -1:
            # print(index, index.shape)
            x = torch.cat(x, dim=2)
            # print(x.shape)
            hand_targets = torch.cat((slow_hand_targets, hand_targets), dim=1)
            # print(hand_targets.shape)
        else:
            x = x[pathway]
            if pathway == 0:
                hand_targets = slow_hand_targets
        B, _, T, H, W = x.shape
    # print(x.size(), 'size check')

    hand_targets = hand_targets.view(B * T, H * W)

    xcam = x[:, 0, :, :, :].reshape(B * T, H * W)
    # xobj = x[:, 1, :, :, :].reshape(B * T, H * W)
    pos_weight_tensor = torch.ones([H * W]).cuda() * pos_weight
    loss_fun = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight_tensor)
    loss = {
        "cam_distill_loss": loss_fun(xcam, hand_targets),
        # "obj_loss": loss_fun(xobj, obj_targets),
    }

    return loss

class CAMLocalHead(nn.Module):
    def __init__(
            self,
            dim_in,
            conv_dims=(512,),
            loss_weight=1.0,
            with_targets=True,
            loss_mode="soft",
            pos_weight=1.0,
    ):
        super(CAMLocalHead, self).__init__()
        self.loss_weight = loss_weight
        self.with_targets = with_targets
        self.loss_mode = loss_mode
        self.pos_weight = pos_weight

        self.num_pathways = len(dim_in)
        self.build_blocks(dim_in, conv_dims)

    def build_blocks(self, dim_in, conv_dims):
        self.blocks = []
        for pathway in range(self.num_pathways):
            self.blocks.append([])
            in_channels = dim_in[pathway]
            for idx, layer_channels in enumerate(conv_dims):
                module = nn.Conv3d(in_channels, layer_channels, (1, 3, 3), stride=1, padding=(0, 1, 1))
                self.add_module("pathway{}_conv_fcn{}".format(pathway, idx), module)
                self.blocks[pathway].append(module)
                in_channels = layer_channels

        self.score_projection = nn.Conv3d(
            in_channels, 1, (1, 1, 1), stride=(1, 1, 1)
        )

    def combine_slow_fast(self, slow_feat, fast_feat):
        fast_feat_cut = fast_feat.index_select(2, torch.tensor([0, 4, 8, 12, 16, 20, 24, 28]))
        return torch.cat([slow_feat, fast_feat_cut], dim=1)

    def cam_query_vn_div(self, x_ori, pred_ori_v, pred_proj_v, n_token):
        """
            Qitong on Jul. 25th, 2022.
        """
        bs, c, t, h, w = x_ori[0].size()
        pred_values_v, pred_idxs_v = pred_ori_v.sort(dim=1, descending=True)
        cls_features_v = torch.stack([torch.matmul(pred_proj_v.weight, ele_f.view(c, -1)) for ele_f in x_ori[0]])
        idxs_list_v = []
        values_list_v = []
        for cf_v, st_x, p_i_v in zip(cls_features_v, x_ori[0], pred_idxs_v):
            st_x_ = st_x.view(c, -1)
            values_v_map_final_norm = (cf_v[p_i_v[0]] - torch.min(cf_v[p_i_v[0]])) / \
                                      (torch.max(cf_v[p_i_v[0]]) - torch.min(cf_v[p_i_v[0]]))
            values_v, idxs_v = values_v_map_final_norm.topk(n_token)

            # res_tensor_list.append(st_x_[:, idxs].permute((1, 0)))
            idx_place_fpv_v = []
            for i in idxs_v.detach().cpu().numpy():
                idx_place_fpv_v.append([int(i / 49), int(i % 49 / 7), (i % 49) % 7])
            idxs_list_v.append(idx_place_fpv_v)

            # idx_place_fpv_n = []
            # for i in idxs_n.detach().cpu().numpy():
            #     idx_place_fpv_n.append([int(i / 49), int(i % 49 / 7), (i % 49) % 7])
            # idxs_list_n.append(idx_place_fpv_n)

            value_all_fpv_v = []
            for i in values_v.detach().cpu().numpy():
                value_all_fpv_v.append(i)
            values_list_v.append(value_all_fpv_v)

            # value_all_fpv_n = []
            # for i in values_n.detach().cpu().numpy():
            #     value_all_fpv_n.append(i)
            # values_list_n.append(value_all_fpv_n)

        cam_maps_v = np.zeros((bs, t, h, w))
        # cam_maps_n = np.zeros((bs, t, h, w))

        bs_idx = 0
        for idx_v, values_v in zip(idxs_list_v, values_list_v):
            for j in range(len(idx_v)):
                # print(len(idx_v), len(idx_n), idx_v[j])
                t, h, w = idx_v[j]
                v = values_v[j]
                cam_maps_v[bs_idx][t][h][w] = v
            # for j in range(len(idx_n)):
            #     t, h, w = idx_n[j]
            #     v = values_n[j]
            #     cam_maps_n[bs_idx][t][h][w] = v
            bs_idx += 1
        return torch.tensor(cam_maps_v).cuda().detach()

    def forward(self, inputs, head, x_fpv_pred, meta=None):
        x = inputs
        if len(inputs) == 1:
            cam_gt = self.cam_query_vn_div(x, F.sigmoid(x_fpv_pred, dim=1), head.projections, n_token=392)
        elif len(inputs) == 2:
            x_input_all = self.combine_slow_fast(x[0], x[1])
            cam_gt = self.cam_query_vn_div(x_input_all, F.sigmoid(x_fpv_pred, dim=1), head.projections, n_token=392)
        else:
            raise NotImplementedError
        for pathway in range(self.num_pathways):
            for layer in self.blocks[pathway]:
                x[pathway] = F.relu(layer(x[pathway]))

            x[pathway] = self.score_projection(x[pathway])

        _, _, _, H, W = x[0].shape
        assert H == W
        # x = [x_i.reshape(B, C, T, H * W) for x_i in x]
        loss = cal_cam_loss(x, cam_gt, H, pos_weight=self.pos_weight, mode=self.loss_mode)
        loss["cam_distill_loss"] *= self.loss_weight

        return loss

class CAMLocalBiHead(nn.Module):
    def __init__(
            self,
            dim_in,
            conv_dims=(512,),
            loss_weight=1.0,
            with_targets=True,
            loss_mode="soft",
            pos_weight=1.0,
    ):
        super(CAMLocalBiHead, self).__init__()
        self.loss_weight = loss_weight
        self.with_targets = with_targets
        self.loss_mode = loss_mode
        self.pos_weight = pos_weight

        self.num_pathways = len(dim_in)
        self.build_blocks(dim_in, conv_dims)

    def build_blocks(self, dim_in, conv_dims):
        self.blocks = []
        for pathway in range(self.num_pathways):
            self.blocks.append([])
            in_channels = dim_in[pathway]
            for idx, layer_channels in enumerate(conv_dims):
                module = nn.Conv3d(in_channels, layer_channels, (1, 3, 3), stride=1, padding=(0, 1, 1))
                self.add_module("pathway{}_conv_fcn{}".format(pathway, idx), module)
                self.blocks[pathway].append(module)
                in_channels = layer_channels

        self.score_projection = nn.Conv3d(
            in_channels, 1, (1, 1, 1), stride=(1, 1, 1)
        )

    def combine_slow_fast(self, slow_feat, fast_feat):
        fast_feat_cut = fast_feat.index_select(2, torch.tensor([0, 4, 8, 12, 16, 20, 24, 28]))
        return torch.cat([slow_feat, fast_feat_cut], dim=1)

    def cam_query_vn_div(self, x_ori, pred_ori_v, pred_proj_v, pred_ori_n, pred_proj_n, n_token):
        """
            Qitong on Jul. 25th, 2022.
        """
        bs, c, t, h, w = x_ori[0].size()
        pred_values_v, pred_idxs_v = pred_ori_v.sort(dim=1, descending=True)
        pred_values_n, pred_idxs_n = pred_ori_n.sort(dim=1, descending=True)
        cls_features_v = torch.stack([torch.matmul(pred_proj_v.weight, ele_f.view(c, -1)) for ele_f in x_ori[0]])
        cls_features_n = torch.stack([torch.matmul(pred_proj_n.weight, ele_f.view(c, -1)) for ele_f in x_ori[0]])
        idxs_list_v = []
        idxs_list_n = []
        values_list_v = []
        values_list_n = []
        for cf_v, cf_n, st_x, p_i_v, p_i_n in zip(cls_features_v, cls_features_n, x_ori[0], pred_idxs_v, pred_idxs_n):
            st_x_ = st_x.view(c, -1)
            values_v_map_final_norm = (cf_v[p_i_v[0]] - torch.min(cf_v[p_i_v[0]])) / \
                                      (torch.max(cf_v[p_i_v[0]]) - torch.min(cf_v[p_i_v[0]]))
            values_n_map_final_norm = (cf_n[p_i_n[0]] - torch.min(cf_n[p_i_n[0]])) / \
                                      (torch.max(cf_n[p_i_n[0]]) - torch.min(cf_n[p_i_n[0]]))
            values_v, idxs_v = values_v_map_final_norm.topk(n_token)
            values_n, idxs_n = values_n_map_final_norm.topk(n_token)
            # res_tensor_list.append(st_x_[:, idxs].permute((1, 0)))
            idx_place_fpv_v = []
            for i in idxs_v.detach().cpu().numpy():
                idx_place_fpv_v.append([int(i / 49), int(i % 49 / 7), (i % 49) % 7])
            idxs_list_v.append(idx_place_fpv_v)

            idx_place_fpv_n = []
            for i in idxs_n.detach().cpu().numpy():
                idx_place_fpv_n.append([int(i / 49), int(i % 49 / 7), (i % 49) % 7])
            idxs_list_n.append(idx_place_fpv_n)

            value_all_fpv_v = []
            for i in values_v.detach().cpu().numpy():
                value_all_fpv_v.append(i)
            values_list_v.append(value_all_fpv_v)

            value_all_fpv_n = []
            for i in values_n.detach().cpu().numpy():
                value_all_fpv_n.append(i)
            values_list_n.append(value_all_fpv_n)

        cam_maps_v = np.zeros((bs, t, h, w))
        cam_maps_n = np.zeros((bs, t, h, w))

        bs_idx = 0
        for idx_v, idx_n, values_v, values_n in zip(idxs_list_v, idxs_list_n, values_list_v, values_list_n):
            for j in range(len(idx_v)):
                # print(len(idx_v), len(idx_n), idx_v[j])
                t, h, w = idx_v[j]
                v = values_v[j]
                cam_maps_v[bs_idx][t][h][w] = v
            for j in range(len(idx_n)):
                t, h, w = idx_n[j]
                v = values_n[j]
                cam_maps_n[bs_idx][t][h][w] = v
            bs_idx += 1
        # return (torch.tensor(cam_maps_v).cuda().detach() + torch.tensor(cam_maps_n).cuda().detach()) / 2
        cam_v_final = torch.tensor(cam_maps_v).cuda().detach()
        cam_n_final = torch.tensor(cam_maps_n).cuda().detach()
        return torch.where(cam_v_final > cam_n_final, cam_v_final, cam_n_final)

    def cam_query_vn_div_2(self, x_ori, pred_ori_v, pred_proj_v, pred_ori_n, pred_proj_n, n_token):
        """
            Qitong on Jul. 25th, 2022.
        """
        bs, c, t, h, w = x_ori[0].size()
        pred_values_v, pred_idxs_v = pred_ori_v.sort(dim=1, descending=True)
        pred_values_n, pred_idxs_n = pred_ori_n.sort(dim=1, descending=True)
        cls_features_v = torch.stack([torch.matmul(pred_proj_v.weight, ele_f.view(c, -1)) for ele_f in x_ori[0]])
        cls_features_n = torch.stack([torch.matmul(pred_proj_n.weight, ele_f.view(c, -1)) for ele_f in x_ori[0]])
        idxs_list_v = []
        idxs_list_n = []
        values_list_v = []
        values_list_n = []
        for cf_v, cf_n, st_x, p_i_v, p_i_n, v_i_v, v_i_n in zip(cls_features_v, cls_features_n, x_ori[0],
                                                                pred_idxs_v, pred_idxs_n,
                                                                pred_values_v, pred_values_n):
            st_x_ = st_x.view(c, -1)
            # for si in p_i_v.size()[0]:
            values_v_map_final_norm = (cf_v[p_i_v[0]] - torch.min(cf_v[p_i_v[0]])) / \
                                      (torch.max(cf_v[p_i_v[0]]) - torch.min(cf_v[p_i_v[0]]))
            values_n_map_final_norm = (cf_n[p_i_n[0]] - torch.min(cf_n[p_i_n[0]])) / \
                                      (torch.max(cf_n[p_i_n[0]]) - torch.min(cf_n[p_i_n[0]]))
            values_v, idxs_v = values_v_map_final_norm.topk(n_token)
            values_n, idxs_n = values_n_map_final_norm.topk(n_token)
            # res_tensor_list.append(st_x_[:, idxs].permute((1, 0)))
            idx_place_fpv_v = []
            for i in idxs_v.detach().cpu().numpy():
                idx_place_fpv_v.append([int(i / 49), int(i % 49 / 7), (i % 49) % 7])
            idxs_list_v.append(idx_place_fpv_v)

            idx_place_fpv_n = []
            for i in idxs_n.detach().cpu().numpy():
                idx_place_fpv_n.append([int(i / 49), int(i % 49 / 7), (i % 49) % 7])
            idxs_list_n.append(idx_place_fpv_n)

            value_all_fpv_v = []
            for i in values_v.detach().cpu().numpy():
                value_all_fpv_v.append(i)
            values_list_v.append(value_all_fpv_v)

            value_all_fpv_n = []
            for i in values_n.detach().cpu().numpy():
                value_all_fpv_n.append(i)
            values_list_n.append(value_all_fpv_n)

        cam_maps_v = np.zeros((bs, t, h, w))
        cam_maps_n = np.zeros((bs, t, h, w))

        bs_idx = 0
        for idx_v, idx_n, values_v, values_n in zip(idxs_list_v, idxs_list_n, values_list_v, values_list_n):
            for j in range(len(idx_v)):
                # print(len(idx_v), len(idx_n), idx_v[j])
                t, h, w = idx_v[j]
                v = values_v[j]
                cam_maps_v[bs_idx][t][h][w] = v
            for j in range(len(idx_n)):
                t, h, w = idx_n[j]
                v = values_n[j]
                cam_maps_n[bs_idx][t][h][w] = v
            bs_idx += 1
        # return (torch.tensor(cam_maps_v).cuda().detach() + torch.tensor(cam_maps_n).cuda().detach()) / 2
        cam_v_final = torch.tensor(cam_maps_v).cuda().detach()
        cam_n_final = torch.tensor(cam_maps_n).cuda().detach()
        return torch.where(cam_v_final > cam_n_final, cam_v_final, cam_n_final)

    def forward(self, inputs, head, x_fpv_pred, meta=None):
        x = inputs
        if len(inputs) == 1:
            cam_gt = self.cam_query_vn_div(x, F.softmax(x_fpv_pred[0], dim=1), head.projections[0],
                                           F.softmax(x_fpv_pred[1], dim=1), head.projections[1], n_token=392)
        elif len(inputs) == 2:
            x_input_all = self.combine_slow_fast(x[0], x[1])
            cam_gt = self.cam_query_vn_div(x_input_all, F.softmax(x_fpv_pred[0], dim=1), head.projections[0],
                                           F.softmax(x_fpv_pred[1], dim=1), head.projections[1], n_token=392)
        else:
            raise NotImplementedError
        for pathway in range(self.num_pathways):
            for layer in self.blocks[pathway]:
                x[pathway] = F.relu(layer(x[pathway]))

            x[pathway] = self.score_projection(x[pathway])

        _, _, _, H, W = x[0].shape
        assert H == W
        # x = [x_i.reshape(B, C, T, H * W) for x_i in x]
        loss = cal_cam_loss(x, cam_gt, H, pos_weight=self.pos_weight, mode=self.loss_mode)
        loss["cam_distill_loss"] *= self.loss_weight

        return loss