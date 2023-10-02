#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

SMALL_NUM = np.log(1e-50)

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]


def loss_fn_kd(outputs, teacher_outputs, alpha=0.0, T=1.0, weight=None, mode="none", cfg=None):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    if mode == "none":
        KL_loss = nn.KLDivLoss(reduction="batchmean",)(
            F.log_softmax(outputs/T, dim=-1),
            F.softmax(teacher_outputs/T, dim=-1),
        ) * (alpha * T * T) 
    else:
        KL_loss = nn.KLDivLoss(reduction="none",)(
            F.log_softmax(outputs/T, dim=-1),
            F.softmax(teacher_outputs/T, dim=-1),
        ) * (alpha * T * T) 
        if mode in ["ego", "nonego"] :
            assert weight is not None
            # print(KL_loss.shape, weight.shape)
            w = F.softmax(weight, dim=-1)
            w = w[:, 1] if mode == "ego" else w[:, 0]
            # w[:] = 1.0
            KL_loss = KL_loss * w.view(-1, 1)
            KL_loss = torch.sum(KL_loss) / torch.sum(w)
        elif mode in ["sample_ego", "sample_nonego"]:
            w = F.softmax(weight, dim=-1)
            w = w[:, 1] if mode == "sample_ego" else w[:, 0]

            n = int(cfg.KD.SAMPLE_RATIO * outputs.shape[0])
            _, idx = torch.topk(w, k=n)

            KL_loss = torch.sum(KL_loss[idx, :]) / n
        elif mode in ["sample_entropy"]:
            w = F.softmax(teacher_outputs/T, dim=-1)
            entropy = torch.distributions.Categorical(w).entropy()

            n = int(cfg.KD.SAMPLE_RATIO * outputs.shape[0])
            _, idx = torch.topk(entropy, k=n, largest=False)

            KL_loss = torch.sum(KL_loss[idx, :]) / n
        else:
            raise NotImplementedError

    return KL_loss


class GCELoss(nn.Module):
    """
        Qitong on May 6th, 2022.
        ref: https://zhuanlan.zhihu.com/p/420913134
    """
    def __init__(self, num_classes=10, q=0.7, eps=1e-20):
        super(GCELoss, self).__init__()
        self.q = q
        self.eps = eps
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=self.eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        # print(label_one_hot)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()


class pNorm(nn.Module):
    def __init__(self, p=0.5):
        super(pNorm, self).__init__()
        self.p = p

    def forward(self, pred, p=None):
        if p:
            self.p = p
        # pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1)

        # 一个很简单的正则
        norm = torch.sum(pred ** self.p, dim=1)
        return norm.mean()

class DCL(object):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, weight_fn=None):
        super(DCL, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def __call__(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
            # print((z1*z2).size(), 1)
            # print((z1@z2.T).size(), 2)
            # print(self.weight_fn(z1, z2))
            # print((z1 * z2).sum(dim=1), 233)
        # print(torch.mm(z1, z1.t()).size(), cross_view_distance.size())
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        # print(neg_similarity.size(), neg_mask.size())
        # print(neg_similarity + neg_mask * SMALL_NUM)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()

class DCL_2(object):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, weight_fn=None):
        super(DCL_2, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def _get_off_diag(self, m):
        n = m.size()[0]
        return m.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)

    def __call__(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity_2 = torch.cat((self._get_off_diag(torch.mm(z1, z1.t())),
                                      self._get_off_diag(cross_view_distance)), dim=1) / self.temperature
        negative_loss = torch.logsumexp(neg_similarity_2, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()

class DCL_3(object):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, weight_fn=None):
        super(DCL_3, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def _get_off_diag(self, m):
        n = m.size()[0]
        return m.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)

    def __call__(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        if z1.size(0) < z2.size(0):
            cross_view_distance = torch.mm(z1, z2.t())
            anchor_bs = cross_view_distance.size(0)
            positive_loss = -torch.diag(cross_view_distance[:, :anchor_bs]) / self.temperature
            if self.weight_fn is not None:
                positive_loss = positive_loss * self.weight_fn(z1, z2)
            neg_1_sims = self._get_off_diag(torch.mm(z1, z1.t()))
            neg_2_sims = self._get_off_diag(cross_view_distance[:, :anchor_bs])
            neg_3_sims = cross_view_distance[:, anchor_bs:]
            print(neg_1_sims.size(), neg_2_sims.size(), neg_3_sims.size())
            neg_similarity_2 = torch.cat([neg_1_sims, neg_2_sims, neg_3_sims], dim=1) / self.temperature
            negative_loss = torch.logsumexp(neg_similarity_2, dim=1, keepdim=False)
            return (positive_loss + negative_loss).mean()
        elif z1.size(0) > z2.size(0):
            cross_view_distance = torch.mm(z1, z2.t())
            anchor_bs = cross_view_distance.size(1)
            positive_loss = -torch.diag(cross_view_distance[:anchor_bs, :]) / self.temperature
            if self.weight_fn is not None:
                positive_loss = positive_loss * self.weight_fn(z1, z2)
            neg_1_sims = self._get_off_diag(torch.mm(z1, z1.t()))
            neg_2_sims = self._get_off_diag(cross_view_distance[:anchor_bs, :])
            neg_3_sims = cross_view_distance[anchor_bs:, :]
            print(neg_1_sims.size(), neg_2_sims.size(), neg_3_sims.size())
            # neg_similarity_2 = torch.cat([neg_2_sims, neg_3_sims], dim=1) / self.temperature
            # print(positive_loss.size(), torch.logsumexp(neg_similarity_2, dim=1, keepdim=False).size())
            # negative_loss = torch.logsumexp(neg_similarity_2, dim=1, keepdim=False)
            negative_loss_1 = torch.logsumexp(neg_1_sims, dim=1, keepdim=False)
            negative_loss_2 = torch.logsumexp(neg_2_sims, dim=1, keepdim=False)
            negative_loss_3 = torch.logsumexp(neg_3_sims, dim=1, keepdim=False)
            # print(negative_loss_2.size())
            return positive_loss.mean() + \
                   (negative_loss_1.mean() + negative_loss_2.mean() + negative_loss_3.mean()) / 3
        else:
            raise NotImplementedError("Go other DCL scenarios!!!")

class DCLW(DCL):
    """
    Decoupled Contrastive Loss with negative von Mises-Fisher weighting proposed in https://arxiv.org/pdf/2110.06848.pdf
    sigma: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """
    def __init__(self, sigma=0.5, temperature=0.1):
        # weight_fn = lambda z1, z2: 2 - z1.size(0) * F.softmax((z1 * z2).sum(dim=1) / sigma, dim=0).squeeze()
        # weight_fn = lambda z1, z2: 2 + z1.size(0) * torch.diagonal(z1 @ z2.T)
        weight_fn = lambda z1, z2: (2 + z1.size(0) * torch.diagonal(z1 @ z2.T)) / 2
        # weight_fn = lambda z1, z2: z1.size(0) * F.softmax((z1 * z2).sum(dim=1) / sigma, dim=0).squeeze()
        super(DCLW, self).__init__(weight_fn=weight_fn, temperature=temperature)

class DCL_OS(object):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, sigma=0.5, temperature=0.1):
        super(DCL_OS, self).__init__()
        self.temperature = temperature
        self.sigma = sigma

    def weight_fn(self, v1, v2):
        """ori from DCL"""
        res = v1.size(0) * F.softmax((v1 * v2).sum(dim=1) / self.sigma, dim=0).squeeze()
        # res = v1.size(0) * F.softmax(torch.diagonal(v1 @ v2.T) / self.sigma, dim=0).squeeze()
        # print((v1 * v2).size(), (v1 * v2).sum(dim=1).size())
        """mine with cos sim"""
        # res = 1 + torch.diagonal(v1 @ v2.T)
        # res = (1 + torch.diagonal(v1 @ v2.T)) / 2
        return res

    def __call__(self, z1, z2, n1, n2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        # positive_loss = positive_loss * self.weight_fn(n1, n2)
        positive_loss = positive_loss
        # print(self.weight_fn(n1, n2), 'dcl weight check~', self.sigma)
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()