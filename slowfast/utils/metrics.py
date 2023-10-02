#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""

import numpy as np
import torch


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    try:
        topks_correct = [top_max_k_correct[:k, :].view(-1).float().sum() for k in ks]
    except:
        topks_correct = [top_max_k_correct[:k, :].contiguous().view(-1).float().sum() for k in ks]
    return topks_correct


def multitask_topks_correct(preds, labels, ks=(1,)):
    """
    Args:
        preds: tuple(torch.FloatTensor), each tensor should be of shape
            [batch_size, class_count], class_count can vary on a per task basis, i.e.
            outputs[i].shape[1] can be different to outputs[j].shape[j].
        labels: tuple(torch.LongTensor), each tensor should be of shape [batch_size]
        ks: tuple(int), compute accuracy at top-k for the values of k specified
            in this parameter.
    Returns:
        tuple(float), same length at topk with the corresponding accuracy@k in.
    """
    max_k = int(np.max(ks))
    task_count = len(preds)
    batch_size = labels[0].size(0)
    all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor)
    if torch.cuda.is_available():
        all_correct = all_correct.cuda()
    for output, label in zip(preds, labels):
        _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
        # Flip batch_size, class_count as .view doesn't work on non-contiguous
        max_k_idx = max_k_idx.t()
        correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
        all_correct.add_(correct_for_task)

    multitask_topks_correct = [
        torch.ge(all_correct[:k].float().sum(0), task_count).float().sum(0) for k in ks
    ]

    return multitask_topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]

from slowfast.datasets.lemma_bi import Metadata

@torch.no_grad()
def gen_pred_labels(composed_labels):
    act_num = 3
    obj_num = 3
    batch_size = composed_labels.size(0)
    act_pred_labels = torch.zeros(batch_size, len(Metadata.action)).cuda(non_blocking=True)
    obj_pred_labels = torch.zeros(batch_size, act_num, obj_num, len(Metadata.object)).cuda(non_blocking=True)
    batch_dict = dict()
    _, hoi_indices = torch.topk(composed_labels, 3, dim=-1)
    for batch_idx in range(hoi_indices.size(0)):
        for class_idx in hoi_indices[batch_idx]:
            if batch_idx not in batch_dict.keys():
                batch_dict[batch_idx] = []
            hoi_str = Metadata.hoi[class_idx]
            act_str, obj_str = hoi_str.split('$')
            pos_strs = obj_str.split(':')
            objs = []
            for pos_idx, pos_objs in enumerate(pos_strs):
                if pos_objs != '':
                    pos_obj_ids = [Metadata.object_index[x] for x in pos_objs.split('@')[1].split('|')]
                    objs.append(pos_obj_ids)
            batch_dict[batch_idx].append([Metadata.action_index[act_str], objs])

    for batch_idx in batch_dict.keys():
        batch_dict[batch_idx] = sorted(batch_dict[batch_idx], key=lambda x : x[0])
        for a_idx, a_info in enumerate(batch_dict[batch_idx]):
            act_id, objs = a_info
            act_pred_labels[batch_idx][act_id] = 1.0
            for pos_idx, o_info in enumerate(objs):
                for obj in o_info:
                    obj_pred_labels[batch_idx][a_idx][pos_idx][obj] = 1.0

    return act_pred_labels, obj_pred_labels

@torch.no_grad()
def binary_correct(pred, labels, thres=0.5, meta=None):
    pred_ = (pred >= thres).type(torch.FloatTensor).cuda()
    truth = (labels >= thres).type(torch.FloatTensor).cuda()
    tp = pred_.mul(truth).sum()
    tn = (1 - pred_).mul(1 - truth).sum()
    fp = pred_.mul(1 - truth).sum()
    fn = (1 - pred_).mul(truth).sum()
    return tp, tn, fp, fn

@torch.no_grad()
def eval_pred(pred, labels, meta, cfg):
    label_logits, act_logits, object_logits, task_logits, label_features = pred
    pred_labels = (torch.sigmoid(label_logits) >= cfg.TEST.THRES).type(torch.float32).cuda()
    tp, tn, fp, fn = binary_correct(pred_labels, labels, cfg.TEST.THRES)

    # Plain label, generate predicted labels
    if isinstance(act_logits, list) or isinstance(object_logits, list):
        act_pred_labels, obj_pred_labels = gen_pred_labels(pred_labels)
    else:
        act_num = act_logits.size(-2)
        _, act_pred_labels = torch.max(act_logits.view(-1, act_logits.size(-1)), dim=-1)
        act_pred_labels = act_pred_labels.view(-1, act_num, 1)
        act_pred_labels = (torch.sum(torch.zeros(act_pred_labels.size(0), act_pred_labels.size(1), len(Metadata.action)).cuda().scatter_(-1, act_pred_labels, 1), dim=1) > 0).type(torch.float)
        obj_pred_labels = (torch.sigmoid(object_logits) >= cfg.TEST.THRES).type(torch.float32).cuda()
    gt_act_labels = meta['action_labels']
    gt_obj_labels = meta['obj_labels']
    return tp, tn, fp, fn, pred_labels, act_pred_labels, obj_pred_labels, gt_act_labels, gt_obj_labels
