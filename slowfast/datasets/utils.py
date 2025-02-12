#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import logging
import numpy as np
import os
import random
import time
from collections import defaultdict
import cv2
import torch
from fvcore.common.file_io import PathManager
# Qitong on Mar. 10th, 2022.
from skimage.segmentation import slic
from skimage.measure import regionprops

from . import transform as transform

logger = logging.getLogger(__name__)


def retry_load_images(image_paths, retry=10, backend="pytorch"):
    """
    This function is to load images with support of retrying for failed load.

    Args:
        image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`.

    Returns:
        imgs (list): list of loaded images.
    """
    for i in range(retry):
        imgs = []
        for image_path in image_paths:
            with PathManager.open(image_path, "rb") as f:
                img_str = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
            imgs.append(img)

        if all(img is not None for img in imgs):
            if backend == "pytorch":
                imgs = torch.as_tensor(np.stack(imgs))
            return imgs
        else:
            logger.warn("Reading failed. Will retry.")
            time.sleep(1.0)
        if i == retry - 1:
            raise Exception("Failed to load images {}".format(image_paths))


def retry_load_images_np(image_paths, retry=10, backend="pytorch"):
    """
        Qitong on Mar. 5th, 2022.
    """
    for i in range(retry):
        imgs = []
        for image_path in image_paths:
            with PathManager.open(image_path, "rb") as f:
                img_str = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
            imgs.append(img)

        if all(img is not None for img in imgs):
            return np.stack(imgs)
        else:
            logger.warn("Reading failed. Will retry.")
            time.sleep(1.0)
        if i == retry - 1:
            raise Exception("Failed to load images {}".format(image_paths))


def get_sequence(center_idx, half_len, sample_rate, num_frames):
    """
    Sample frames among the corresponding clip.

    Args:
        center_idx (int): center frame idx for current clip
        half_len (int): half of the clip length
        sample_rate (int): sampling rate for sampling frames inside of the clip
        num_frames (int): number of expected sampled frames

    Returns:
        seq (list): list of indexes of sampled frames in this clip.
    """
    seq = list(range(center_idx - half_len, center_idx + half_len, sample_rate))

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1
    return seq


def pack_pathway_output(cfg, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if cfg.DATA.REVERSE_INPUT_CHANNEL:
        frames = frames[[2, 1, 0], :, :, :]
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frame_list = [frames]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                cfg.MODEL.ARCH,
                cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frame_list

def my_pack_pathway_output(arch, reverse_input_channel,
                           single_pathway_arch, multi_pathway_arch, slowfast_alpha, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if reverse_input_channel:
        frames = frames[[2, 1, 0], :, :, :]
    if arch in single_pathway_arch:
        frame_list = [frames]
    elif arch in multi_pathway_arch:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                arch,
                single_pathway_arch + multi_pathway_arch,
            )
        )
    return frame_list


# def pack_pathway_output(cfg_dict, frames):
#     """
#     Qitong on Jan. 24th, 2022.
#     """
#     if cfg_dict['DATA.REVERSE_INPUT_CHANNEL']:
#         frames = frames[[2, 1, 0], :, :, :]
#     if cfg_dict['MODEL.ARCH'] in cfg_dict['MODEL.SINGLE_PATHWAY_ARCH']:
#         frame_list = [frames]
#     elif cfg_dict['MODEL.ARCH'] in cfg_dict['MODEL.MULTI_PATHWAY_ARCH']:
#         fast_pathway = frames
#         # Perform temporal sampling from the fast pathway.
#         slow_pathway = torch.index_select(
#             frames,
#             1,
#             torch.linspace(
#                 0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
#             ).long(),
#         )
#         frame_list = [slow_pathway, fast_pathway]
#     else:
#         raise NotImplementedError(
#             "Model arch {} is not in {}".format(
#                 cfg.MODEL.ARCH,
#                 cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
#             )
#         )
#     return frame_list


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        frames, _ = transform.random_short_side_scale_jitter(
            images=frames,
            min_size=min_scale,
            max_size=max_scale,
            inverse_uniform_sampling=inverse_uniform_sampling,
        )
        frames, _ = transform.random_crop(frames, crop_size)
        if random_horizontal_flip:
            frames, _ = transform.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = transform.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def as_binary_vector(labels, num_classes):
    """
    Construct binary label vector given a list of label indices.
    Args:
        labels (list): The input label list.
        num_classes (int): Number of classes of the label vector.
    Returns:
        labels (numpy array): the resulting binary vector.
    """
    label_arr = np.zeros((num_classes,))

    for lbl in set(labels):
        label_arr[lbl] = 1.0
    return label_arr


def aggregate_labels(label_list):
    """
    Join a list of label list.
    Args:
        labels (list): The input label list.
    Returns:
        labels (list): The joint list of all lists in input.
    """
    all_labels = []
    for labels in label_list:
        for l in labels:
            all_labels.append(l)
    return list(set(all_labels))


def convert_to_video_level_labels(labels):
    """
    Aggregate annotations from all frames of a video to form video-level labels.
    Args:
        labels (list): The input label list.
    Returns:
        labels (list): Same as input, but with each label replaced by
        a video-level one.
    """
    for video_id in range(len(labels)):
        video_level_labels = aggregate_labels(labels[video_id])
        for i in range(len(labels[video_id])):
            labels[video_id][i] = video_level_labels
    return labels


def load_image_lists(frame_list_file, prefix="", return_list=False):
    """
    Load image paths and labels from a "frame list".
    Each line of the frame list contains:
    `original_vido_id video_id frame_id path labels`
    Args:
        frame_list_file (string): path to the frame list.
        prefix (str): the prefix for the path.
        return_list (bool): if True, return a list. If False, return a dict.
    Returns:
        image_paths (list or dict): list of list containing path to each frame.
            If return_list is False, then return in a dict form.
        labels (list or dict): list of list containing label of each frame.
            If return_list is False, then return in a dict form.
    """
    image_paths = defaultdict(list)
    labels = defaultdict(list)
    with PathManager.open(frame_list_file, "r") as f:
        assert f.readline().startswith("original_vid")
        for line in f:
            row = line.split()
            # original_vido_id video_id frame_id path labels
            assert len(row) == 5
            video_name = row[0]
            if prefix == "":
                path = row[3]
            else:
                path = os.path.join(prefix, row[3])
            image_paths[video_name].append(path)
            frame_labels = row[-1].replace('"', "")
            if frame_labels != "":
                labels[video_name].append(
                    [int(x) for x in frame_labels.split(",")]
                )
            else:
                labels[video_name].append([])

    if return_list:
        keys = image_paths.keys()
        image_paths = [image_paths[key] for key in keys]
        labels = [labels[key] for key in keys]
        return image_paths, labels
    return dict(image_paths), dict(labels)


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def get_random_sampling_rate(long_cycle_sampling_rate, sampling_rate):
    """
    When multigrid training uses a fewer number of frames, we randomly
    increase the sampling rate so that some clips cover the original span.
    """
    if long_cycle_sampling_rate > 0:
        assert long_cycle_sampling_rate >= sampling_rate
        return random.randint(sampling_rate, long_cycle_sampling_rate)
    else:
        return sampling_rate


def revert_tensor_normalize(tensor, mean, std):
    """
    Revert normalization for a given tensor by multiplying by the std and adding the mean.
    Args:
        tensor (tensor): tensor to revert normalization.
        mean (tensor or list): mean value to add.
        std (tensor or list): std to multiply.
    """
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor * std
    tensor = tensor + mean
    return tensor

def py_nms(dets, thresh):
    """Pure Python NMS baseline."""
    # print(dets.shape)
    #x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    #每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #order是按照score降序排序的
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        #计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        #找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep

def get_center_frame_boxes(frames, num_bbox, bbox_switch,
                           random_min_square=0.1, random_max_square=0.5, random_hw_ratio=2.0,
                           slic_n_segments=16, slic_compactness=16, min_relative_square=0.05, nms_thresh=0.5):
    """

    Qitong on Mar. 5th, 2022.

    Args:
        frames: (T * H * W * C) one video clips; type: np.array
        num_bbox: (int) number of bbox.
        bbox_switch: (int) type of bboxes; refer: https://arxiv.org/pdf/2112.05181.pdf

    Returns:
        bboxes: (num_bbox, 5) multiple bboxes

    """
    video_height, video_width = frames.shape[1], frames.shape[2]
    video_frame_square = video_height * video_height
    bboxes = []
    if bbox_switch == 1:
        """random bboxes"""
        n_bbox = 0
        while n_bbox < num_bbox:
            bbox_x1 = float(random.randint(0, video_height))
            bbox_y1 = float(random.randint(0, video_width))
            bbox_x2 = float(random.randint(bbox_x1, video_height))
            bbox_y2 = float(random.randint(bbox_y1, video_width))
            bbox_height = bbox_x2 - bbox_x1
            bbox_width = bbox_y2 - bbox_y1
            if bbox_height * bbox_width > random_min_square * video_frame_square \
                    and bbox_height * bbox_width < random_max_square * video_frame_square \
                    and bbox_height / bbox_width > (1 / random_hw_ratio) \
                    and bbox_height / bbox_width < random_hw_ratio:
                bboxes.append([bbox_x1, bbox_y1, bbox_x2, bbox_y2, 1.0])
                n_bbox += 1
        return np.array(bboxes)
    elif bbox_switch == 2:
        """SLIC bboxes"""
        ci = int(frames.shape[0] / 2)
        H, W = frames.shape[1], frames.shape[2]
        img = frames[ci]
        # print(ci, img.shape)
        segments = slic(img, n_segments=slic_n_segments, compactness=slic_compactness)

        an = 0
        ast_bboxes = []
        for region in regionprops(segments):
            minr, minc, maxr, maxc = region.bbox
            if (maxc - minc) * (maxr - minr) > min_relative_square * H * W:
                ast_bboxes.append([float(minr), float(minc), float(maxr), float(maxc), 1.0])
            an += 1
        bst_bboxes_idx = py_nms(np.array(ast_bboxes), thresh=nms_thresh)
        bboxes = [ast_bboxes[i] for i in bst_bboxes_idx]
        if np.array(bboxes).shape[0] >= num_bbox:
            return np.array(bboxes)[:num_bbox]
        else:
            """extra random bboxes"""
            n_bbox = np.array(bboxes).shape[0]
            while n_bbox < num_bbox:
                bbox_x1 = float(random.randint(0, video_height))
                bbox_y1 = float(random.randint(0, video_width))
                bbox_x2 = float(random.randint(bbox_x1, video_height))
                bbox_y2 = float(random.randint(bbox_y1, video_width))
                bbox_height = bbox_x2 - bbox_x1
                bbox_width = bbox_y2 - bbox_y1
                if bbox_height * bbox_width > random_min_square * video_frame_square \
                        and bbox_height * bbox_width < random_max_square * video_frame_square \
                        and bbox_height / bbox_width > (1 / random_hw_ratio) \
                        and bbox_height / bbox_width < random_hw_ratio:
                    bboxes.append([bbox_x1, bbox_y1, bbox_x2, bbox_y2, 1.0])
                    n_bbox += 1
            return np.array(bboxes)

    elif bbox_switch == 3:
        """Graph-based Segmentation bboxes"""
        pass
    else:
        raise NotImplementedError

import pickle, json
from slowfast.datasets.lemma_tpv import Metadata
from pathlib import Path
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.structures.instances import Instances

def parse_gt_result(path, cfg):
    symbols = ['vid_name', 'frame_id', 'pid', 'flow', 'rgb', 'bbox', 'action', 'task', 'hoi']
    with (path).open('rb') as f:
        frame_infos = pickle.load(f)
    frame_paths = {}
    gt_bboxes_labels = {}
    gt_tasks = {}
    vid_name_to_idx = {}
    vid_idx_to_name = []
    for frame_info in frame_infos:
        vid_name = frame_info[symbols.index('vid_name')]
        # print(vid_name)
        frame_id = int(frame_info[symbols.index('frame_id')])
        bbox = frame_info[symbols.index('bbox')]
        task = frame_info[symbols.index('task')]
        pid = frame_info[symbols.index('pid')]
        label = frame_info[symbols.index('action')]
        hoi = frame_info[symbols.index('hoi')]
        if vid_name not in vid_name_to_idx.keys():
            idx = len(vid_name_to_idx)
            vid_name_to_idx[vid_name] = idx
            vid_idx_to_name.append(vid_name)
            gt_bboxes_labels[vid_name] = {}
            frame_paths[vid_name] = {}
            gt_tasks[vid_name] = {}
        if frame_id not in gt_bboxes_labels[vid_name].keys():
            gt_bboxes_labels[vid_name][frame_id] = [[bbox, label, task, hoi, pid]]
        else:
            gt_bboxes_labels[vid_name][frame_id].append([bbox, label, task, hoi, pid])
        if frame_id not in frame_paths[vid_name].keys():
            frame_paths[vid_name][frame_id] = frame_info[symbols.index(cfg.EXP.IMG_TYPE)]
    return frame_paths, gt_bboxes_labels, vid_idx_to_name, vid_name_to_idx

def my_parse_gt_result(path, image_type):
    symbols = ['vid_name', 'frame_id', 'pid', 'flow', 'rgb', 'bbox', 'action', 'task', 'hoi']
    with (path).open('rb') as f:
        frame_infos = pickle.load(f)
    frame_paths = {}
    gt_bboxes_labels = {}
    gt_tasks = {}
    vid_name_to_idx = {}
    vid_idx_to_name = []
    for frame_info in frame_infos:
        vid_name = frame_info[symbols.index('vid_name')]
        # print(vid_name)
        frame_id = int(frame_info[symbols.index('frame_id')])
        bbox = frame_info[symbols.index('bbox')]
        task = frame_info[symbols.index('task')]
        pid = frame_info[symbols.index('pid')]
        label = frame_info[symbols.index('action')]
        hoi = frame_info[symbols.index('hoi')]
        if vid_name not in vid_name_to_idx.keys():
            idx = len(vid_name_to_idx)
            vid_name_to_idx[vid_name] = idx
            vid_idx_to_name.append(vid_name)
            gt_bboxes_labels[vid_name] = {}
            frame_paths[vid_name] = {}
            gt_tasks[vid_name] = {}
        if frame_id not in gt_bboxes_labels[vid_name].keys():
            gt_bboxes_labels[vid_name][frame_id] = [[bbox, label, task, hoi, pid]]
        else:
            gt_bboxes_labels[vid_name][frame_id].append([bbox, label, task, hoi, pid])
        if frame_id not in frame_paths[vid_name].keys():
            frame_paths[vid_name][frame_id] = frame_info[symbols.index(image_type)]
    return frame_paths, gt_bboxes_labels, vid_idx_to_name, vid_name_to_idx

def get_selected_indices(frame_paths, bboxes_labels, vid_name_to_idx, cfg, mode, anno_rel_path, split_file=None):
    if split_file is not None:
        with split_file.open('r') as f:
            split = json.load(f)
    else:
        split = list()
    selected_indices = list()
    split += [vid_name + '|{}'.format(x) for x in ['P1', 'P2'] for vid_name in split]
    # print(split, mode)
    # print(frame_paths)

    if mode == 'train':
        with (
                Path("/usa/wqtwjt/LEMMA/LEMMA_anno/annotations")
                / anno_rel_path / '{}_fpv_segments_indices.p'.format(cfg.EXP.TASK, cfg.EXP.VIEW_TYPE)
            ).open('rb') as f:
            indices = pickle.load(f)
        for pair in indices:
            video_name, seg_start, length = pair
            if len(split) != 0 and video_name not in split:
                continue
            flag = False
            for (bbox, label, task, hoi, pid) in bboxes_labels[video_name][seg_start]:
                if Metadata.hoi_index["null$"] not in hoi:
                    flag = True
            if flag:
                for _ in range(3):
                    selected_indices.append((vid_name_to_idx[video_name], seg_start, length))
        # print(mode, len(selected_indices))
    else:
        if mode == 'train':
            target_fps = 4
        elif mode == 'all':
            target_fps = 5
        else:
            target_fps = 5
        sample_freq = 24 // target_fps
        test_count = set()
        # import time
        for video_name, frames in frame_paths.items():
            # print(frame_paths.keys())
            # print(type(frames))
            # time.sleep(0.5)
            if len(split) != 0 and video_name not in split:
                continue
            test_count.add(video_name)
            frame_paths[video_name] = [x for _, x in frame_paths[video_name].items()]
            num_frames = len(frames)
            for i in range(num_frames):
                if (i + 1) % sample_freq == 0:

                    # Generating Non-null actions
                    # flag = False
                    # for (bbox, label, task, hoi, pid) in bboxes_labels[video_name][i]:
                    #     # print(hoi)
                    #     if mode == 'train':
                    #         if Metadata.hoi_index["null$"] not in hoi:
                    #             flag = True
                    #     else:
                    #         flag = True
                    # if flag:
                    #     # print(2333333)
                    #     selected_indices.append((vid_name_to_idx[video_name], i, -1))
                    selected_indices.append((vid_name_to_idx[video_name], i, -1))
        # for ii in range(len(selected_indices)):
        #     print(selected_indices[ii])
        #     if ii > 10:
        #         break
        # print(mode, len(selected_indices))
        # print(Metadata.hoi_index)
    return selected_indices

def my_get_selected_indices(frame_paths, bboxes_labels, vid_name_to_idx, mode, anno_rel_path,
                            split_file=None, task="rec"):
    if split_file is not None:
        with split_file.open('r') as f:
            split = json.load(f)
    else:
        split = list()
    selected_indices = list()
    split += [vid_name + '|{}'.format(x) for x in ['P1', 'P2'] for vid_name in split]
    # print(split, mode)
    # print(frame_paths)

    if mode == 'train':
        with (
                Path("/usa/wqtwjt/LEMMA/LEMMA_anno/annotations")
                / anno_rel_path / '{}_fpv_segments_indices.p'.format(task, "tpv")
            ).open('rb') as f:
            indices = pickle.load(f)
        for pair in indices:
            video_name, seg_start, length = pair
            if len(split) != 0 and video_name not in split:
                continue
            flag = False
            for (bbox, label, task, hoi, pid) in bboxes_labels[video_name][seg_start]:
                if Metadata.hoi_index["null$"] not in hoi:
                    flag = True
            if flag:
                for _ in range(3):
                    # print(1)
                    selected_indices.append((vid_name_to_idx[video_name], seg_start, length))
        # print(mode, len(selected_indices))
    else:
        if mode == 'train':
            target_fps = 4
        elif mode == 'all':
            target_fps = 5
        else:
            target_fps = 5
        sample_freq = 24 // target_fps
        test_count = set()
        # import time
        for video_name, frames in frame_paths.items():
            # print(frame_paths.keys())
            # print(type(frames))
            # time.sleep(0.5)
            if len(split) != 0 and video_name not in split:
                continue
            test_count.add(video_name)
            frame_paths[video_name] = [x for _, x in frame_paths[video_name].items()]
            num_frames = len(frames)
            for i in range(num_frames):
                if (i + 1) % sample_freq == 0:

                    # Generating Non-null actions
                    # flag = False
                    # for (bbox, label, task, hoi, pid) in bboxes_labels[video_name][i]:
                    #     # print(hoi)
                    #     if mode == 'train':
                    #         if Metadata.hoi_index["null$"] not in hoi:
                    #             flag = True
                    #     else:
                    #         flag = True
                    # if flag:
                    #     # print(2333333)
                    #     selected_indices.append((vid_name_to_idx[video_name], i, -1))
                    selected_indices.append((vid_name_to_idx[video_name], i, -1))
        # for ii in range(len(selected_indices)):
        #     print(selected_indices[ii])
        #     if ii > 10:
        #         break
        # print(mode, len(selected_indices))
        # print(Metadata.hoi_index)
    return selected_indices

def temporal_sample_offset(length, target_length):
    start = random.randint(0, max(length - target_length - 1, 0))
    return start

def get_frames(frame_list, src='rgb'):
    if src == 'rgb':
        imgs = [cv2.imread(img_path[0]) for img_path in frame_list]
        if all(img is not None for img in imgs):
            imgs = torch.as_tensor(np.stack(imgs))
        else:
            logger.error('Failed to load {} image'.format(src))
            raise Exception("Failed to load images {}".format(frame_list))
    else:
        img_xs = [cv2.imread(img_path[0], cv2.IMREAD_GRAYSCALE) for img_path in frame_list]
        img_ys = [cv2.imread(img_path[1], cv2.IMREAD_GRAYSCALE) for img_path in frame_list]
        if all(img is not None for img in img_xs) and all(img is not None for img in img_ys):
            # stack flow images
            img_xs = torch.as_tensor(np.stack(img_xs))
            img_ys = torch.as_tensor(np.stack(img_ys))
            print(img_xs)
            imgs = torch.cat((img_xs, img_ys), dim=0)
        else:
            logger.error('Failed to load {} image'.format(src))
            raise Exception('Failed to load {} images {}'.format(src, frame_list))
    return imgs


def to_tensor(x):
    return torch.FloatTensor(x)


def parse_detection_result(path, cfg):
    pred_symbols = ['vid_name', 'frame_id', 'bbox', 'score']
    with path.open('rb') as f:
        pred_infos = pickle.load(f)

    bboxes_labels = {}
    for pred_info in pred_infos:
        vid_name = pred_info[pred_symbols.index('vid_name')]
        frame_id = pred_info[pred_symbols.index('frame_id')]
        bbox = pred_info[pred_symbols.index('bbox')]
        score = pred_info[pred_symbols.index('score')]
        if score < cfg.DATA.DETECTION_SCORE_THRESH:
            continue
        if vid_name not in bboxes_labels.keys():
            bboxes_labels[vid_name] = {}
        if frame_id not in bboxes_labels[vid_name].keys():
            bboxes_labels[vid_name][frame_id] = [[bbox, np.array([-1])]]
        else:
            bboxes_labels[vid_name][frame_id].append([bbox, np.array([-1])])
    return bboxes_labels


def visualize_sequence(img, bboxes, labels, cfg):
    mapping = {'verb': 'action', 'noun': 'object', 'hoi': 'hoi'}
    MetadataCatalog.get('vis').set(thing_classes=getattr(Metadata, mapping[cfg.EXP.LABEL_TYPE]))
    metadata = MetadataCatalog.get('vis')
    classes = list()
    boxes = list()
    for box, label in zip(bboxes, labels):
        for idx, x in enumerate(label):
            if x == 1:
                classes.append(idx)
                boxes.append(box)
    outputs = {"instances": Instances((img.shape[0], img.shape[1]), pred_boxes=boxes, pred_classes=classes)}
    v = Visualizer(img,
                    metadata=metadata,
                    scale=0.8,
                    instance_mode=ColorMode.IMAGE
                )
    vis = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
    plt.imshow(vis)
    plt.show()

def get_lemma_to_epic100_labels():
    # 23 -> 7
    verb_dict = {
        0: 10, 1: 2, 2: 4, 3: 45, 4: 7, 5: 60, 6: 35, 7: 26,
        8: 0, 9: 3, 10: 6, 11: -1, 12: 9, 13: 1, 14: -1, 15: 57,
        16: 88, 17: 67, 18: 13, 19: 8, 20: 6, 21: 24, 22: 2,
        23: 7, 24: -1
    }
    # 1 -> -1
    # 22 -> 19
    # 29 -> 249
    # 30 -> 6
    # 41 -> 201
    # 48 -> 182
    # 51 -> 29
    # 56 -> 61
    # 60 -> 22
    noun_dict = {
        0: 11, 1: -1, 2: -1, 3: -1, 4: 44, 5: 4, 6: 4, 7: 18, 8: 13,
        9: 187, 10: 63, 11: 140, 12: 15, 13: 7,
        14: 33, 15: 99, 16: 3, 17: -1, 18: 221, 19: 59,
        20: 8, 21: 119, 22: 19, 23: 159, 24: 14, 25: 12,
        26: -1, 27: 183, 28: 106, 29: 249, 30: 6,
        31: 6, 32: 249, 33: 120, 34: 6, 35: 28, 36: 90,
        37: 64, 38: 155, 39: 167, 40: 5, 41: 201, 42: 2, 43: 153,
        44: 221, 45: 164, 46: 247, 47: 63, 48: 182, 49: 1, 50: 24, 51: 29,
        52: 6, 53: 132, 54: 88, 55: 43, 56: 61, 57: 245, 58: -1,
        59: 27, 60: 22, 61: 27, 62: 273, 63: 26
    }
    return verb_dict, noun_dict

def get_label_dict():
    f2 = open("/data/remote/Deep_REAL_Datasets/LEMMA/LEMMA_anno/EPIC_many_shot_actions.csv")
    lemma_data = f2.readlines()
    label_dict = {}
    idx = 0
    for ele in lemma_data:
        try:
            n1 = int(ele.split("\"")[1].replace(" ", "").replace("(", "").replace(")", "").split(",")[0])
            n2 = int(ele.split("\"")[1].replace(" ", "").replace("(", "").replace(")", "").split(",")[1])
            label_dict[str([n1, n2])] = idx
            idx += 1
        except:
            pass
    return label_dict, idx