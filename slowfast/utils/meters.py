#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer
from sklearn.metrics import average_precision_score

import slowfast.datasets.ava_helper as ava_helper
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.utils.ava_eval_helper import evaluate_ava, read_csv, read_exclusions, read_labelmap

from slowfast.datasets.lemma_tpv import Metadata
import json
import sklearn.metrics as sk_metrics
from pathlib import Path

logger = logging.get_logger(__name__)


def get_ava_mini_groundtruth(full_groundtruth):
    """
    Get the groundtruth annotations corresponding the "subset" of AVA val set.
    We define the subset to be the frames such that (second % 4 == 0).
    We optionally use subset for faster evaluation during training
    (in order to track training progress).
    Args:
        full_groundtruth(dict): list of groundtruth.
    """
    ret = [defaultdict(list), defaultdict(list), defaultdict(list)]

    for i in range(3):
        for key in full_groundtruth[i].keys():
            if int(key.split(",")[1]) % 4 == 0:
                ret[i][key] = full_groundtruth[i][key]
    return ret

class AVAMeter(object):
    """
    Measure the AVA train, val, and test stats.
    """

    def __init__(self, overall_iters, cfg, mode):
        """
            overall_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
            mode (str): `train`, `val`, or `test` mode.
        """
        self.cfg = cfg
        self.lr = None
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.full_ava_test = cfg.AVA.FULL_TEST_ON_VAL
        self.mode = mode
        self.iter_timer = Timer()
        self.all_preds = []
        self.all_ori_boxes = []
        self.all_metadata = []
        self.overall_iters = overall_iters
        self.excluded_keys = read_exclusions(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.EXCLUSION_FILE)
        )
        self.categories, self.class_whitelist = read_labelmap(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.LABEL_MAP_FILE)
        )
        gt_filename = os.path.join(
            cfg.AVA.ANNOTATION_DIR, cfg.AVA.GROUNDTRUTH_FILE
        )
        self.full_groundtruth = read_csv(gt_filename, self.class_whitelist)
        self.mini_groundtruth = get_ava_mini_groundtruth(self.full_groundtruth)

        _, self.video_idx_to_name = ava_helper.load_image_lists(
            cfg, mode == "train"
        )

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        Log the stats.
        Args:
            cur_epoch (int): the current epoch.
            cur_iter (int): the current iteration.
        """

        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return

        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        if self.mode == "train":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "time_diff": self.iter_timer.seconds(),
                "mode": self.mode,
                "loss": self.loss.get_win_median(),
                "lr": self.lr,
            }
        elif self.mode == "val":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "time_diff": self.iter_timer.seconds(),
                "mode": self.mode,
            }
        elif self.mode == "test":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "time_diff": self.iter_timer.seconds(),
                "mode": self.mode,
            }
        else:
            raise NotImplementedError("Unknown mode: {}".format(self.mode))

        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()

        self.all_preds = []
        self.all_ori_boxes = []
        self.all_metadata = []

    def update_stats(self, preds, ori_boxes, metadata, loss=None, lr=None):
        """
        Update the current stats.
        Args:
            preds (tensor): prediction embedding.
            ori_boxes (tensor): original boxes (x1, y1, x2, y2).
            metadata (tensor): metadata of the AVA data.
            loss (float): loss value.
            lr (float): learning rate.
        """
        if self.mode in ["val", "test"]:
            self.all_preds.append(preds)
            self.all_ori_boxes.append(ori_boxes)
            self.all_metadata.append(metadata)
        if loss is not None:
            self.loss.add_value(loss)
        if lr is not None:
            self.lr = lr

    def finalize_metrics(self, log=True):
        """
        Calculate and log the final AVA metrics.
        """
        all_preds = torch.cat(self.all_preds, dim=0)
        all_ori_boxes = torch.cat(self.all_ori_boxes, dim=0)
        all_metadata = torch.cat(self.all_metadata, dim=0)

        if self.mode == "test" or (self.full_ava_test and self.mode == "val"):
            groundtruth = self.full_groundtruth
        else:
            groundtruth = self.mini_groundtruth

        self.full_map = evaluate_ava(
            all_preds,
            all_ori_boxes,
            all_metadata.tolist(),
            self.excluded_keys,
            self.class_whitelist,
            self.categories,
            groundtruth=groundtruth,
            video_idx_to_name=self.video_idx_to_name,
        )
        if log:
            stats = {"mode": self.mode, "map": self.full_map}
            logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        if self.mode in ["val", "test"]:
            self.finalize_metrics(log=False)
            stats = {
                "_type": "{}_epoch".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "mode": self.mode,
                "map": self.full_map,
                "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
                "RAM": "{:.2f}/{:.2f} GB".format(*misc.cpu_mem_usage()),
            }
            logging.log_json_stats(stats)

class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        multi_label=False,
        ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """
        print("test numvideo:{} numclips:{}".format(num_videos, num_clips))
        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.multi_label = multi_label
        self.ensemble_method = ensemble_method
        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_cls))
        if multi_label:
            self.video_preds -= 1e10

        self.video_labels = (
            torch.zeros((num_videos, num_cls))
            if multi_label
            else torch.zeros((num_videos)).long()
        )
        self.clip_count = torch.zeros((num_videos)).long()
        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        if self.multi_label:
            self.video_preds -= 1e10
        self.video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                # print(self.video_labels[vid_id].type(torch.FloatTensor),
                #       labels[ind].type(torch.FloatTensor))
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )
            self.video_labels[vid_id] = labels[ind]
            if self.ensemble_method == "sum":
                self.video_preds[vid_id] += preds[ind]
            elif self.ensemble_method == "max":
                self.video_preds[vid_id] = torch.max(
                    self.video_preds[vid_id], preds[ind]
                )
            else:
                raise NotImplementedError(
                    "Ensemble Method {} is not supported".format(
                        self.ensemble_method
                    )
                )
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(i, k)
                            for i, k in enumerate(self.clip_count.tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        stats = {"split": "test_final"}
        if self.multi_label:
            map = get_map(
                self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy()
            )
            stats["map"] = map
        else:
            num_topks_correct = metrics.topks_correct(
                self.video_preds, self.video_labels, ks
            )
            topks = [
                (x / self.video_preds.size(0)) * 100.0
                for x in num_topks_correct
            ]
            assert len({len(ks), len(topks)}) == 1
            for k, topk in zip(ks, topks):
                stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
        logging.log_json_stats(stats)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        
        self.extra_stats = {}
        self.extra_stats_total = {}
        self.log_period = cfg.LOG_PERIOD

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

        for key in self.extra_stats.keys():
            self.extra_stats[key].reset()
            self.extra_stats_total[key] = 0.0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size, stats={}):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        if not self._cfg.DATA.MULTI_LABEL:
            # Current minibatch stats
            self.mb_top1_err.add_value(top1_err)
            self.mb_top5_err.add_value(top5_err)
            # Aggregate stats
            self.num_top1_mis += top1_err * mb_size
            self.num_top5_mis += top5_err * mb_size
        
        for key in stats.keys():
            if key not in self.extra_stats:
                self.extra_stats[key] = ScalarMeter(self.log_period)
                self.extra_stats_total[key] = 0.0
            self.extra_stats[key].add_value(stats[key])
            self.extra_stats_total[key] += stats[key] * mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats[key].get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f} GB".format(*misc.cpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            avg_loss = self.loss_total / self.num_samples
            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["loss"] = avg_loss
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples
        logging.log_json_stats(stats)


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        self.max_map = 0
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

        self.extra_stats = {}
        self.extra_stats_total = {}
        self.log_period = cfg.LOG_PERIOD

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

        for key in self.extra_stats.keys():
            self.extra_stats[key].reset()
            self.extra_stats_total[key] = 0.0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_err, top5_err, mb_size, stats={}):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size
    
        for key in stats.keys():
            if key not in self.extra_stats:
                self.extra_stats[key] = ScalarMeter(self.log_period)
                self.extra_stats_total[key] = 0.0
            self.extra_stats[key].add_value(stats[key])
            self.extra_stats_total[key] += stats[key] * mb_size

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats[key].get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f} GB".format(*misc.cpu_mem_usage()),
        }
        if self._cfg.DATA.MULTI_LABEL:
            stats["map"] = get_map(
                torch.cat(self.all_preds).cpu().numpy(),
                torch.cat(self.all_labels).cpu().numpy(),
            )
            is_best = stats["map"] > self.max_map
            self.max_map = max(self.max_map, stats["map"])
        else:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            is_best = top1_err < self.min_top1_err
            self.min_top1_err = min(self.min_top1_err, top1_err)
            self.min_top5_err = min(self.min_top5_err, top5_err)

            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["min_top1_err"] = self.min_top1_err
            stats["min_top5_err"] = self.min_top5_err
        
        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples

        logging.log_json_stats(stats)

        return is_best


class RecMeter_bih(object):
    def __init__(self, epoch_iters, cfg, mode):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): settings.
        """
        self._cfg = cfg
        self._mode = mode
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        # self.loss_con = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.avg_pre_f = ScalarMeter(cfg.LOG_PERIOD)
        self.avg_rec_f = ScalarMeter(cfg.LOG_PERIOD)
        self.avg_f1_f = ScalarMeter(cfg.LOG_PERIOD)
        self.avg_pre_t = ScalarMeter(cfg.LOG_PERIOD)
        self.avg_rec_t = ScalarMeter(cfg.LOG_PERIOD)
        self.avg_f1_t = ScalarMeter(cfg.LOG_PERIOD)

        self.num_samples = 0
        self.all_label_preds_f = []
        self.all_act_preds_f = []
        self.all_obj_preds_f = []
        # self.all_labels_f = []
        self.all_act_labels_f = []
        self.all_obj_labels_f = []
        self.all_label_preds_t = []
        self.all_act_preds_t = []
        self.all_obj_preds_t = []
        # self.all_labels_t = []
        self.all_act_labels_t = []
        self.all_obj_labels_t = []
        self.all_labels = []
        self.tp_f = 0
        self.fp_f = 0
        self.fn_f = 0
        self.tp_t = 0
        self.fp_t = 0
        self.fn_t = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        """Qitong on Nov. 18th, 2021."""
        # self.loss_con.reset()
        self.loss_total = 0.0
        self.lr = None
        self.avg_pre_f.reset()
        self.avg_rec_f.reset()
        self.avg_f1_f.reset()
        self.avg_pre_t.reset()
        self.avg_rec_t.reset()
        self.avg_f1_t.reset()

        self.num_samples = 0

        self.all_label_preds_f = []
        self.all_act_preds_f = []
        self.all_obj_preds_f = []
        # self.all_labels_f = []
        self.all_act_labels_f = []
        self.all_obj_labels_f = []
        self.all_label_preds_t = []
        self.all_act_preds_t = []
        self.all_obj_preds_t = []
        self.all_act_labels_t = []
        self.all_obj_labels_t = []
        self.all_labels = []

        self.tp_f = 0
        self.fp_f = 0
        self.fn_f = 0
        self.tp_t = 0
        self.fp_t = 0
        self.fn_t = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, tp_f, fp_f, fn_f, pred_labels_f, act_pred_labels_f, obj_pred_labels_f,
            gt_act_labels_f, gt_obj_labels_f,
            tp_t, fp_t, fn_t, pred_labels_t, act_pred_labels_t, obj_pred_labels_t,
            gt_act_labels_t, gt_obj_labels_t,
            labels, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        if tp_f == 0:
            avg_pre_f = 0
            avg_rec_f = 0
            avg_f1_f = 0
        else:
            avg_pre_f = tp_f / (tp_f + fp_f)
            avg_rec_f = tp_f / (tp_f + fn_f)
            avg_f1_f = (2.0 * tp_f) / (2.0 * tp_f + fn_f + fp_f)
        if tp_t == 0:
            avg_pre_t = 0
            avg_rec_t = 0
            avg_f1_t = 0
        else:
            avg_pre_t = tp_t / (tp_t + fp_t)
            avg_rec_t = tp_t / (tp_t + fn_t)
            avg_f1_t = (2.0 * tp_t) / (2.0 * tp_t + fn_t + fp_t)
        self.avg_pre_f.add_value(avg_pre_f)
        self.avg_rec_f.add_value(avg_rec_f)
        self.avg_f1_f.add_value(avg_f1_f)
        self.avg_pre_t.add_value(avg_pre_t)
        self.avg_rec_t.add_value(avg_rec_t)
        self.avg_f1_t.add_value(avg_f1_t)
        self.tp_f += tp_f
        self.fn_f += fn_f
        self.fp_f += fp_f
        self.tp_t += tp_t
        self.fn_t += fn_t
        self.fp_t += fp_t
        self.num_samples += mb_size

        self.all_label_preds_f.append(pred_labels_f)
        self.all_act_preds_f.append(act_pred_labels_f)
        self.all_obj_preds_f.append(obj_pred_labels_f)
        self.all_act_labels_f.append(gt_act_labels_f)
        self.all_obj_labels_f.append(gt_obj_labels_f)
        self.all_label_preds_t.append(pred_labels_t)
        self.all_act_preds_t.append(act_pred_labels_t)
        self.all_obj_preds_t.append(obj_pred_labels_t)
        self.all_act_labels_t.append(gt_act_labels_t)
        self.all_obj_labels_t.append(gt_obj_labels_t)

        self.all_labels.append(labels)

        if self._mode == 'train':
            self.loss.add_value(loss)
            self.loss_total += loss * mb_size
            self.lr = lr

    def update_stats_small(self, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """

        if self._mode == 'train':
            self.loss.add_value(loss)
            self.loss_total += loss * mb_size
            self.lr = lr

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
                self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "R": "[{}/{}|{}/{}]".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH, cur_iter + 1, self.epoch_iters),
            "eta": eta,
            "pre_f": self.avg_pre_f.get_win_median(),
            "rec_f": self.avg_rec_f.get_win_median(),
            "f1_f": self.avg_f1_f.get_win_median(),
            "pre_t": self.avg_pre_t.get_win_median(),
            "rec_t": self.avg_rec_t.get_win_median(),
            "f1_t": self.avg_f1_t.get_win_median(),
        }
        if self._mode == 'train':
            stats["id"] = "train_iter"
            stats["loss"] = self.loss.get_win_median()
            stats["lr"] = self.lr
        else:
            stats["id"] = "val_iter"
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
                self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        results = self.finalize_metrics()
        stats = {
            "R": "[{}/{}]".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "eta": eta,
        }
        if self._mode == 'train':
            stats["loss"] = self.loss_total / self.num_samples
            stats["lr"] = self.lr
            stats["id"] = "train"
        else:
            stats["id"] = "val"
        log_list = ['prec', 'rec', 'f1']
        for key in results.keys():
            # for log_idx, val in enumerate(log_list):
            stats['{}'.format(key)] = results[key][:-1]

        logging.log_json_stats(stats)
        save_path = Path(self._cfg.OUTPUT_DIR) / 'log'
        if not save_path.exists():
            save_path.mkdir(parents=True)
        with (save_path / '{}_{}.json'.format(self._mode, cur_epoch)).open('w') as f:
            json.dump(stats, f)

    def finalize_metrics(self):
        results = dict()
        all_preds_f = torch.cat(self.all_label_preds_f, dim=0).numpy()
        # all_labels_f = torch.cat(self.all_labels_f, dim=0).numpy()
        all_act_preds_f = torch.cat(self.all_act_preds_f, dim=0).numpy()
        all_act_labels_f = (torch.cat(self.all_act_labels_f, dim=0) + 1)
        all_preds_t = torch.cat(self.all_label_preds_t, dim=0).numpy()
        # all_labels_t = torch.cat(self.all_labels_t, dim=0).numpy()
        all_act_preds_t = torch.cat(self.all_act_preds_t, dim=0).numpy()
        all_act_labels_t = (torch.cat(self.all_act_labels_t, dim=0) + 1)
        all_labels = torch.cat(self.all_labels, dim=0).numpy()
        # Add one additional dimension for
        all_act_labels_f = (torch.sum(
            torch.zeros(all_act_labels_f.size(0), all_act_labels_f.size(1), (len(Metadata.action) + 1)).scatter_(-1,
                                                                                                             all_act_labels_f,
                                                                                                             1)[:, :,
            1:], dim=1) > 0).type(torch.float).numpy()
        all_act_labels_t = (torch.sum(
            torch.zeros(all_act_labels_t.size(0), all_act_labels_t.size(1), (len(Metadata.action) + 1)).scatter_(-1,
                                                                                                                 all_act_labels_t,
                                                                                                                 1)[:,
            :,
            1:], dim=1) > 0).type(torch.float).numpy()
        all_obj_preds_f = torch.cat(self.all_obj_preds_f, dim=0)
        all_obj_preds_f = all_obj_preds_f.view(-1, all_obj_preds_f.size(-1)).numpy()
        all_obj_labels_f = torch.cat(self.all_obj_labels_f, dim=0)
        all_obj_labels_f = all_obj_labels_f.view(-1, all_obj_labels_f.size(-1)).numpy()
        all_obj_preds_t = torch.cat(self.all_obj_preds_t, dim=0)
        all_obj_preds_t = all_obj_preds_t.view(-1, all_obj_preds_t.size(-1)).numpy()
        all_obj_labels_t = torch.cat(self.all_obj_labels_t, dim=0)
        all_obj_labels_t = all_obj_labels_t.view(-1, all_obj_labels_t.size(-1)).numpy()

        # binary cross entropy
        results['hois_fpv'] = sk_metrics.precision_recall_fscore_support(all_labels, all_preds_f,
                                                                     labels=list(range(len(Metadata.hoi))),
                                                                     average='micro')
        results['actions_fpv'] = sk_metrics.precision_recall_fscore_support(all_act_labels_f, all_act_preds_f,
                                                                        labels=list(range(len(Metadata.action))),
                                                                        average='micro')
        results['objects_fpv'] = sk_metrics.precision_recall_fscore_support(all_obj_labels_f, all_obj_preds_f,
                                                                        labels=list(range(len(Metadata.object))),
                                                                        average='micro')
        results['hois_tpv'] = sk_metrics.precision_recall_fscore_support(all_labels, all_preds_t,
                                                                         labels=list(range(len(Metadata.hoi))),
                                                                         average='micro')
        results['actions_tpv'] = sk_metrics.precision_recall_fscore_support(all_act_labels_t, all_act_preds_t,
                                                                            labels=list(range(len(Metadata.action))),
                                                                            average='micro')
        results['objects_tpv'] = sk_metrics.precision_recall_fscore_support(all_obj_labels_t, all_obj_preds_t,
                                                                            labels=list(range(len(Metadata.object))),
                                                                            average='micro')
        return results

class RecMeter_fpv(object):
    def __init__(self, epoch_iters, cfg, mode):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): settings.
        """
        self._cfg = cfg
        self._mode = mode
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        # self.loss_con = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.avg_pre_f = ScalarMeter(cfg.LOG_PERIOD)
        self.avg_rec_f = ScalarMeter(cfg.LOG_PERIOD)
        self.avg_f1_f = ScalarMeter(cfg.LOG_PERIOD)
        self.avg_pre_t = ScalarMeter(cfg.LOG_PERIOD)
        self.avg_rec_t = ScalarMeter(cfg.LOG_PERIOD)
        self.avg_f1_t = ScalarMeter(cfg.LOG_PERIOD)

        self.num_samples = 0
        self.all_label_preds_f = []
        self.all_act_preds_f = []
        self.all_obj_preds_f = []
        # self.all_labels_f = []
        self.all_act_labels_f = []
        self.all_obj_labels_f = []
        self.all_label_preds_t = []
        self.all_act_preds_t = []
        self.all_obj_preds_t = []
        # self.all_labels_t = []
        self.all_act_labels_t = []
        self.all_obj_labels_t = []
        self.all_labels = []
        self.tp_f = 0
        self.fp_f = 0
        self.fn_f = 0
        self.tp_t = 0
        self.fp_t = 0
        self.fn_t = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        """Qitong on Nov. 18th, 2021."""
        # self.loss_con.reset()
        self.loss_total = 0.0
        self.lr = None
        self.avg_pre_f.reset()
        self.avg_rec_f.reset()
        self.avg_f1_f.reset()

        self.num_samples = 0

        self.all_label_preds_f = []
        self.all_act_preds_f = []
        self.all_obj_preds_f = []
        # self.all_labels_f = []
        self.all_act_labels_f = []
        self.all_obj_labels_f = []
        self.all_labels = []

        self.tp_f = 0
        self.fp_f = 0
        self.fn_f = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, tp_f, fp_f, fn_f, pred_labels_f, act_pred_labels_f, obj_pred_labels_f,
            gt_act_labels_f, gt_obj_labels_f,
            labels, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        if tp_f == 0:
            avg_pre_f = 0
            avg_rec_f = 0
            avg_f1_f = 0
        else:
            avg_pre_f = tp_f / (tp_f + fp_f)
            avg_rec_f = tp_f / (tp_f + fn_f)
            avg_f1_f = (2.0 * tp_f) / (2.0 * tp_f + fn_f + fp_f)
        self.avg_pre_f.add_value(avg_pre_f)
        self.avg_rec_f.add_value(avg_rec_f)
        self.avg_f1_f.add_value(avg_f1_f)
        self.tp_f += tp_f
        self.fn_f += fn_f
        self.fp_f += fp_f
        self.num_samples += mb_size

        self.all_label_preds_f.append(pred_labels_f)
        self.all_act_preds_f.append(act_pred_labels_f)
        self.all_obj_preds_f.append(obj_pred_labels_f)
        self.all_act_labels_f.append(gt_act_labels_f)
        self.all_obj_labels_f.append(gt_obj_labels_f)

        self.all_labels.append(labels)

        if self._mode == 'train':
            self.loss.add_value(loss)
            self.loss_total += loss * mb_size
            self.lr = lr

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
                self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "R": "[{}/{}|{}/{}]".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH, cur_iter + 1, self.epoch_iters),
            "eta": eta,
            "pre_f": self.avg_pre_f.get_win_median(),
            "rec_f": self.avg_rec_f.get_win_median(),
            "f1_f": self.avg_f1_f.get_win_median(),
        }
        if self._mode == 'train':
            stats["id"] = "train_iter"
            stats["loss"] = self.loss.get_win_median()
            stats["lr"] = self.lr
        else:
            stats["id"] = "val_iter"
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
                self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        results = self.finalize_metrics()
        stats = {
            "R": "[{}/{}]".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "eta": eta,
        }
        if self._mode == 'train':
            stats["loss"] = self.loss_total / self.num_samples
            stats["lr"] = self.lr
            stats["id"] = "train"
        else:
            stats["id"] = "val"
        log_list = ['prec', 'rec', 'f1']
        for key in results.keys():
            # for log_idx, val in enumerate(log_list):
            stats['{}'.format(key)] = results[key][:-1]

        logging.log_json_stats(stats)
        save_path = Path(self._cfg.OUTPUT_DIR) / 'log'
        if not save_path.exists():
            save_path.mkdir(parents=True)
        with (save_path / '{}_{}.json'.format(self._mode, cur_epoch)).open('w') as f:
            json.dump(stats, f)

    def finalize_metrics(self):
        results = dict()
        all_preds_f = torch.cat(self.all_label_preds_f, dim=0).numpy()
        all_labels = torch.cat(self.all_labels, dim=0).squeeze().numpy()
        all_act_preds_f = torch.cat(self.all_act_preds_f, dim=0).numpy()
        all_act_labels_f = (torch.cat(self.all_act_labels_f, dim=0) + 1)
        # all_preds_t = torch.cat(self.all_label_preds_t, dim=0).numpy()
        print(all_preds_f.shape, all_labels.shape, all_act_preds_f.shape, all_act_labels_f.shape)

        # Add one additional dimension for
        # all_act_labels_f = (torch.sum(
        #     torch.zeros(all_act_labels_f.size(0), all_act_labels_f.size(1), (len(Metadata.action) + 1)).scatter_(-1,
        #                                                                                                      all_act_labels_f,
        #                                                                                                      1)[:, :,
        #     1:], dim=1) > 0).type(torch.float).numpy()
        all_obj_preds_f = torch.cat(self.all_obj_preds_f, dim=0)
        # all_obj_preds_f = all_obj_preds_f.view(-1, all_obj_preds_f.size(-1)).numpy()
        all_obj_labels_f = torch.cat(self.all_obj_labels_f, dim=0)
        # all_obj_labels_f = all_obj_labels_f.view(-1, all_obj_labels_f.size(-1)).numpy()

        # binary cross entropy
        results['hois_fpv'] = sk_metrics.precision_recall_fscore_support(all_labels, all_preds_f,
                                                                     labels=list(range(len(Metadata.hoi))),
                                                                     average='micro')
        # results['actions_fpv'] = sk_metrics.precision_recall_fscore_support(all_act_labels_f, all_act_preds_f,
        #                                                                 labels=list(range(len(Metadata.action))),
        #                                                                 average='micro')
        # results['objects_fpv'] = sk_metrics.precision_recall_fscore_support(all_obj_labels_f, all_obj_preds_f,
        #                                                                 labels=list(range(len(Metadata.object))),
        #                                                                 average='micro')
        return results

def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    logger.info("Getting mAP for {} examples".format(preds.shape[0]))

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap
