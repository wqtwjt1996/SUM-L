#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
from itertools import chain as chain
import pandas as pd
import json
import pickle
import numpy as np
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import slowfast.utils.logging as logging
from pathlib import Path

from . import utils as utils
from .build import DATASET_REGISTRY
import slowfast.datasets.utils as data_utils
import slowfast.datasets.transform as slowfast_transform

logger = logging.get_logger(__name__)

class Metadata(object):
    action = json.load((Path('/usa/wqtwjt/LEMMA/LEMMA_anno/annotations/metadata') / 'all_acts.json').open('r'))
    action_index = {v : k for k, v in enumerate(action)}

    object = json.load((Path('/usa/wqtwjt/LEMMA/LEMMA_anno/annotations/metadata') / 'all_objs.json').open('r'))
    object_index = {v : k for k, v in enumerate(object)}

    task = json.load((Path('/usa/wqtwjt/LEMMA/LEMMA_anno/annotations/metadata') / 'all_tasks.json').open('r'))
    task_index = {v : k for k, v in enumerate(task)}

    hoi = json.load((Path('/usa/wqtwjt/LEMMA/LEMMA_anno/annotations/metadata') / 'all_hois.json').open('r'))
    hoi_index = {v : k for k, v in enumerate(hoi)}

@DATASET_REGISTRY.register()
class Epickitchen100_lemma_txt(torch.utils.data.Dataset):
    '''
    Support epic-55 and epic-100
    '''

    def __init__(self, cfg, mode, num_retries=10):
        """
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Epic ".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.is_epic100 = cfg.DATA.EPIC_100

        self._video_meta = {}
        self.lemma_dict_bool = False
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Epic {}...".format(mode))
        self._construct_loader()

        """lemma"""
        self._video_type = "1tpv"
        self._task = "rec"
        self._img_type = "rgb"
        self._label_type = "hoi"

        # Segment sampling parameters
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._clip_length = cfg.DATA.NUM_FRAMES
        self._window_len = self._clip_length * self._sample_rate

        # Normalization parameters
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = False

        # self._crop_size = cfg.DATA.CROP_SIZE
        self._crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
        self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
        self._use_color_augmentation = True
        self._pca_jitter_only = True
        self._pca_eigval = [0.225, 0.224, 0.229]
        self._pca_eigvec = [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ]
        self.verb_dict, self.noun_dict = data_utils.get_lemma_to_epic100_labels()
        # self.label_dict_action = data_utils.get_label_dict()

        print(len(self))
        logger.info('Constructing Lemma_seg_bi {}'.format(mode))
        self._load_data(verbose=False)
        if self.mode == "train" and self.lemma_dict_bool:
            with open("/data/remote/Deep_REAL_Datasets/LEMMA/LEMMA_anno/all_hois.json", 'r') as load_f:
                self.lemma_hoi_list = json.load(load_f)
            lemma_all_dict = {}
            for i in range(self._get_lemma_len()):
                fpv_image_paths, tpv_image_paths, verb_num, noun_num, hois = self._forward_lemma(i)
                lemma_all_dict[i] = [fpv_image_paths, tpv_image_paths, verb_num, noun_num, hois]
            ep_all_dict = {}
            for i in range(len(self)):
                # if i > 100:
                #     break
                pseudo_pair_same_list, pseudo_pair_subset_list, \
                pseudo_pair_verb_list, pseudo_pair_noun_list, \
                label_verb, label_noun = self._find_pseudo_pair(i, lemma_all_dict)
                if len(pseudo_pair_same_list) > 0:
                    print(i, label_verb, label_noun, pseudo_pair_same_list[0])
                else:
                    print(i, label_verb, label_noun, pseudo_pair_same_list)
                ep_all_dict[i] = {"same": pseudo_pair_same_list, "sub": pseudo_pair_subset_list,
                                  "sub_verb": pseudo_pair_verb_list, "sub_noun": pseudo_pair_noun_list}
            json_str = json.dumps(ep_all_dict, indent=4)
            with open('/usa/wqtwjt/Ego-Exo-2/ego_iccv_23_big_files/pesudo_ep100_data_1108_32.json', 'w') as json_file:
                json_file.write(json_str)
            print("Done!!")
            exit(32)
        else:
            with open("/usa/wqtwjt/Ego-Exo-2/ego_iccv_23_big_files/pesudo_ep100_data_1108_32.json", 'r') as load_f:
                self.load_dict = json.load(load_f)
            self.dict_txt = self._load_one_pair()
            logger.info('Loading pseudo pair dictionary finished! (1108 32)')

    def _load_one_pair(self):
        f = open("/usa/wqtwjt/Ego-Exo-2/0818_txt_pos_mining_2.txt", "r")
        dict_txt = {}
        for data in f.readlines():
            # print(data[:-1].split())
            if int(data[:-1].split(",")[1]) == 0:
                dict_txt[int(data[:-1].split(",")[0])] = [0, -1]
            else:
                dict_txt[int(data[:-1].split(",")[0])] = [int(data[:-1].split(",")[1]), int(data[:-1].split(",")[-3])]
        f.close()
        return dict_txt

    def _load_data(self, verbose=False):
        self.fpv_frame_paths = {}
        self.fpv_gt_bboxes_labels = {}
        self.fpv_bboxes_labels = {}
        self.fpv_gt_tasks = {}
        self.fpv_vid_idx_to_name = None
        self.fpv_selected_indices = list()

        self.tpv_frame_paths = {}
        self.tpv_gt_bboxes_labels = {}
        self.tpv_bboxes_labels = {}
        self.tpv_gt_tasks = {}
        self.tpv_vid_idx_to_name = None
        self.tpv_selected_indices = list()

        # Load Image, GT, Detections
        keys = ['frame_paths', 'gt_bboxes_labels', 'gt_tasks', 'bboxes_labels', 'vid_idx_to_name', 'vid_name_to_idx']
        if not verbose:
            anno_rel_path = 'full'
            split_file = Path("/usa/wqtwjt/LEMMA/LEMMA_anno/annotations") / 'splits' / '{}_{}.json'.format(
                self._video_type,
                self.mode)
            cache = False
            cache_path_fpv = Path("./save_path_lemma") / '{}_{}_{}_{}'.format(self._task, 'fpv',
                                                                                   "rgb", self._video_type)
            cache_path_tpv = Path("./save_path_lemma") / '{}_{}_{}_{}'.format(self._task, 'tpv',
                                                                                   "rgb", self._video_type)
        else:
            anno_rel_path = 'unit_test'
            split_file = None
            cache = False
            cache_path_fpv = None
            cache_path_tpv = None

        if cache and cache_path_fpv.exists() and cache_path_tpv.exists():
            raise NotImplementedError
        else:
            fpv_gt_path = Path("/usa/wqtwjt/LEMMA/LEMMA_anno/annotations") / anno_rel_path / '{}_fpv_frames.p'.format(self._task)
            (
                self.fpv_frame_paths,
                self.fpv_gt_bboxes_labels,
                self.fpv_vid_idx_to_name,
                self.fpv_vid_name_to_idx
            ) = data_utils.my_parse_gt_result(fpv_gt_path, "rgb")
            tpv_gt_path = Path("/usa/wqtwjt/LEMMA/LEMMA_anno/annotations") / anno_rel_path / '{}_tpv_frames.p'.format(self._task)
            (
                self.tpv_frame_paths,
                self.tpv_gt_bboxes_labels,
                self.tpv_vid_idx_to_name,
                self.tpv_vid_name_to_idx
            ) = data_utils.my_parse_gt_result(tpv_gt_path, "rgb")

            if cache and not cache_path_fpv.exists() and not cache_path_tpv.exists():
                raise NotImplementedError

        # Load embeddings
        with Path("/usa/wqtwjt/LEMMA/intermediate/embeddings/embedding.p").open('rb') as f:
            embeddings = pickle.load(f)
        self._action_embeddings = embeddings['action']
        self._task_embeddings = embeddings['task']

        self.fpv_selected_indices = data_utils.my_get_selected_indices(self.fpv_frame_paths, self.fpv_gt_bboxes_labels,
                                                                       self.fpv_vid_name_to_idx, self.mode,
                                                                       anno_rel_path, split_file=split_file,
                                                                       task=self._task)
        self.tpv_selected_indices = data_utils.my_get_selected_indices(self.tpv_frame_paths, self.tpv_gt_bboxes_labels,
                                                                       self.tpv_vid_name_to_idx, self.mode,
                                                                       anno_rel_path, split_file=split_file,
                                                                       task=self._task)
        self.print_summary()

    def print_summary(self):
        logger.info('=== MICASA Segment dataset summary ===')
        logger.info('Split: {}, View: fpv & tpv, Src: {}, Task: {}'.format(self.mode,
                                                                           self._img_type, self._label_type))
        logger.info('Number of fpv total videos: {}'.format(len(self.fpv_frame_paths)))
        logger.info('Number of tpv total videos: {}'.format(len(self.tpv_frame_paths)))
        fpv_total_frames = sum(
            len(fpv_frame_path) for _, fpv_frame_path in self.fpv_frame_paths.items()
        )
        logger.info('Number of total fpv frames: {}'.format(fpv_total_frames))
        logger.info('Number of selected fpv frames: {}'.format(self._get_lemma_len()))
        tpv_total_frames = sum(
            len(tpv_frame_path) for _, tpv_frame_path in self.tpv_frame_paths.items()
        )
        logger.info('Number of total tpv frames: {}'.format(tpv_total_frames))
        logger.info('Number of selected tpv frames: {}'.format(self._get_lemma_len()))

        total_boxes = sum(
            len(self.tpv_gt_bboxes_labels[self.tpv_vid_idx_to_name[vid_idx]][center_idx])
            for vid_idx, center_idx, _ in self.tpv_selected_indices
        )
        logger.info('Number of used boxes in tpv: {}'.format(total_boxes))


    def _clean_lemma_narration(self, narr):
        res = narr.replace("${s0}@", " ")
        res = res.replace("${m0}@", " ")
        res = res.replace(":{s1}@", " with ")
        res = res.replace(":{m1}@", " to the ")
        res = res.replace(":{s2}@", " with ")
        res = res.replace(":{m2}@", " with ")
        res = res.replace("|", " and ")
        return res

    def _forward_lemma(self, index):
        fpv_vid_idx, fpv_center_idx, _ = self.fpv_selected_indices[index]
        tpv_vid_idx, tpv_center_idx, _ = self.tpv_selected_indices[index]
        fpv_vid_name = self.fpv_vid_idx_to_name[fpv_vid_idx]
        tpv_vid_name = self.tpv_vid_idx_to_name[tpv_vid_idx]
        assert fpv_vid_name == tpv_vid_name
        fpv_seq = utils.get_sequence(
            fpv_center_idx,
            self._window_len // 2,
            self._sample_rate,
            num_frames=len(self.fpv_frame_paths[fpv_vid_name]),
        )
        tpv_seq = utils.get_sequence(
            tpv_center_idx,
            self._window_len // 2,
            self._sample_rate,
            num_frames=len(self.tpv_frame_paths[tpv_vid_name]),
        )

        verb_dict, noun_dict = utils.get_lemma_to_epic100_labels()
        # label_dict_action, len_dict = utils.get_label_dict()
        fpv_image_paths = [[str(Path('/usa/wqtwjt/LEMMA/LEMMA_dataset') / x)
                            for x in self.fpv_frame_paths[fpv_vid_name][frame]] for frame in fpv_seq]
        tpv_image_paths = [[str(Path('/usa/wqtwjt/LEMMA/LEMMA_dataset') / x)
                            for x in self.tpv_frame_paths[tpv_vid_name][frame]] for frame in tpv_seq]
        tpv_bboxes_labels = self.tpv_gt_bboxes_labels[tpv_vid_name][tpv_center_idx]
        label_dict = dict()
        label_dict['verb'] = torch.zeros([len(tpv_bboxes_labels), len(Metadata.action)])
        label_dict['noun'] = torch.zeros([len(tpv_bboxes_labels), len(Metadata.object)])
        hoiss = []
        for bbox_idx, (bbox, labels, tasks, hois, pid) in enumerate(tpv_bboxes_labels):
            labels = sorted(labels, key=lambda x: x[0])
            for a_idx, label in enumerate(labels):
                action = label[0]
                label_dict['verb'][bbox_idx][action] = 1.0
                objs = label[1]
                for pos_idx, obj_idx in enumerate(objs):
                    label_dict['noun'][bbox_idx][obj_idx] = 1.0
                # print(action, objs)
            hoiss.append(hois)
        verb_label_epic_num = torch.nonzero(label_dict['verb'].squeeze()).squeeze(dim=1).numpy()
        verb_num = []
        for ele in verb_label_epic_num:
            verb_num.append(verb_dict[ele])
        noun_label_epic_num = torch.nonzero(label_dict['noun'].squeeze()).squeeze(dim=1).numpy()
        noun_num = []
        for ele in noun_label_epic_num:
            noun_num.append(noun_dict[ele])
        return fpv_image_paths, tpv_image_paths, verb_num, noun_num, hoiss

    def _forward_lemma_small(self, index):
        fpv_vid_idx, fpv_center_idx, _ = self.fpv_selected_indices[index]
        tpv_vid_idx, tpv_center_idx, _ = self.tpv_selected_indices[index]
        fpv_vid_name = self.fpv_vid_idx_to_name[fpv_vid_idx]
        tpv_vid_name = self.tpv_vid_idx_to_name[tpv_vid_idx]
        assert fpv_vid_name == tpv_vid_name

        tpv_bboxes_labels = self.tpv_gt_bboxes_labels[tpv_vid_name][tpv_center_idx]
        label_dict = dict()
        label_dict['verb'] = torch.zeros([len(tpv_bboxes_labels), len(Metadata.action)])
        label_dict['noun'] = torch.zeros([len(tpv_bboxes_labels), len(Metadata.object)])
        for bbox_idx, (bbox, labels, tasks, hois, pid) in enumerate(tpv_bboxes_labels):
            labels = sorted(labels, key=lambda x: x[0])
            for a_idx, label in enumerate(labels):
                action = label[0]
                label_dict['verb'][bbox_idx][action] = 1.0
                objs = label[1]
                for pos_idx, obj_idx in enumerate(objs):
                    label_dict['noun'][bbox_idx][obj_idx] = 1.0
        verb_label_epic_num = torch.nonzero(label_dict['verb'].squeeze()).squeeze(dim=1).numpy()
        verb_num = []
        for ele in verb_label_epic_num:
            verb_num.append(ele)
        noun_label_epic_num = torch.nonzero(label_dict['noun'].squeeze()).squeeze(dim=1).numpy()
        noun_num = []
        for ele in noun_label_epic_num:
            noun_num.append(ele)
        return verb_num, noun_num

    def _find_pseudo_pair(self, index, lemma_all_dict):
        label_verb, label_noun = [int(self._videos[index]["verb_label"])], [int(self._videos[index]["noun_label"])]
        pseudo_pair_same_list = []
        pseudo_pair_subset_list = []
        pseudo_pair_verb_list = []
        pseudo_pair_noun_list = []
        max_num = 3
        for k, v in lemma_all_dict.items():
            if set(label_verb) == set(v[2]) and set(label_noun) == set(v[3]):
                if len(pseudo_pair_same_list) < max_num:
                    narr = self._clean_lemma_narration(self.lemma_hoi_list[v[4][0][0]])
                    pseudo_pair_same_list.append({k: [v[2], v[3], v[1], narr]})
            elif set(label_verb).issubset(set(v[2])) and set(label_noun).issubset(set(v[3])):
                if len(pseudo_pair_subset_list) < max_num:
                    narr = self._clean_lemma_narration(self.lemma_hoi_list[v[4][0][0]])
                    pseudo_pair_subset_list.append({k: [v[2], v[3], v[1], narr]})
            elif set(label_verb).issubset(set(v[2])) and not set(label_noun).issubset(set(v[3])):
                if len(pseudo_pair_verb_list) < max_num:
                    narr = self._clean_lemma_narration(self.lemma_hoi_list[v[4][0][0]])
                    pseudo_pair_verb_list.append({k: [v[2], v[3], v[1], narr]})
            elif not set(label_verb).issubset(set(v[2])) and set(label_noun).issubset(set(v[3])):
                if len(pseudo_pair_noun_list) < max_num:
                    narr = self._clean_lemma_narration(self.lemma_hoi_list[v[4][0][0]])
                    pseudo_pair_noun_list.append({k: [v[2], v[3], v[1], narr]})
        return pseudo_pair_same_list, pseudo_pair_subset_list, \
               pseudo_pair_verb_list, pseudo_pair_noun_list, label_verb, label_noun

    def load_annotations(self, path_file):
        data = pd.read_pickle(path_file)

        videos = []
        for tup in data.iterrows():
            series = tup[1]
            item = {
                "participant_id": series["participant_id"],
                "video_id": series["video_id"],
                "start_frame": series["start_frame"],
                "stop_frame": series['stop_frame'],
                "verb_label": series.get("verb_class", -1),
                "noun_label": series.get("noun_class", -1),
                "narration": series["narration"],
            }
            videos.append(item)

        return videos

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == "train":
            data_list = self.cfg.TRAIN.TRAIN_DATA_LIST
        elif self.mode == "val":
            data_list = self.cfg.TRAIN.VAL_DATA_LIST
        else:
            data_list = self.cfg.TEST.DATA_LIST

        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR,
            data_list,
        )
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._videos = self.load_annotations(path_to_file)

        self._videos = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._videos]
            )
        )
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [range(self._num_clips) for _ in range(len(self._videos))]
            )
        )

        logger.info(
            "Epic dataloader constructed (size: {}) from {}".format(
                len(self._videos), path_to_file
            )
        )

    def get_frame_path(self, frame, index):
        """Qitong on Dec. 11th, 2021."""
        if self.is_epic100:
            # return f"{self.cfg.DATA.PATH_PREFIX}/{self._videos[index]['participant_id']}" \
            #     f"/rgb_frames/{self._videos[index]['video_id']}/frame_{frame:010}.jpg"
            fst_dir = f"{self.cfg.DATA.PATH_PREFIX}/{self._videos[index]['participant_id']}" \
                      f"/{self._videos[index]['video_id']}/frame_{frame:010}.jpg"
            snd_dir = f"data/epic-55/train_rgb_frames/{self._videos[index]['participant_id']}" \
                      f"/{self._videos[index]['video_id']}/frame_{frame:010}.jpg"
            trd_dir = f"/data/local/EP55/test/{self._videos[index]['participant_id']}" \
                      f"/{self._videos[index]['video_id']}/frame_{frame:010}.jpg"
            if os.path.exists(fst_dir):
                return fst_dir
            elif os.path.exists(snd_dir):
                return snd_dir
            elif os.path.exists(trd_dir):
                return trd_dir
            else:
                raise FileNotFoundError("Imgs are unexisted! Please check your epic dataset configurations!!!")
        else:
            return f"{self.cfg.DATA.PATH_PREFIX}/{self._videos[index]['video_id'].split('_')[0]}" \
                   f"/{self._videos[index]['video_id']}" \
                   f"/frame_{frame:010}.jpg"

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["val"]:
            temporal_sample_index = int(self.cfg.TEST.NUM_ENSEMBLE_VIEWS / 2)
            spatial_sample_index = 1

            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3

        elif self.mode in ["test"]:
            temporal_sample_index = (
                    self._spatial_temporal_idx[index]
                    // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        """loading lemma pseudo pair"""
        meta = {}
        # print(list(self.load_dict[str(index)]["same"][0].values()))
        if len(self.load_dict[str(index)]["same"]) > 0:
            # ri = self.dict_txt[index][1]
            # assert self.dict_txt[index][0] == 2
            # ri = 0
            ri = random.randint(0, 2)
            verb_num = list(self.load_dict[str(index)]["same"][ri].values())[0][0]
            noun_num = list(self.load_dict[str(index)]["same"][ri].values())[0][1]
            meta["tpv_image_paths"] = list(self.load_dict[str(index)]["same"][ri].values())[0][2]
            meta["tpv_narration"] = list(self.load_dict[str(index)]["same"][ri].values())[0][3]
            meta["tpv_narration_all"] = ""
            # for ii in range(min(len(self.load_dict[str(index)]["same"]), 30)):
            #     meta["tpv_narration_all"] += list(self.load_dict[str(index)]["same"][ii].values())[0][3]
            #     meta["tpv_narration_all"] += "###"
            meta["same_idx"] = 2
            lemma_idx = self.load_dict[str(index)]["same"][ri].keys()
        elif len(self.load_dict[str(index)]["sub"]) > 0:
            # ri = self.dict_txt[index][1]
            # print(self.dict_txt[index][0])
            # ri = 0
            ri = random.randint(0, 2)
            # assert self.dict_txt[index][0] == 1
            verb_num = list(self.load_dict[str(index)]["sub"][ri].values())[0][0]
            noun_num = list(self.load_dict[str(index)]["sub"][ri].values())[0][1]
            meta["tpv_image_paths"] = list(self.load_dict[str(index)]["sub"][ri].values())[0][2]
            meta["tpv_narration"] = list(self.load_dict[str(index)]["sub"][ri].values())[0][3]
            meta["tpv_narration_all"] = ""
            # for ii in range(min(len(self.load_dict[str(index)]["sub"]), 30)):
            #     meta["tpv_narration_all"] += list(self.load_dict[str(index)]["sub"][ii].values())[0][3]
            #     meta["tpv_narration_all"] += "###"
            meta["same_idx"] = 1
            lemma_idx = self.load_dict[str(index)]["sub"][ri].keys()
        elif len(self.load_dict[str(index)]["sub_verb"]) > 0:
            # ri = self.dict_txt[index][1]
            # assert self.dict_txt[index][0] == 3
            # ri = 0
            ri = random.randint(0, 2)
            verb_num = list(self.load_dict[str(index)]["sub_verb"][ri].values())[0][0]
            noun_num = list(self.load_dict[str(index)]["sub_verb"][ri].values())[0][1]
            meta["tpv_image_paths"] = list(self.load_dict[str(index)]["sub_verb"][ri].values())[0][2]
            meta["tpv_narration"] = list(self.load_dict[str(index)]["sub_verb"][ri].values())[0][3]
            meta["tpv_narration_all"] = ""
            # for ii in range(min(len(self.load_dict[str(index)]["sub_verb"]), 30)):
            #     meta["tpv_narration_all"] += list(self.load_dict[str(index)]["sub_verb"][ii].values())[0][3]
            #     meta["tpv_narration_all"] += "###"
            meta["same_idx"] = 3
            lemma_idx = self.load_dict[str(index)]["sub_verb"][ri].keys()
        elif len(self.load_dict[str(index)]["sub_noun"]) > 0:
            # ri = self.dict_txt[index][1]
            # assert self.dict_txt[index][0] == 4
            # ri = 0
            ri = random.randint(0, 2)
            verb_num = list(self.load_dict[str(index)]["sub_noun"][ri].values())[0][0]
            noun_num = list(self.load_dict[str(index)]["sub_noun"][ri].values())[0][1]
            meta["tpv_image_paths"] = list(self.load_dict[str(index)]["sub_noun"][ri].values())[0][2]
            meta["tpv_narration"] = list(self.load_dict[str(index)]["sub_noun"][ri].values())[0][3]
            meta["tpv_narration_all"] = ""
            # for ii in range(min(len(self.load_dict[str(index)]["sub_noun"]), 30)):
            #     meta["tpv_narration_all"] += list(self.load_dict[str(index)]["sub_noun"][ii].values())[0][3]
            #     meta["tpv_narration_all"] += "###"
            meta["same_idx"] = 4
            lemma_idx = self.load_dict[str(index)]["sub_noun"][ri].keys()
        else:
            while True:
                ri_epic = random.randint(0, len(self) - 1)
                if len(self.load_dict[str(ri_epic)]["sub"]) > 0:
                    break
            ri = random.randint(0, len(self.load_dict[str(ri_epic)]["sub"]) - 1)
            # print(len(self.load_dict[str(ri_epic)]["sub"]), ri)
            verb_num = list(self.load_dict[str(ri_epic)]["sub"][ri].values())[0][0]
            noun_num = list(self.load_dict[str(ri_epic)]["sub"][ri].values())[0][1]
            meta["tpv_image_paths"] = list(self.load_dict[str(ri_epic)]["sub"][ri].values())[0][2]
            meta["tpv_narration"] = list(self.load_dict[str(ri_epic)]["sub"][ri].values())[0][3]
            meta["tpv_narration_all"] = ""
            meta["same_idx"] = 0
            lemma_idx = self.load_dict[str(ri_epic)]["sub"][ri].keys()
        # print(meta["tpv_image_paths"])
        verb_num_ori, noun_num_ori = self._forward_lemma_small(int(list(lemma_idx)[0]))
        tpv_imgs = data_utils.get_frames(meta["tpv_image_paths"], src=self._img_type)
        # From T H W C -> T C H W
        tpv_imgs = tpv_imgs.permute(0, 3, 1, 2)
        tpv_imgs, tpv_bboxes = self._images_and_boxes_preprocessing(tpv_imgs, boxes=None)
        # From T C H W -> C T H W
        tpv_imgs = tpv_imgs.permute(1, 0, 2, 3)
        tpv_imgs = utils.my_pack_pathway_output(self.cfg.MODEL.ARCH, self.cfg.DATA.REVERSE_INPUT_CHANNEL,
                                                self.cfg.MODEL.SINGLE_PATHWAY_ARCH,
                                                self.cfg.MODEL.MULTI_PATHWAY_ARCH,
                                                4, tpv_imgs)
        # print(tpv_imgs[0].size())
        label_verb_tensor = torch.zeros(self.cfg.MODEL.NUM_CLASSES_LIST[0])
        for ele in verb_num:
            label_verb_tensor[ele] = 1
        meta["verb_vec_ep"] = label_verb_tensor
        label_noun_tensor = torch.zeros(self.cfg.MODEL.NUM_CLASSES_LIST[1])
        for ele in noun_num:
            label_noun_tensor[ele] = 1
        meta["noun_vec_ep"] = label_noun_tensor

        label_verb_tensor = torch.zeros(25)
        for ele in verb_num_ori:
            label_verb_tensor[ele] = 1
        meta["verb_vec"] = label_verb_tensor
        label_noun_tensor = torch.zeros(64)
        for ele in noun_num_ori:
            label_noun_tensor[ele] = 1
        meta["noun_vec"] = label_noun_tensor

        meta["fpv_narration"] = self._videos[index]["narration"]

        num_frames = self.cfg.DATA.NUM_FRAMES
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        start_frame, end_frame = int(self._videos[index]["start_frame"]), int(self._videos[index]["stop_frame"])
        video_length = end_frame - start_frame + 1

        clip_length = (num_frames - 1) * sampling_rate + 1
        if temporal_sample_index == -1:
            if clip_length > video_length:
                start = random.randint(video_length - clip_length, 0)
            else:
                start = random.randint(0, video_length - clip_length)
        else:
            gap = float(max(video_length - clip_length, 0)) / (
                    self.cfg.TEST.NUM_ENSEMBLE_VIEWS - 1
            )
            start = int(round(gap * temporal_sample_index))

        seq = [
            max(min(start + i * sampling_rate, video_length - 1), 0)
            for i in range(num_frames)
        ]
        frames = torch.as_tensor(
            utils.retry_load_images(
                [self.get_frame_path(frame + start_frame, index) for frame in seq],
                self._num_retries,
            )
        )

        label = torch.tensor((int(self._videos[index]["verb_label"]), int(self._videos[index]["noun_label"])))

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
        )
        frames = utils.pack_pathway_output(self.cfg, frames)
        # return frames, label, index, {}
        return frames, tpv_imgs, label, index, meta, self.get_frame_path(seq[0] + start_frame, index)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._videos)

    def _get_lemma_len(self):
        assert len(self.fpv_selected_indices) == len(self.tpv_selected_indices)
        for idx in range(len(self.fpv_selected_indices)):
            assert self.fpv_selected_indices[idx] == self.tpv_selected_indices[idx]
        return len(self.tpv_selected_indices)

    def _images_and_boxes_preprocessing(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        height, width = imgs.shape[2], imgs.shape[3]
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, 1].
        if boxes is not None:
            boxes = slowfast_transform.clip_boxes_to_image(boxes, height, width)

        # Train split
        imgs, boxes = slowfast_transform.random_short_side_scale_jitter(
            imgs,
            min_size=self._jitter_min_scale,
            max_size=self._jitter_max_scale,
            boxes=boxes,
        )
        imgs, boxes = slowfast_transform.random_crop(
            imgs, self._crop_size, boxes=boxes
        )

        # Random flip.
        imgs, boxes = slowfast_transform.horizontal_flip(0.5, imgs, boxes=boxes)

        # Do color augmentation (after divided by 255.0).
        if self.mode == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = slowfast_transform.color_jitter(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = slowfast_transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = slowfast_transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        if boxes is not None:
            boxes = slowfast_transform.clip_boxes_to_image(
                boxes, self._crop_size, self._crop_size
            )

        return imgs, boxes
