import logging
import pickle
import json

from pathlib import Path
import numpy as np

import torch.utils.data

import slowfast.datasets.transform as slowfast_transform
import slowfast.datasets.utils as slowfast_datautils
from slowfast.datasets.build import DATASET_REGISTRY

import slowfast.datasets.utils as data_utils
# from slowfast.datasets.metadata import Metadata

logger = logging.getLogger(__name__)

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
class Lemma_seg_tpv(torch.utils.data.Dataset):

    def __init__(self, cfg, mode, subclass='all'):
        assert mode in ['train', 'val', 'test', 'all'], 'Split [{}] not supported for Lemma_seg'.format(mode)
        self._mode = mode
        self._cfg = cfg
        self._video_meta = {}
        self._task = cfg.EXP.TASK
        # self._view = cfg.EXP.VIEW_TYPE
        self._img_type = cfg.EXP.IMG_TYPE
        self._label_type = cfg.EXP.LABEL_TYPE
        self._video_type = cfg.EXP.VIDEO_TYPE
        self._num_classes = cfg.MODEL.NUM_CLASSES

        # Segment sampling parameters
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._clip_length = cfg.DATA.NUM_FRAMES
        self._window_len = self._clip_length * self._sample_rate

        # Normalization parameters
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = False

        self._crop_size = cfg.DATA.CROP_SIZE
        if self._mode == 'train':
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
        else:
            self._test_force_flip = False

        logger.info('Constructing Lemma_seg_bi {}'.format(mode))
        self._load_data(verbose=False)

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
            split_file = Path("/usa/wqtwjt/LEMMA/LEMMA_anno/annotations") / 'splits' / '{}_{}.json'.format(self._cfg.EXP.VIDEO_TYPE,
                                                                                           self._mode)
            cache = False
            cache_path_fpv = Path("./save_path_lemma") / '{}_{}_{}_{}'.format(self._task, 'fpv',
                                                                                   self._img_type, self._video_type)
            cache_path_tpv = Path("./save_path_lemma") / '{}_{}_{}_{}'.format(self._task, 'tpv',
                                                                                   self._img_type, self._video_type)
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
            ) = data_utils.parse_gt_result(fpv_gt_path, self._cfg)
            tpv_gt_path = Path("/usa/wqtwjt/LEMMA/LEMMA_anno/annotations") / anno_rel_path / '{}_tpv_frames.p'.format(self._task)
            (
                self.tpv_frame_paths,
                self.tpv_gt_bboxes_labels,
                self.tpv_vid_idx_to_name,
                self.tpv_vid_name_to_idx
            ) = data_utils.parse_gt_result(tpv_gt_path, self._cfg)

            if cache and not cache_path_fpv.exists() and not cache_path_tpv.exists():
                raise NotImplementedError

        # Load embeddings
        with Path("/usa/wqtwjt/LEMMA/intermediate/embeddings/embedding.p").open('rb') as f:
            embeddings = pickle.load(f)
        self._action_embeddings = embeddings['action']
        self._task_embeddings = embeddings['task']

        self.fpv_selected_indices = data_utils.get_selected_indices(self.fpv_frame_paths, self.fpv_gt_bboxes_labels,
                                                                    self.fpv_vid_name_to_idx, self._cfg, self._mode,
                                                                    anno_rel_path, split_file=split_file)
        self.tpv_selected_indices = data_utils.get_selected_indices(self.tpv_frame_paths, self.tpv_gt_bboxes_labels,
                                                                    self.tpv_vid_name_to_idx, self._cfg, self._mode,
                                                                    anno_rel_path, split_file=split_file)
        self.print_summary()

    def print_summary(self):
        logger.info('=== MICASA Segment dataset summary ===')
        logger.info('Split: {}, View: tpv, Src: {}, Task: {}'.format(self._mode,
                                                                           self._img_type, self._label_type))
        logger.info('Number of tpv total videos: {}'.format(len(self.tpv_frame_paths)))
        tpv_total_frames = sum(
            len(tpv_frame_path) for _, tpv_frame_path in self.tpv_frame_paths.items()
        )
        logger.info('Number of total tpv frames: {}'.format(tpv_total_frames))
        logger.info('Number of selected tpv frames: {}'.format(len(self)))

        total_boxes = sum(
            len(self.tpv_gt_bboxes_labels[self.tpv_vid_idx_to_name[vid_idx]][center_idx])
            for vid_idx, center_idx, _ in self.tpv_selected_indices
        )
        logger.info('Number of used boxes in tpv: {}'.format(total_boxes))

    def __len__(self):
        return len(self.tpv_selected_indices)

    def __getitem__(self, index, verbose=False):
        if self._mode != 'train':
            tpv_vid_idx, tpv_center_idx, _ = self.tpv_selected_indices[index]
        else:
            tpv_vid_idx, tpv_center_idx, tpv_segment_length = self.tpv_selected_indices[index]
            start = data_utils.temporal_sample_offset(tpv_segment_length, self._window_len)
            tpv_center_idx = tpv_center_idx + start

        tpv_vid_name = self.tpv_vid_idx_to_name[tpv_vid_idx]

        tpv_seq = slowfast_datautils.get_sequence(
            tpv_center_idx,
            self._window_len // 2,
            self._sample_rate,
            num_frames=len(self.tpv_frame_paths[tpv_vid_name]),
        )
        # print(len(seq))

        tpv_image_paths = [[str(Path('/usa/wqtwjt/LEMMA/LEMMA_dataset') / x)
                            for x in self.tpv_frame_paths[tpv_vid_name][frame]] for frame in tpv_seq]
        tpv_imgs = data_utils.get_frames(tpv_image_paths, src=self._img_type)

        assert (tpv_vid_name in self.tpv_gt_bboxes_labels.keys()
                and tpv_center_idx in self.tpv_gt_bboxes_labels[tpv_vid_name].keys()), \
            '{} has no frame {}'.format(tpv_vid_name, tpv_center_idx)
        tpv_bboxes_labels = self.tpv_gt_bboxes_labels[tpv_vid_name][tpv_center_idx]
        # print(bboxes_labels)
        pos_max = 3
        act_max = 3

        label_dict = dict()
        label_dict['verb'] = torch.zeros([len(tpv_bboxes_labels), len(Metadata.action)])
        label_dict['noun'] = torch.zeros([len(tpv_bboxes_labels), len(Metadata.object)])
        label_dict['hoi'] = torch.zeros([len(tpv_bboxes_labels), len(Metadata.hoi)])
        action_labels = (torch.ones([len(tpv_bboxes_labels), act_max, 1]).type(torch.LongTensor) * -1.).type(torch.long)
        object_labels = torch.zeros([len(tpv_bboxes_labels), act_max, pos_max, len(Metadata.object)])
        task_labels = torch.zeros([len(tpv_bboxes_labels), act_max, len(Metadata.task)])
        num_actions = torch.zeros([len(tpv_bboxes_labels), 1])
        num_objs = torch.zeros([len(tpv_bboxes_labels), act_max, 1])
        bboxes = torch.zeros([len(tpv_bboxes_labels), 4])

        if self._mode == 'train':
            action_embeddings = torch.zeros([len(tpv_bboxes_labels), act_max, self._action_embeddings[0].shape[0]])
            task_embeddings = torch.zeros([len(tpv_bboxes_labels), act_max, self._task_embeddings[0].shape[0]])
        else:
            action_embeddings = torch.cat([
                torch.cat([
                    torch.tensor(self._action_embeddings).unsqueeze(0)
                    for _ in range(act_max)], dim=0).unsqueeze(0)
                for _ in range(len(tpv_bboxes_labels))], dim=0).type(torch.float32)
            task_embeddings = torch.cat([
                torch.cat([
                    torch.tensor(self._task_embeddings).unsqueeze(0)
                    for _ in range(act_max)], dim=0).unsqueeze(0)
                for _ in range(len(tpv_bboxes_labels))], dim=0).type(torch.float32)

        for bbox_idx, (bbox, labels, tasks, hois, pid) in enumerate(tpv_bboxes_labels):
            num_actions[bbox_idx] = len(labels)
            labels = sorted(labels, key=lambda x: x[0])
            for a_idx, label in enumerate(labels):
                action = label[0]
                action_labels[bbox_idx][a_idx] = int(action)
                label_dict['verb'][bbox_idx][action] = 1.0
                objs = label[1]
                num_objs[bbox_idx][a_idx] = len(objs)
                for pos_idx, obj_idx in enumerate(objs):
                    object_labels[bbox_idx][a_idx][pos_idx].scatter_(0, torch.tensor(obj_idx), 1.0)
                    label_dict['noun'][bbox_idx][obj_idx] = 1.0
                if self._mode == 'train':
                    action_embeddings[bbox_idx][a_idx] = torch.tensor(self._action_embeddings[action])
                for task_idx in tasks[a_idx]:
                    task_labels[bbox_idx][a_idx][task_idx] = 1.0
                if self._mode == 'train':
                    task_embeddings[bbox_idx][a_idx] = torch.mean(
                        torch.tensor([self._task_embeddings[tsk] for tsk in tasks[a_idx]]),
                        dim=0, keepdim=True)
                label_dict['hoi'][bbox_idx][hois[a_idx]] = 1.0
            bboxes[bbox_idx] = torch.tensor(bbox).type(torch.float32)
        labels = label_dict[self._cfg.EXP.LABEL_TYPE]
        bboxes = bboxes.numpy()
        ori_bboxes = bboxes.copy()
        metadata = [[tpv_vid_idx, tpv_center_idx, x[4]] for x in tpv_bboxes_labels]

        if verbose:
            data_utils.visualize_sequence(tpv_imgs[tpv_seq.index(tpv_center_idx)], bboxes, labels, self._cfg)

        # From T H W C -> T C H W
        tpv_imgs = tpv_imgs.permute(0, 3, 1, 2)
        tpv_imgs, tpv_bboxes = self._images_and_boxes_preprocessing(tpv_imgs, boxes=bboxes)
        # From T C H W -> C T H W
        tpv_imgs = tpv_imgs.permute(1, 0, 2, 3)
        tpv_imgs = slowfast_datautils.pack_pathway_output(self._cfg, tpv_imgs)

        meta = dict()
        meta['metadata'] = metadata
        meta['boxes'] = tpv_bboxes
        meta['ori_boxes'] = ori_bboxes

        meta['num_actions'] = num_actions
        meta['action_labels'] = action_labels
        meta['obj_labels'] = object_labels
        meta['num_objects'] = num_objs

        meta['verb_label'] = label_dict['verb']
        meta['noun_label'] = label_dict['noun']

        if self._cfg.EXP.SUPERVISION != 'none' or self._cfg.EXP.MODEL_TYPE == 'composed':
            meta['num_actions'] = num_actions
            if self._cfg.EXP.SUPERVISION != 'none':
                meta['task_labels'] = task_labels

        if self._cfg.EXP.MODEL_TYPE != 'plain':
            meta['task_embed'] = task_embeddings
            meta['action_embed'] = action_embeddings

        return tpv_imgs, labels, index, meta

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

        if self._mode == "train":
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
        elif self._mode == "val":
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = slowfast_transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            # Apply center crop for val split
            imgs, boxes = slowfast_transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = slowfast_transform.horizontal_flip(1, imgs, boxes=boxes)
        elif self._mode == "test" or self._mode == 'all':
            # Test split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = slowfast_transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            # Apply center crop for val split
            imgs, boxes = slowfast_transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = slowfast_transform.horizontal_flip(1, imgs, boxes=boxes)
        else:
            raise NotImplementedError(
                "{} split not supported yet!".format(self._split)
            )

        # Do color augmentation (after divided by 255.0).
        if self._mode == "train" and self._use_color_augmentation:
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
