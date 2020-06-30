# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from six.moves import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse
import json
# Must happen before importing COCO API (which imports matplotlib)
import utils.env as envu

envu.set_up_matplotlib()
from pycocotools import mask as COCOmask

import utils.boxes as box_utils
from core.config import cfg
from utils.timer import Timer
import utils.io_utils as io

logger = logging.getLogger(__name__)


class JsonDataset(object):
    """A class representing a COCO json dataset."""

    def __init__(self, name):
        logger.debug('Creating: {}'.format(name))
        self.name = name

        self.corr_mat = np.load("data/corre_hico.npy")

        if "train" in name:
            self.split = "train"
            self.image_directory = 'data/images/train2015'
        else:
            self.split = 'test'
            self.image_directory = 'data/images/test2015'

        self.debug_timer = Timer()

        self._init_keypoints()

        if self.split == 'train':
            self.image_data = io.load_json_object("data/Trainval_Faster_RCNN_R-50-PFN_2x_HICO_DET_pose.json")
        else:
            self.image_data = io.load_json_object("data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_pose.json")

        gt_anno = io.load_json_object("data/anno_list.json")
        self.gt_anno = {anno['global_id']: anno for anno in gt_anno}


    def _init_keypoints(self):
        keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear',
                     'right_ear', 'left_shoulder', 'right_shoulder',
                     'left_elbow', 'right_elbow', 'left_wrist',
                     'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                     'right_knee', 'left_ankle', 'right_ankle']

        self.keypoints_to_id_map = dict(
            zip(keypoints, range(len(keypoints))))
        self.keypoints = keypoints
        self.num_keypoints = len(keypoints)
        if cfg.KRCNN.NUM_KEYPOINTS != -1:
            assert cfg.KRCNN.NUM_KEYPOINTS == self.num_keypoints, \
                "number of keypoints should equal when using multiple datasets"
        else:
            cfg.KRCNN.NUM_KEYPOINTS = self.num_keypoints
        self.keypoint_flip_map = {
            'left_eye': 'right_eye',
            'left_ear': 'right_ear',
            'left_shoulder': 'right_shoulder',
            'left_elbow': 'right_elbow',
            'left_wrist': 'right_wrist',
            'left_hip': 'right_hip',
            'left_knee': 'right_knee',
            'left_ankle': 'right_ankle'}

    def get_roidb(
            self,
            gt=True,
            crowd_filter_thresh=0
    ):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'
        image_ids = list(self.image_data.keys())
        print(len(image_ids))
        image_ids.sort()

        if cfg.DEBUG:
            image_list = ["37"]
            # image_list = io.load_json_object('data/image_id_list_part.json')
            roidb = [{"image_id": image_id} for image_id in image_list]
            # print(len(roidb))
        else:
            roidb = [{"image_id": image_id} for image_id in image_ids]

        for entry in roidb:
            self._prep_roidb_entry(entry)

        for i, entry in enumerate(roidb):
            self._add_hico_precomp_bbox_keypoints(entry, self.image_data)

        for entry in roidb:
            self._add_gt_annotations(entry)

        print('origin_num: ', len(roidb))  # 9658
        roidb_ = []
        for entry in roidb:
            if len(entry['gt_human_boxes']) > 0:
                assert "precomp_boxes" in entry
                roidb_.append(entry)

        print('after_remove: ', len(roidb_))  # 9546

        return roidb_

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path
        img_id = "HICO_train2015_" + str(entry['image_id']).zfill(8) + ".jpg"
        # im_path = os.path.join(self.image_directory, img_id)
        # if self.split=='test':
        #     img_id = "HICO_test2015_" + str(entry['image_id']).zfill(8) + ".jpg"
        #     im_path = os.path.join(self.image_directory,img_id)
        im_path = os.path.join(self.image_directory,
                               f"HICO_{self.split}2015_" + str(entry['image_id']).zfill(8) + ".jpg")
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image'] = im_path
        entry['flipped'] = False

    def _add_hico_precomp_bbox_keypoints(self, entry, precomp_bbox_keypoints):
        im_id = str(entry['image_id'])
        value = np.array(precomp_bbox_keypoints[im_id], dtype=np.float32)

        entry['precomp_boxes'] = value[:, :4]
        entry['precomp_score'] = value[:, 4]
        entry['precomp_cate'] = value[:, 5]

        kp = value[:, 6:]
        x = kp[:, 0::3]  # 0-indexed x coordinates
        y = kp[:, 1::3]  # 0-indexed y coordinates
        # 0: not labeled; 1: labeled, not inside mask;
        # 2: labeled and inside mask
        v = kp[:, 2::3]
        num_keypoints = kp.shape[-1] / 3
        # print(num_keypoints, self.num_keypoints)
        assert num_keypoints == self.num_keypoints
        gt_kps = np.ones((kp.shape[0], 3, self.num_keypoints), dtype=np.float32)
        for i in range(self.num_keypoints):
            gt_kps[:, 0, i] = x[:, i]
            gt_kps[:, 1, i] = y[:, i]
            gt_kps[:, 2, i] = v[:, i]
        entry['precomp_keypoints'] = gt_kps

        prob_mask = np.empty([0, 117])
        for cat in entry['precomp_cate']:
            prob_mask = np.concatenate([prob_mask, self.corr_mat[int(cat) - 1][np.newaxis, :]], axis=0)
        entry['precomp_prob_mask'] = prob_mask

    def _add_gt_annotations(self, entry):
        image_id = entry['image_id']
        img = str(image_id).zfill(8)
        global_id = "HICO_" + self.split + '2015_' + img
        gt_anno = self.gt_anno[global_id]
        entry['width'] = gt_anno['image_size'][1]
        entry['height'] = gt_anno['image_size'][0]

        human_boxes = []
        object_boxes = []
        hoi_ids = []

        hois = gt_anno['hois']
        for hoi in hois:
            pair_indexs = hoi['connections']
            for index in pair_indexs:
                human_box = hoi['human_bboxes'][index[0]]
                obj_box = hoi['object_bboxes'][index[1]]
                hoi_id = hoi['id']

                save = False
                for i in range(len(human_boxes)):
                    if human_boxes[i] == human_box and object_boxes[i] == obj_box:
                        hoi_ids[i].append(hoi_id)
                        save = True
                        break
                if not save:
                    human_boxes.append(human_box)
                    object_boxes.append(obj_box)
                    hoi_ids.append([hoi_id])

        entry['gt_human_boxes'] = np.array(human_boxes)
        entry['gt_object_boxes'] = np.array(object_boxes)

        hoi_ids = np.array(hoi_ids)
        entry['gt_hoi_ids'] = hoi_ids

        gt_hoi_vec = np.zeros([len(hoi_ids), 600])
        gt_verb_vec = np.zeros([len(hoi_ids), 117])
        gt_verb_index = np.zeros([len(hoi_ids), 1], dtype=np.int32)
        gt_obj_cat = -np.ones([len(hoi_ids), 1])
        gt_cluster_index = np.zeros([len(hoi_ids), 1], dtype=np.int32)

        hoi_list = json.load(open("data/hoi_list.json", 'r'))
        hoi_idx_to_verb_idx = {int(item["id"]) - 1: int(item['verb_id']) - 1
                               for item in hoi_list}
        hoi_idx_to_obj_idx = {int(item["id"]) - 1: int(item['object_index'])
                              for item in hoi_list}
        hoi_idx_to_cluster_idx = {int(item["id"]) - 1: int(item['cluster_index'])
                                  for item in hoi_list}

        for i in range(len(hoi_ids)):
            hoi_id = hoi_ids[i]
            for index in hoi_id:
                gt_hoi_vec[i][int(index) - 1] = 1
                gt_verb_vec[i][hoi_idx_to_verb_idx[int(index) - 1]] = 1
                gt_verb_index[i] = hoi_idx_to_verb_idx[int(index) - 1]
                gt_cluster_index[i] = hoi_idx_to_cluster_idx[int(index) - 1]
                if gt_obj_cat[i] == -1:
                    gt_obj_cat[i] = hoi_idx_to_obj_idx[int(index) - 1] + 1
                else:
                    assert gt_obj_cat[i] == hoi_idx_to_obj_idx[int(index) - 1] + 1
        entry['gt_hoi_vec'] = gt_hoi_vec
        entry['gt_cluster_index'] = gt_cluster_index
        entry['gt_verb_vec'] = gt_verb_vec
        entry['gt_obj_cat'] = gt_obj_cat
        entry['gt_verb_index'] = gt_verb_index

    def _get_gt_keypoints(self, obj):
        """Return ground truth keypoints."""
        if 'keypoints' not in obj:
            return -np.ones((3, self.num_keypoints), dtype=np.float32)
            # return None
        kp = np.array(obj['keypoints'])
        x = kp[0::3]  # 0-indexed x coordinates
        y = kp[1::3]  # 0-indexed y coordinates
        # 0: not labeled; 1: labeled, not inside mask;
        # 2: labeled and inside mask
        v = kp[2::3]
        num_keypoints = len(obj['keypoints']) / 3
        assert num_keypoints == self.num_keypoints
        gt_kps = np.ones((3, self.num_keypoints), dtype=np.float32)
        for i in range(self.num_keypoints):
            gt_kps[0, i] = x[i]
            gt_kps[1, i] = y[i]
            gt_kps[2, i] = v[i]
        return gt_kps


def add_proposals(roidb, rois, im_info, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    scales = im_info[:, 2]
    if cfg.HICO.USE_PRECOMP_BOX:
        assert rois is None
        for i in range(len(roidb)):
            data_augmentation(roidb[i], im_info[i])
    else:
        box_list = []
        for i in range(len(roidb)):
            inv_im_scale = 1. / scales[i]
            idx = np.where(rois[:, 0] == i)[0]
            box_list.append(rois[idx, 1:] * inv_im_scale)

    if crowd_thresh > 0:
        _filter_crowd_proposals(roidb, crowd_thresh)


def data_augmentation(entry, im_info):
    # for i, entry in enumerate(roidb):
    h, w, r = im_info
    boxes = entry['precomp_boxes']
    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]
    x_center = (x0 + x1) / 2
    y_center = (y0 + y1) / 2
    width = x1 - x0
    height = y1 - y0

    ratio_x = np.random.uniform(0.8 * 1.1, 1.1 * 1.1, boxes.shape[0])  # range between 0.7 to 1.3
    ratio_y = np.random.uniform(0.8 * 1.1, 1.1 * 1.1, boxes.shape[0])  # range between 0.7 to 1.3

    # ratio_x = np.random.uniform(0.7*1.15, 1.3*1.15, boxes.shape[0]) # range between 0.7 to 1.3
    # ratio_y = np.random.uniform(0.7*1.15, 1.3*1.15, boxes.shape[0]) # range between 0.7 to 1.3

    offset_x = np.random.uniform(-0.05, 0.05, boxes.shape[0])  # range between -0.05 to 0.05
    offset_y = np.random.uniform(-0.05, 0.05, boxes.shape[0])  # range between -0.05 to 0.05

    x_center = x_center + offset_x * width
    y_center = y_center + offset_y * height
    x0_new = np.clip(x_center - width * ratio_x / 2., 0, w / r)
    x1_new = np.clip(x_center + width * ratio_x / 2., 0, w / r)
    y0_new = np.clip(y_center - height * ratio_y / 2., 0, h / r)
    y1_new = np.clip(y_center + height * ratio_y / 2., 0, h / r)

    entry['precomp_boxes'] = np.concatenate(([x0_new], [y0_new], [x1_new], [y1_new]), 0).T


def data_augmentation_gt(entry, im_info):
    h, w, r = im_info
    boxes = entry['boxes']
    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]
    x_center = (x0 + x1) / 2
    y_center = (y0 + y1) / 2
    width = x1 - x0
    height = y1 - y0

    ratio_x = np.random.uniform(0.7 * 1.15, 1.3 * 1.15, boxes.shape[0])  # range between 0.7 to 1.3
    ratio_y = np.random.uniform(0.7 * 1.15, 1.3 * 1.15, boxes.shape[0])  # range between 0.7 to 1.3
    # ratio_x = 1.
    # ratio_y = 1.

    offset_x = np.random.uniform(-0.1, 0.1, boxes.shape[0])  # range between -0.05 to 0.05
    offset_y = np.random.uniform(-0.1, 0.1, boxes.shape[0])  # range between -0.05 to 0.05

    # offset_x = 0.
    # offset_y = 0.

    x_center = x_center + offset_x * width
    y_center = y_center + offset_y * height
    x0_new = np.clip(x_center - width * ratio_x / 2., 0, w / r)
    x1_new = np.clip(x_center + width * ratio_x / 2., 0, w / r)
    y0_new = np.clip(y_center - height * ratio_y / 2., 0, h / r)
    y1_new = np.clip(y_center + height * ratio_y / 2., 0, h / r)

    entry['boxes_aug'] = np.concatenate(([x0_new], [y0_new], [x1_new], [y1_new]), 0).T


def _merge_compute_boxes_into_roidb(roidb):
    """Add proposal boxes to each roidb entry."""
    for i, entry in enumerate(roidb):
        precomp_boxes = entry['precomp_boxes']
        # print(len(precomp_boxes)) # 100

        gt_human_boxes = entry['gt_human_boxes']
        num_human = len(gt_human_boxes)

        gt_object_boxes = entry['gt_object_boxes']
        gt_boxes = np.concatenate([gt_human_boxes, gt_object_boxes])
        # print(gt_boxes.shape)

        proposal_to_gt_overlaps = box_utils.bbox_overlaps(
            precomp_boxes.astype(dtype=np.float32, copy=False),
            gt_boxes.astype(dtype=np.float32, copy=False)
        )
        # print(proposal_to_gt_overlaps.shape)  # (100, 32)

        maxes = proposal_to_gt_overlaps.max(axis=1)
        entry['max_overlaps'] = np.array(maxes)

        # argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
        # # print(entry['gt_obj_prob_mask'])
        # # print(len(entry['gt_obj_prob_mask'])) # 16
        # # print(argmaxes)
        # # print(len(argmaxes)) # 100
        #
        # pred_obj_prob_mask = np.zeros([len(precomp_boxes), 117])
        # all_gt_obj_prob_mask = np.concatenate(
        #     [entry['gt_human_mask'].repeat(num_human, 0), entry['gt_obj_prob_mask']])
        # # print(entry['gt_obj_prob_mask'].shape) # (16, 117)
        # # print(all_gt_obj_prob_mask.shape) # (32, 117)
        # pred_obj_prob_mask[np.arange(len(precomp_boxes))] = all_gt_obj_prob_mask[argmaxes]
        # entry['pred_obj_prob_mask'] = pred_obj_prob_mask
        # # print(np.where(pred_obj_prob_mask > 0))
        #
        # pred_obj_cate = np.zeros([len(precomp_boxes), 1])
        # all_gt_obj_cates = np.concatenate(
        #     [np.array([1]).repeat(num_human, 0)[:, np.newaxis], entry['gt_obj_cat']])
        # # print(all_gt_obj_cates.shape) # (32, 1)
        # # print(all_gt_obj_cates)
        # pred_obj_cate[np.arange(len(precomp_boxes))] = all_gt_obj_cates[argmaxes]
        # # print(pred_obj_cate)
        # entry['precomp_obj_cate'] = pred_obj_cate
        # entry['box_index_to_gt_index'] = argmaxes
        #


def _merge_proposal_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb) * 2
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        # print('len boxes:', len(boxes))
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            # import ipdb; ipdb.set_trace()
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]

        # entry['boxes'] = boxes.astype(entry['boxes'].dtype, copy=False)
        # entry['box_to_gt_ind_map'] = box_to_gt_ind_map.astype(entry['box_to_gt_ind_map'].dtype, copy=False)

        # gt_to_classes = -np.ones(len(entry['box_to_gt_ind_map']))
        # matched_ids = np.where(entry['box_to_gt_ind_map']>-1)[0]
        # gt_to_classes[matched_ids] = entry['gt_classes'][entry['box_to_gt_ind_map'][matched_ids]]
        # entry['gt_classes'] = gt_to_classes

        # entry['seg_areas'] = np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        # entry['gt_overlaps'] = gt_overlaps
        # entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])

        # is_to_crowd = np.ones(len(entry['box_to_gt_ind_map']))
        # is_to_crowd[matched_ids] = entry['is_crowd'][entry['box_to_gt_ind_map'][matched_ids]]
        # entry['is_crowd'] = is_to_crowd

        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )

        gt_to_classes = -np.ones(len(box_to_gt_ind_map))
        matched_ids = np.where(box_to_gt_ind_map > -1)[0]
        gt_to_classes[matched_ids] = entry['gt_classes'][box_to_gt_ind_map[matched_ids]]

        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            gt_to_classes
            # np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        )
        entry['seg_areas'] = np.append(
            entry['seg_areas'],
            np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])

        is_to_crowd = np.ones(len(box_to_gt_ind_map))
        is_to_crowd[matched_ids] = entry['is_crowd'][box_to_gt_ind_map[matched_ids]]

        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            is_to_crowd
            # np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
        )
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )


def _filter_crowd_proposals(roidb, crowd_thresh):
    """Finds proposals that are inside crowd regions and marks them as
    overlap = -1 with each ground-truth rois, which means they will be excluded
    from training.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(entry['is_crowd'] == 1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        crowd_boxes = box_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = box_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        iscrowd_flags = [int(True)] * len(crowd_inds)
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        gt_overlaps[non_gt_inds[bad_inds], :] = -1
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)


def _add_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def _sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]
