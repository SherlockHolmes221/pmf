import math

import numpy as np
import matplotlib

from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction

matplotlib.use('agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import nn as mynn
from core.config import cfg


class SH_VCM(nn.Module):

    def __init__(self, roi_xform_func, spatial_scale, box_head,
                 interaction_num_action_classes=117):
        super().__init__()

        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.crop_size = cfg.HICO.PART_CROP_SIZE
        self.box_head = box_head

        hidden_dim = 256
        hidden_dim_hou = 1024

        self.interaction_num_action_classes = interaction_num_action_classes
        self.human_fc1 = nn.Linear(hidden_dim_hou, 512)
        self.human_fc2 = nn.Linear(512, interaction_num_action_classes)

        self.object_fc1 = nn.Linear(hidden_dim_hou, 512)
        self.object_fc2 = nn.Linear(512, interaction_num_action_classes)

        self.union_fc1 = nn.Linear(hidden_dim_hou, 512)
        self.union_fc2 = nn.Linear(512, interaction_num_action_classes)

        #######################################################
        self.part_num = 17
        self.part_fc1 = nn.Linear((self.part_num + 1) * 258 * self.crop_size ** 2, 1024)
        self.part_fc2 = nn.Linear(1024, hidden_dim * 2)
        self.part_fc3 = nn.Linear(hidden_dim * 2, interaction_num_action_classes)

        self.mlp = nn.Sequential(
            nn.Linear(3 * 64 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, self.part_num)
        )
        self.pose_fc1 = nn.Linear(3 * cfg.KRCNN.HEATMAP_SIZE ** 2, 512)
        self.pose_fc2 = nn.Linear(512, hidden_dim)
        self.pose_fc3 = nn.Linear(hidden_dim, interaction_num_action_classes)

    def detectron_weight_mapping(self):
        # hc is human centric branch
        # io is interaction branch object part
        detectron_weight_mapping = {
            'human_fc1.weight': 'hc_fc1_w',
            'human_fc1.bias': 'hc_fc1_b',
            'human_fc2.weight': 'hc_fc2_w',
            'human_fc2.bias': 'hc_fc2_b',
            'human_action_score.weight': 'hc_score_w',
            'human_action_score.bias': 'hc_score_b',
            'interaction_fc1.weight': 'inter_fc1_w',
            'interaction_fc1.bias': 'inter_fc1_b',
        }
        return detectron_weight_mapping, []

    def forward(self, x, hoi_blob):
        device_id = x[0].get_device()

        # get inds from numpy
        interaction_human_inds = torch.from_numpy(
            hoi_blob['interaction_human_inds']).long().cuda(device_id)
        interaction_object_inds = torch.from_numpy(
            hoi_blob['interaction_object_inds']).long().cuda(device_id)

        # object_mask = torch.from_numpy(hoi_blob["object_mask"]).float().cuda(device_id)
        #########################################################

        x_human = self.box_head(
            x, hoi_blob,
            blob_rois='human_boxes',
        )
        x_object = self.box_head(
            x, hoi_blob,
            blob_rois='object_boxes'
        )
        x_union = self.box_head(
            x, hoi_blob,
            blob_rois='union_boxes',
        )
        x_union = x_union.view(x_union.size(0), -1)
        x_human = x_human.view(x_human.size(0), -1)
        x_object = x_object.view(x_object.size(0), -1)
        #########################################################
        x_human = F.relu(self.human_fc1(x_human), inplace=True)
        x_human = self.human_fc2(x_human)[interaction_human_inds]
        #########################################################
        x_object = F.relu(self.object_fc1(x_object), inplace=True)
        x_object = self.object_fc2(x_object)[interaction_object_inds]
        #########################################################
        x_union = F.relu(self.union_fc1(x_union), inplace=True)
        x_union = self.union_fc2(x_union)
        #########################################################

        kps_pred = hoi_blob['poseconfig']
        if isinstance(kps_pred, np.ndarray):
            kps_pred = torch.from_numpy(kps_pred).cuda(device_id)
        poseconfig = kps_pred.view(kps_pred.size(0), -1)
        x_pose_line = F.relu(self.pose_fc1(poseconfig), inplace=True)
        x_pose_line = F.relu(self.pose_fc2(x_pose_line), inplace=True)
        x_pose_line = self.pose_fc3(x_pose_line)
        #########################################################

        coord_x, coord_y = np.meshgrid(np.arange(x.shape[-1]), np.arange(x.shape[-2]))
        coords = np.stack((coord_x, coord_y), axis=0).astype(np.float32)
        coords = torch.from_numpy(coords).cuda(device_id)
        x_coords = coords.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)  # 1 x 2 x H x W
        x_new = torch.cat((x, x_coords), dim=1)
        x_object2 = self.roi_xform(
            x_new, hoi_blob,
            blob_rois='object_boxes',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=self.crop_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        part_boxes = torch.from_numpy(
            hoi_blob['part_boxes']).cuda(device_id)
        x_pose = self.crop_pose_map(x_new, part_boxes, hoi_blob['flag'], self.crop_size)
        x_pose = x_pose[interaction_human_inds]

        x_object2 = x_object2.unsqueeze(dim=1)  # N x 1 x 258 x 5 x 5
        x_object2 = x_object2[interaction_object_inds]
        center_xy = x_object2[:, :, -2:, 2:3, 2:3]  # N x 1 x 2 x 1 x 1
        x_pose[:, :, -2:] = x_pose[:, :, -2:] - center_xy  # N x 1 x 2 x 5 x 5
        x_object2[:, :, -2:] = x_object2[:, :, -2:] - center_xy  # N x 17 x 2 x 5 x 5
        x_pose = torch.cat((x_pose, x_object2), dim=1)  # N x 18 x 258 x 5 x 5
        semantic_atten = F.sigmoid(self.mlp(poseconfig))
        semantic_atten = semantic_atten.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # N x 17 x 1 x 1 x 1
        x_pose_new = torch.zeros(x_pose.shape).cuda(device_id)
        x_pose_new[:, :17] = x_pose[:, :17] * semantic_atten
        x_pose_new[:, 17] = x_pose[:, 17]
        ## fuse the pose attention information
        x_pose = x_pose_new.view(x_pose_new.shape[0], -1)
        x_pose = F.relu(self.part_fc1(x_pose), inplace=True)
        x_pose = F.relu(self.part_fc2(x_pose), inplace=True)
        x_pose = self.part_fc3(x_pose)
        #######################################################################################
        factor_scores = 0
        factor_scores += x_human
        factor_scores += x_object
        factor_scores += x_union
        factor_scores += x_pose
        factor_scores += x_pose_line
        verb_prob = nn.Sigmoid()(factor_scores)

        interaction_action_score = verb_prob

        hoi_blob['interaction_action_score'] = interaction_action_score  ### multi classification score
        hoi_blob['interaction_affinity_score'] = torch.zeros((factor_scores.shape[0], 1)).cuda(device_id)

        return hoi_blob

    def crop_pose_map(self, union_feats, part_boxes, flag, crop_size):
        triplets_num, part_num, _ = part_boxes.shape

        ret = torch.zeros((triplets_num, part_num, union_feats.shape[1], crop_size, crop_size)).cuda(
            union_feats.get_device())
        part_feats = RoIAlignFunction(crop_size, crop_size, self.spatial_scale,
                                      cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO)(
            union_feats, part_boxes.view(-1, part_boxes.shape[-1])).view(ret.shape)

        valid_n, valid_p = np.where(flag > 0)

        if len(valid_n) > 0:
            ret[valid_n, valid_p] = part_feats[valid_n, valid_p]
        return ret

    @staticmethod
    def loss(hoi_blob):
        interaction_action_score = hoi_blob['interaction_action_score']

        device_id = interaction_action_score.get_device()
        interaction_action_labels = torch.from_numpy(hoi_blob['interaction_action_labels']).float().cuda(device_id)

        criterion = nn.BCELoss()
        interaction_action_loss = criterion(interaction_action_score, interaction_action_labels)

        # get interaction branch predict action accuracy
        interaction_action_preds = \
            (interaction_action_score.sigmoid() > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        interaction_action_accuray_cls = interaction_action_preds.eq(interaction_action_labels).float().mean()

        #####################################################################################
        interaction_affinity_loss = torch.zeros(interaction_action_loss.shape).cuda(device_id)
        interaction_affinity_cls = torch.zeros(interaction_action_accuray_cls.shape).cuda(device_id)  # add

        return interaction_action_loss, interaction_affinity_loss, \
               interaction_action_accuray_cls, interaction_affinity_cls

