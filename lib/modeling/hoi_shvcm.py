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


class CSP_PAM_LPA_EMB_ATT_LOSS(nn.Module):

    def __init__(self, roi_xform_func, spatial_scale, box_head,
                 interaction_num_action_classes=187, share=True,
                 se=False, add_score=False, pam=True, lpa=True):
        super().__init__()
        self.share_att = share
        self.se = se
        self.add_score = add_score
        self.pam = pam
        self.lpa = lpa

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
        if lpa:
            add = 600
        else:
            add = 0
        self.pose_fc1 = nn.Linear(3 * cfg.KRCNN.HEATMAP_SIZE ** 2 + add, 512)
        self.pose_fc2 = nn.Linear(512, hidden_dim)
        self.pose_fc3 = nn.Linear(hidden_dim, interaction_num_action_classes)

        self.subject_embedding = WordVecProjectionNet(in_channel=600, out_channel_list=[1024, 1024])
        self.union_embedding = WordVecProjectionNet(in_channel=600, out_channel_list=[1024, 1024])
        self.object_embedding = WordVecProjectionNet(in_channel=600, out_channel_list=[1024, 1024])

        if self.se:
            self.subject_attention = MLP(in_channel=1024, out_channel_list=[512, 1024], activation_list=[True, False],
                                         bn_list=[False, False], drop_out_list=[False] * 2)
            self.object_attention = MLP(in_channel=1024, out_channel_list=[512, 1024], activation_list=[True, False],
                                        bn_list=[False, False], drop_out_list=[False] * 2)
            if not self.share_att:
                self.union_attention = MLP(in_channel=1024, out_channel_list=[512, 1024], activation_list=[True, False],
                                           bn_list=[False, False], drop_out_list=[False] * 2)
        else:
            self.subject_attention = MLP(in_channel=1024, out_channel_list=[1024, 1024], activation_list=[True, False],
                                         bn_list=[False, False], drop_out_list=[False] * 2)
            self.object_attention = MLP(in_channel=1024, out_channel_list=[1024, 1024], activation_list=[True, False],
                                        bn_list=[False, False], drop_out_list=[False] * 2)
            if not self.share_att:
                self.union_attention = MLP(in_channel=1024, out_channel_list=[1024, 1024],
                                           activation_list=[True, False],
                                           bn_list=[False, False], drop_out_list=[False] * 2)

        if self.pam:
            self.factor_attention = FactorAttention(600, 3, use_sigmoid_or_softmax='sigmoid')

        #########################################################

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

        human_scores = hoi_blob["human_scores"][interaction_human_inds]
        # if isinstance(human_scores, np.float32):
        #     human_scores = np.array([human_scores])
        # if len(human_scores.shape) == 1:
        #     human_scores = human_scores[:, np.newaxis]
        human_scores = torch.from_numpy(human_scores).cuda(device_id)

        object_scores = hoi_blob["object_scores"][interaction_object_inds]
        # if isinstance(object_scores, np.float32):
        #     object_scores = np.array([object_scores])
        # if len(object_scores.shape) == 1:
        #     object_scores = object_scores[:, np.newaxis]
        object_scores = torch.from_numpy(object_scores).cuda(device_id)

        object_mask = torch.from_numpy(hoi_blob["object_mask"]).float().cuda(device_id)
        verb_object_vec = torch.from_numpy(hoi_blob["verb_object_vec"]).float().cuda(device_id)
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
        x_human = x_human[interaction_human_inds]
        human_embedding = self.subject_embedding(verb_object_vec)
        human_embedding = x_human * human_embedding
        if self.add_score:
            human_embedding_score = nn.Sigmoid()(torch.sum(human_embedding, 1)) * torch.squeeze(human_scores)
        else:
            human_embedding_score = nn.Sigmoid()(torch.sum(human_embedding, 1))
        human_attention = self.subject_attention(human_embedding)
        human_attention = nn.Sigmoid()(human_attention)
        x_human = x_human * (1 + human_attention)

        x_human = F.relu(self.human_fc1(x_human), inplace=True)
        x_human = self.human_fc2(x_human)
        #########################################################

        x_object = x_object[interaction_object_inds]
        object_embedding = self.object_embedding(verb_object_vec)
        object_embedding = x_object * object_embedding
        if self.add_score:
            object_embedding_score = nn.Sigmoid()(torch.sum(object_embedding, 1)) * torch.squeeze(object_scores)
        else:
            object_embedding_score = nn.Sigmoid()(torch.sum(object_embedding, 1))

        object_attention = self.object_attention(object_embedding)
        object_attention = nn.Sigmoid()(object_attention)
        x_object = x_object * (1 + object_attention)

        x_object = F.relu(self.object_fc1(x_object), inplace=True)
        x_object = self.object_fc2(x_object)

        #########################################################
        union_embedding = self.union_embedding(verb_object_vec)
        union_embedding = x_union * union_embedding
        if self.add_score:
            union_embedding_score = nn.Sigmoid()(torch.sum(union_embedding, 1)) * torch.squeeze(
                human_scores) * torch.squeeze(object_scores)
        else:
            union_embedding_score = nn.Sigmoid()(torch.sum(union_embedding, 1))
        if self.share_att:
            union_attention = self.object_attention(union_embedding)
        else:
            union_attention = self.union_attention(union_embedding)
        union_attention = nn.Sigmoid()(union_attention)
        x_union = x_union * (1 + union_attention)

        x_union = F.relu(self.union_fc1(x_union), inplace=True)
        x_union = self.union_fc2(x_union)
        #########################################################

        kps_pred = hoi_blob['poseconfig']
        if isinstance(kps_pred, np.ndarray):
            kps_pred = torch.from_numpy(kps_pred).cuda(device_id)
        poseconfig = kps_pred.view(kps_pred.size(0), -1)
        if self.lpa:
            poseconfig_ = torch.cat((poseconfig, verb_object_vec), dim=1)
        else:
            poseconfig_ = poseconfig
        x_pose_line = F.relu(self.pose_fc1(poseconfig_), inplace=True)
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

        pam_socre = self.factor_attention(verb_object_vec)

        factor_scores = 0
        factor_scores += x_human * pam_socre[:, 0].view(pam_socre.size()[0], -1)
        factor_scores += x_object * pam_socre[:, 1].view(pam_socre.size()[0], -1)
        factor_scores += x_union
        factor_scores += x_pose
        factor_scores += x_pose_line * pam_socre[:, 2].view(pam_socre.size()[0], -1)
        verb_prob = nn.Sigmoid()(factor_scores)

        interaction_action_score = verb_prob * object_mask

        hoi_blob['interaction_action_score'] = interaction_action_score  ### multi classification score
        hoi_blob['interaction_affinity_score'] = torch.zeros((factor_scores.shape[0], 1)).cuda(device_id)

        hoi_blob['human_embedding_score'] = human_embedding_score
        hoi_blob['object_embedding_score'] = object_embedding_score
        hoi_blob['union_embedding_score'] = union_embedding_score

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
        interaction_action_labels = torch.from_numpy(hoi_blob['interaction_action_labels']).float().cuda(
            device_id)

        copy_num = hoi_blob['copy_num']
        origin_len = len(copy_num)
        assert origin_len > 0
        interaction_action_loss = nn.BCELoss(reduction='none')(interaction_action_score, interaction_action_labels)
        assert np.sum(copy_num) == interaction_action_loss.size()[0]
        interaction_action_loss = torch.sum(interaction_action_loss) / 117 / origin_len

        # get interaction branch predict action accuracy
        interaction_action_preds = \
            (interaction_action_score.sigmoid() > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        interaction_action_accuray_cls = interaction_action_preds.eq(interaction_action_labels).float().mean()

        #####################################################################################
        interaction_affinity_loss = torch.zeros(interaction_action_loss.shape).cuda(device_id)
        interaction_affinity_cls = torch.zeros(interaction_action_accuray_cls.shape).cuda(device_id)  # add

        human_embedding_score = hoi_blob['human_embedding_score']
        object_embedding_score = hoi_blob['object_embedding_score']
        union_embedding_score = hoi_blob['union_embedding_score']
        interaction_action_labels_1 = torch.from_numpy(hoi_blob['interaction_action_labels_1']).float().cuda(device_id)

        human_embedding_score_loss = nn.BCELoss(reduction='none')(human_embedding_score, interaction_action_labels_1)
        union_embedding_score_loss = nn.BCELoss(reduction='none')(union_embedding_score, interaction_action_labels_1)
        object_embedding_score_loss = nn.BCELoss(reduction='none')(object_embedding_score, interaction_action_labels_1)

        human_embedding_score_loss = torch.sum(human_embedding_score_loss) / 117 / origin_len
        union_embedding_score_loss = torch.sum(union_embedding_score_loss) / 117 / origin_len
        object_embedding_score_loss = torch.sum(object_embedding_score_loss) / 117 / origin_len

        human_embedding_preds = \
            (human_embedding_score > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        human_accuray_cls = human_embedding_preds.eq(interaction_action_labels_1).float().mean()
        object_embedding_preds = \
            (object_embedding_score > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        object_accuray_cls = object_embedding_preds.eq(interaction_action_labels_1).float().mean()
        union_embedding_preds = \
            (union_embedding_score > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        union_accuray_cls = union_embedding_preds.eq(interaction_action_labels_1).float().mean()

        return interaction_action_loss, interaction_affinity_loss, \
               interaction_action_accuray_cls, interaction_affinity_cls, \
               human_embedding_score_loss, union_embedding_score_loss, object_embedding_score_loss, \
               human_accuray_cls, object_accuray_cls, union_accuray_cls


class SH_PAM_LPA_EMB_ATT_LOSS(nn.Module):

    def __init__(self, roi_xform_func, spatial_scale, box_head,
                 interaction_num_action_classes=117, share=True,
                 se=False, add_score=True, pam=True, lpa=True):
        super().__init__()
        self.share_att = share
        self.se = se
        self.add_score = add_score
        self.pam = pam
        self.lpa = lpa

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
        if lpa:
            add = 600
        else:
            add = 0
        self.pose_fc1 = nn.Linear(3 * cfg.KRCNN.HEATMAP_SIZE ** 2 + add, 512)
        self.pose_fc2 = nn.Linear(512, hidden_dim)
        self.pose_fc3 = nn.Linear(hidden_dim, interaction_num_action_classes)

        self.subject_embedding = WordVecProjectionNet(in_channel=600, out_channel_list=[1024, 1024])
        self.union_embedding = WordVecProjectionNet(in_channel=600, out_channel_list=[1024, 1024])
        self.object_embedding = WordVecProjectionNet(in_channel=600, out_channel_list=[1024, 1024])

        if self.se:
            self.subject_attention = MLP(in_channel=1024, out_channel_list=[512, 1024], activation_list=[True, False],
                                         bn_list=[False, False], drop_out_list=[False] * 2)
            self.object_attention = MLP(in_channel=1024, out_channel_list=[512, 1024], activation_list=[True, False],
                                        bn_list=[False, False], drop_out_list=[False] * 2)
            if not self.share_att:
                self.union_attention = MLP(in_channel=1024, out_channel_list=[512, 1024], activation_list=[True, False],
                                           bn_list=[False, False], drop_out_list=[False] * 2)
        else:
            self.subject_attention = MLP(in_channel=1024, out_channel_list=[1024, 1024], activation_list=[True, False],
                                         bn_list=[False, False], drop_out_list=[False] * 2)
            self.object_attention = MLP(in_channel=1024, out_channel_list=[1024, 1024], activation_list=[True, False],
                                        bn_list=[False, False], drop_out_list=[False] * 2)
            if not self.share_att:
                self.union_attention = MLP(in_channel=1024, out_channel_list=[1024, 1024],
                                           activation_list=[True, False],
                                           bn_list=[False, False], drop_out_list=[False] * 2)

        if self.pam:
            self.factor_attention = FactorAttention(600, 3, use_sigmoid_or_softmax='sigmoid')

        #########################################################

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

        human_scores = hoi_blob["human_scores"][interaction_human_inds]
        if isinstance(human_scores, np.float32):
            human_scores = np.array([human_scores])
        if len(human_scores.shape) == 1:
            human_scores = human_scores[:, np.newaxis]
        human_scores = torch.from_numpy(human_scores).cuda(device_id)

        object_scores = hoi_blob["object_scores"][interaction_object_inds]
        if isinstance(object_scores, np.float32):
            object_scores = np.array([object_scores])
        if len(object_scores.shape) == 1:
            object_scores = object_scores[:, np.newaxis]
        object_scores = torch.from_numpy(object_scores).cuda(device_id)

        object_mask = torch.from_numpy(hoi_blob["object_mask"]).float().cuda(device_id)
        verb_object_vec = torch.from_numpy(hoi_blob["verb_object_vec"]).float().cuda(device_id)
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
        x_human = x_human[interaction_human_inds]
        human_embedding = self.subject_embedding(verb_object_vec)
        human_embedding = x_human * human_embedding
        if self.add_score:
            human_embedding_score = nn.Sigmoid()(torch.sum(human_embedding, 1)) * torch.squeeze(human_scores)
        else:
            human_embedding_score = nn.Sigmoid()(torch.sum(human_embedding, 1))
        human_attention = self.subject_attention(human_embedding)
        human_attention = nn.Sigmoid()(human_attention)
        x_human = x_human * (1 + human_attention)

        x_human = F.relu(self.human_fc1(x_human), inplace=True)
        x_human = self.human_fc2(x_human)
        #########################################################

        x_object = x_object[interaction_object_inds]
        object_embedding = self.object_embedding(verb_object_vec)
        object_embedding = x_object * object_embedding
        if self.add_score:
            object_embedding_score = nn.Sigmoid()(torch.sum(object_embedding, 1)) * torch.squeeze(object_scores)
        else:
            object_embedding_score = nn.Sigmoid()(torch.sum(object_embedding, 1))

        object_attention = self.object_attention(object_embedding)
        object_attention = nn.Sigmoid()(object_attention)
        x_object = x_object * (1 + object_attention)

        x_object = F.relu(self.object_fc1(x_object), inplace=True)
        x_object = self.object_fc2(x_object)

        #########################################################
        union_embedding = self.union_embedding(verb_object_vec)
        union_embedding = x_union * union_embedding
        if self.add_score:
            union_embedding_score = nn.Sigmoid()(torch.sum(union_embedding, 1)) * torch.squeeze(
                human_scores) * torch.squeeze(object_scores)
        else:
            union_embedding_score = nn.Sigmoid()(torch.sum(union_embedding, 1))
        if self.share_att:
            union_attention = self.object_attention(union_embedding)
        else:
            union_attention = self.union_attention(union_embedding)
        union_attention = nn.Sigmoid()(union_attention)
        x_union = x_union * (1 + union_attention)

        x_union = F.relu(self.union_fc1(x_union), inplace=True)
        x_union = self.union_fc2(x_union)
        #########################################################

        kps_pred = hoi_blob['poseconfig']
        if isinstance(kps_pred, np.ndarray):
            kps_pred = torch.from_numpy(kps_pred).cuda(device_id)
        poseconfig = kps_pred.view(kps_pred.size(0), -1)
        if self.lpa:
            poseconfig_ = torch.cat((poseconfig, verb_object_vec), dim=1)
        else:
            poseconfig_ = poseconfig
        x_pose_line = F.relu(self.pose_fc1(poseconfig_), inplace=True)
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
        pam_socre = self.factor_attention(verb_object_vec)
        factor_scores = 0
        factor_scores += x_human * pam_socre[:, 0].view(pam_socre.size()[0], -1)
        factor_scores += x_object * pam_socre[:, 1].view(pam_socre.size()[0], -1)
        factor_scores += x_union
        factor_scores += x_pose
        factor_scores += x_pose_line * pam_socre[:, 2].view(pam_socre.size()[0], -1)
        verb_prob = nn.Sigmoid()(factor_scores)
        interaction_action_score = verb_prob * object_mask

        hoi_blob['interaction_action_score'] = interaction_action_score  ### multi classification score
        hoi_blob['interaction_affinity_score'] = torch.zeros((factor_scores.shape[0], 1)).cuda(device_id)

        hoi_blob['human_embedding_score'] = human_embedding_score
        hoi_blob['object_embedding_score'] = object_embedding_score
        hoi_blob['union_embedding_score'] = union_embedding_score

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
        interaction_action_labels = torch.from_numpy(hoi_blob['interaction_action_labels']).float().cuda(
            device_id)
        # todo
        interaction_action_loss = nn.BCELoss()(interaction_action_score, interaction_action_labels)

        # get interaction branch predict action accuracy
        interaction_action_preds = \
            (interaction_action_score.sigmoid() > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        interaction_action_accuray_cls = interaction_action_preds.eq(interaction_action_labels).float().mean()

        #####################################################################################
        interaction_affinity_loss = torch.zeros(interaction_action_loss.shape).cuda(device_id)
        interaction_affinity_cls = torch.zeros(interaction_action_accuray_cls.shape).cuda(device_id)  # add

        human_embedding_score = hoi_blob['human_embedding_score']
        object_embedding_score = hoi_blob['object_embedding_score']
        union_embedding_score = hoi_blob['union_embedding_score']
        interaction_action_labels_1 = torch.from_numpy(hoi_blob['interaction_action_labels_1']).float().cuda(device_id)

        human_embedding_score_loss = nn.BCELoss()(human_embedding_score, interaction_action_labels_1) * 0.1
        union_embedding_score_loss = nn.BCELoss()(union_embedding_score, interaction_action_labels_1) * 0.1
        object_embedding_score_loss = nn.BCELoss()(object_embedding_score, interaction_action_labels_1) * 0.1

        human_embedding_preds = \
            (human_embedding_score > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        human_accuray_cls = human_embedding_preds.eq(interaction_action_labels_1).float().mean()
        object_embedding_preds = \
            (object_embedding_score > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        object_accuray_cls = object_embedding_preds.eq(interaction_action_labels_1).float().mean()
        union_embedding_preds = \
            (union_embedding_score > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        union_accuray_cls = union_embedding_preds.eq(interaction_action_labels_1).float().mean()

        return interaction_action_loss, interaction_affinity_loss, \
               interaction_action_accuray_cls, interaction_affinity_cls, \
               human_embedding_score_loss, union_embedding_score_loss, object_embedding_score_loss, \
               human_accuray_cls, object_accuray_cls, union_accuray_cls


class SH_PAM_LPA_EMB_ATT_LOSS_2(nn.Module):

    def __init__(self, roi_xform_func, spatial_scale, box_head,
                 interaction_num_action_classes=117, share=True,
                 se=False, add_score=True, pam=True, lpa=True):
        super().__init__()
        self.share_att = share
        self.se = se
        self.add_score = add_score
        self.pam = pam
        self.lpa = lpa

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
        if lpa:
            add = 600
        else:
            add = 0
        self.pose_fc1 = nn.Linear(3 * cfg.KRCNN.HEATMAP_SIZE ** 2 + add, 512)
        self.pose_fc2 = nn.Linear(512, hidden_dim)
        self.pose_fc3 = nn.Linear(hidden_dim, interaction_num_action_classes)

        self.subject_embedding = WordVecProjectionNet(in_channel=600, out_channel_list=[1024, 1024])
        self.union_embedding = WordVecProjectionNet(in_channel=600, out_channel_list=[1024, 1024])
        self.object_embedding = WordVecProjectionNet(in_channel=600, out_channel_list=[1024, 1024])

        if self.se:
            self.subject_attention = MLP(in_channel=1024, out_channel_list=[512, 1024], activation_list=[True, False],
                                         bn_list=[False, False], drop_out_list=[False] * 2)
            self.object_attention = MLP(in_channel=1024, out_channel_list=[512, 1024], activation_list=[True, False],
                                        bn_list=[False, False], drop_out_list=[False] * 2)
            if not self.share_att:
                self.union_attention = MLP(in_channel=1024, out_channel_list=[512, 1024], activation_list=[True, False],
                                           bn_list=[False, False], drop_out_list=[False] * 2)
        else:
            self.subject_attention = MLP(in_channel=1024, out_channel_list=[1024, 1024], activation_list=[True, False],
                                         bn_list=[False, False], drop_out_list=[False] * 2)
            self.object_attention = MLP(in_channel=1024, out_channel_list=[1024, 1024], activation_list=[True, False],
                                        bn_list=[False, False], drop_out_list=[False] * 2)
            if not self.share_att:
                self.union_attention = MLP(in_channel=1024, out_channel_list=[1024, 1024],
                                           activation_list=[True, False],
                                           bn_list=[False, False], drop_out_list=[False] * 2)

        if self.pam:
            self.factor_attention = FactorAttention(600, 3, use_sigmoid_or_softmax='sigmoid')

        #########################################################

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

        human_scores = hoi_blob["human_scores"][interaction_human_inds]
        # if isinstance(human_scores, np.float32):
        #     human_scores = np.array([human_scores])
        # if len(human_scores.shape) == 1:
        #     human_scores = human_scores[:, np.newaxis]
        human_scores = torch.from_numpy(human_scores).cuda(device_id)

        object_scores = hoi_blob["object_scores"][interaction_object_inds]
        # if isinstance(object_scores, np.float32):
        #     object_scores = np.array([object_scores])
        # if len(object_scores.shape) == 1:
        #     object_scores = object_scores[:, np.newaxis]
        object_scores = torch.from_numpy(object_scores).cuda(device_id)

        object_mask = torch.from_numpy(hoi_blob["object_mask"]).float().cuda(device_id)
        verb_object_vec = torch.from_numpy(hoi_blob["verb_object_vec"]).float().cuda(device_id)
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
        x_human = x_human[interaction_human_inds]
        human_embedding = self.subject_embedding(verb_object_vec)
        human_embedding = x_human * human_embedding
        if self.add_score:
            human_embedding_score = nn.Sigmoid()(torch.sum(human_embedding, 1)) * torch.squeeze(human_scores)
        else:
            human_embedding_score = nn.Sigmoid()(torch.sum(human_embedding, 1))
        human_attention = self.subject_attention(human_embedding)
        human_attention = nn.Sigmoid()(human_attention)
        x_human = x_human * (1 + human_attention)

        x_human = F.relu(self.human_fc1(x_human), inplace=True)
        x_human = self.human_fc2(x_human)
        #########################################################

        x_object = x_object[interaction_object_inds]
        object_embedding = self.object_embedding(verb_object_vec)
        object_embedding = x_object * object_embedding
        if self.add_score:
            object_embedding_score = nn.Sigmoid()(torch.sum(object_embedding, 1)) * torch.squeeze(object_scores)
        else:
            object_embedding_score = nn.Sigmoid()(torch.sum(object_embedding, 1))

        object_attention = self.object_attention(object_embedding)
        object_attention = nn.Sigmoid()(object_attention)
        x_object = x_object * (1 + object_attention)

        x_object = F.relu(self.object_fc1(x_object), inplace=True)
        x_object = self.object_fc2(x_object)

        #########################################################
        union_embedding = self.union_embedding(verb_object_vec)
        union_embedding = x_union * union_embedding
        if self.add_score:
            union_embedding_score = nn.Sigmoid()(torch.sum(union_embedding, 1)) * torch.squeeze(
                human_scores) * torch.squeeze(object_scores)
        else:
            union_embedding_score = nn.Sigmoid()(torch.sum(union_embedding, 1))
        if self.share_att:
            union_attention = self.object_attention(union_embedding)
        else:
            union_attention = self.union_attention(union_embedding)
        union_attention = nn.Sigmoid()(union_attention)
        x_union = x_union * (1 + union_attention)

        x_union = F.relu(self.union_fc1(x_union), inplace=True)
        x_union = self.union_fc2(x_union)
        #########################################################

        kps_pred = hoi_blob['poseconfig']
        if isinstance(kps_pred, np.ndarray):
            kps_pred = torch.from_numpy(kps_pred).cuda(device_id)
        poseconfig = kps_pred.view(kps_pred.size(0), -1)
        if self.lpa:
            poseconfig_ = torch.cat((poseconfig, verb_object_vec), dim=1)
        else:
            poseconfig_ = poseconfig
        x_pose_line = F.relu(self.pose_fc1(poseconfig_), inplace=True)
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

        pam_socre = self.factor_attention(verb_object_vec)

        factor_scores = 0
        factor_scores += x_human * pam_socre[:, 0].view(pam_socre.size()[0], -1)
        factor_scores += x_object * pam_socre[:, 1].view(pam_socre.size()[0], -1)
        factor_scores += x_union
        factor_scores += x_pose
        factor_scores += x_pose_line * pam_socre[:, 2].view(pam_socre.size()[0], -1)
        verb_prob = nn.Sigmoid()(factor_scores)

        interaction_action_score = verb_prob * object_mask

        hoi_blob['interaction_action_score'] = interaction_action_score  ### multi classification score
        hoi_blob['interaction_affinity_score'] = torch.zeros((factor_scores.shape[0], 1)).cuda(device_id)

        hoi_blob['human_embedding_score'] = human_embedding_score
        hoi_blob['object_embedding_score'] = object_embedding_score
        hoi_blob['union_embedding_score'] = union_embedding_score

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
        interaction_action_labels = torch.from_numpy(hoi_blob['interaction_action_labels']).float().cuda(
            device_id)

        copy_num = hoi_blob['copy_num']
        origin_len = len(copy_num)
        assert origin_len > 0
        interaction_action_loss = nn.BCELoss(reduction='none')(interaction_action_score, interaction_action_labels)
        assert np.sum(copy_num) == interaction_action_loss.size()[0]
        interaction_action_loss = torch.sum(interaction_action_loss) / 117 / origin_len

        # get interaction branch predict action accuracy
        interaction_action_preds = \
            (interaction_action_score.sigmoid() > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        interaction_action_accuray_cls = interaction_action_preds.eq(interaction_action_labels).float().mean()

        #####################################################################################
        interaction_affinity_loss = torch.zeros(interaction_action_loss.shape).cuda(device_id)
        interaction_affinity_cls = torch.zeros(interaction_action_accuray_cls.shape).cuda(device_id)  # add

        human_embedding_score = hoi_blob['human_embedding_score']
        object_embedding_score = hoi_blob['object_embedding_score']
        union_embedding_score = hoi_blob['union_embedding_score']
        interaction_action_labels_1 = torch.from_numpy(hoi_blob['interaction_action_labels_1']).float().cuda(device_id)

        human_embedding_score_loss = nn.BCELoss()(human_embedding_score, interaction_action_labels_1) * 0.1
        union_embedding_score_loss = nn.BCELoss()(union_embedding_score, interaction_action_labels_1) * 0.1
        object_embedding_score_loss = nn.BCELoss()(object_embedding_score, interaction_action_labels_1) * 0.1

        human_embedding_preds = \
            (human_embedding_score > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        human_accuray_cls = human_embedding_preds.eq(interaction_action_labels_1).float().mean()
        object_embedding_preds = \
            (object_embedding_score > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        object_accuray_cls = object_embedding_preds.eq(interaction_action_labels_1).float().mean()
        union_embedding_preds = \
            (union_embedding_score > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        union_accuray_cls = union_embedding_preds.eq(interaction_action_labels_1).float().mean()

        return interaction_action_loss, interaction_affinity_loss, \
               interaction_action_accuray_cls, interaction_affinity_cls, \
               human_embedding_score_loss, union_embedding_score_loss, object_embedding_score_loss, \
               human_accuray_cls, object_accuray_cls, union_accuray_cls


class SH_PAM_LPA_EMB_ATT(nn.Module):

    def __init__(self, roi_xform_func, spatial_scale, box_head,
                 interaction_num_action_classes=117, share=True,
                 se=False, add_score=True, pam=True, lpa=True):
        super().__init__()
        self.share_att = share
        self.se = se
        self.add_score = add_score
        self.pam = pam
        self.lpa = lpa

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
        if lpa:
            add = 600
        else:
            add = 0
        self.pose_fc1 = nn.Linear(3 * cfg.KRCNN.HEATMAP_SIZE ** 2 + add, 512)
        self.pose_fc2 = nn.Linear(512, hidden_dim)
        self.pose_fc3 = nn.Linear(hidden_dim, interaction_num_action_classes)

        self.subject_embedding = WordVecProjectionNet(in_channel=600, out_channel_list=[1024, 1024])
        self.union_embedding = WordVecProjectionNet(in_channel=600, out_channel_list=[1024, 1024])
        self.object_embedding = WordVecProjectionNet(in_channel=600, out_channel_list=[1024, 1024])

        if self.se:
            self.subject_attention = MLP(in_channel=1024, out_channel_list=[512, 1024], activation_list=[True, False],
                                         bn_list=[False, False], drop_out_list=[False] * 2)
            self.object_attention = MLP(in_channel=1024, out_channel_list=[512, 1024], activation_list=[True, False],
                                        bn_list=[False, False], drop_out_list=[False] * 2)
            if not self.share_att:
                self.union_attention = MLP(in_channel=1024, out_channel_list=[512, 1024], activation_list=[True, False],
                                           bn_list=[False, False], drop_out_list=[False] * 2)
        else:
            self.subject_attention = MLP(in_channel=1024, out_channel_list=[1024, 1024], activation_list=[True, False],
                                         bn_list=[False, False], drop_out_list=[False] * 2)
            self.object_attention = MLP(in_channel=1024, out_channel_list=[1024, 1024], activation_list=[True, False],
                                        bn_list=[False, False], drop_out_list=[False] * 2)
            if not self.share_att:
                self.union_attention = MLP(in_channel=1024, out_channel_list=[1024, 1024],
                                           activation_list=[True, False],
                                           bn_list=[False, False], drop_out_list=[False] * 2)

        if self.pam:
            self.factor_attention = FactorAttention(600, 3, use_sigmoid_or_softmax='sigmoid')

        #########################################################

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

        # human_scores = hoi_blob["human_scores"][interaction_human_inds]
        # if isinstance(human_scores, np.float32):
        #     human_scores = np.array([human_scores])
        # if len(human_scores.shape) == 1:
        #     human_scores = human_scores[:, np.newaxis]
        # human_scores = torch.from_numpy(human_scores).cuda(device_id)
        #
        # object_scores = hoi_blob["object_scores"][interaction_object_inds]
        # if isinstance(object_scores, np.float32):
        #     object_scores = np.array([object_scores])
        # if len(object_scores.shape) == 1:
        #     object_scores = object_scores[:, np.newaxis]
        # object_scores = torch.from_numpy(object_scores).cuda(device_id)

        object_mask = torch.from_numpy(hoi_blob["object_mask"]).float().cuda(device_id)
        verb_object_vec = torch.from_numpy(hoi_blob["verb_object_vec"]).float().cuda(device_id)
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
        x_human = x_human[interaction_human_inds]
        human_embedding = self.subject_embedding(verb_object_vec)
        human_embedding = x_human * human_embedding
        # if self.add_score:
        #     human_embedding_score = nn.Sigmoid()(torch.sum(human_embedding, 1)) * torch.squeeze(human_scores)
        # else:
        #     human_embedding_score = nn.Sigmoid()(torch.sum(human_embedding, 1))
        human_attention = self.subject_attention(human_embedding)
        human_attention = nn.Sigmoid()(human_attention)
        x_human = x_human * (1 + human_attention)

        x_human = F.relu(self.human_fc1(x_human), inplace=True)
        x_human = self.human_fc2(x_human)
        #########################################################

        x_object = x_object[interaction_object_inds]
        object_embedding = self.object_embedding(verb_object_vec)
        object_embedding = x_object * object_embedding
        # if self.add_score:
        #     object_embedding_score = nn.Sigmoid()(torch.sum(object_embedding, 1)) * torch.squeeze(human_scores)
        # else:
        #     object_embedding_score = nn.Sigmoid()(torch.sum(object_embedding, 1))

        object_attention = self.object_attention(object_embedding)
        object_attention = nn.Sigmoid()(object_attention)
        x_object = x_object * (1 + object_attention)

        x_object = F.relu(self.object_fc1(x_object), inplace=True)
        x_object = self.object_fc2(x_object)

        #########################################################
        union_embedding = self.union_embedding(verb_object_vec)
        union_embedding = x_union * union_embedding
        # if self.add_score:
        #     union_embedding_score = nn.Sigmoid()(torch.sum(union_embedding, 1)) * torch.squeeze(
        #         human_scores) * torch.squeeze(object_scores)
        # else:
        #     union_embedding_score = nn.Sigmoid()(torch.sum(union_embedding, 1))
        if self.share_att:
            union_attention = self.object_attention(union_embedding)
        else:
            union_attention = self.union_attention(union_embedding)
        union_attention = nn.Sigmoid()(union_attention)
        x_union = x_union * (1 + union_attention)

        x_union = F.relu(self.union_fc1(x_union), inplace=True)
        x_union = self.union_fc2(x_union)
        #########################################################

        kps_pred = hoi_blob['poseconfig']
        if isinstance(kps_pred, np.ndarray):
            kps_pred = torch.from_numpy(kps_pred).cuda(device_id)
        poseconfig = kps_pred.view(kps_pred.size(0), -1)
        if self.lpa:
            poseconfig_ = torch.cat((poseconfig, verb_object_vec), dim=1)
        else:
            poseconfig_ = poseconfig
        x_pose_line = F.relu(self.pose_fc1(poseconfig_), inplace=True)
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

        pam_socre = self.factor_attention(verb_object_vec)

        factor_scores = 0
        factor_scores += x_human * pam_socre[:, 0].view(pam_socre.size()[0], -1)
        factor_scores += x_object * pam_socre[:, 1].view(pam_socre.size()[0], -1)
        factor_scores += x_union
        factor_scores += x_pose
        factor_scores += x_pose_line * pam_socre[:, 2].view(pam_socre.size()[0], -1)
        verb_prob = nn.Sigmoid()(factor_scores)

        interaction_action_score = verb_prob * object_mask

        hoi_blob['interaction_action_score'] = interaction_action_score  ### multi classification score
        hoi_blob['interaction_affinity_score'] = torch.zeros((factor_scores.shape[0], 1)).cuda(device_id)

        hoi_blob['human_embedding_score'] = torch.zeros((factor_scores.shape[0], 1)).cuda(device_id)
        hoi_blob['object_embedding_score'] = torch.zeros((factor_scores.shape[0], 1)).cuda(device_id)
        hoi_blob['union_embedding_score'] = torch.zeros((factor_scores.shape[0], 1)).cuda(device_id)

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
        interaction_action_labels = torch.from_numpy(hoi_blob['interaction_action_labels']).float().cuda(
            device_id)

        copy_num = hoi_blob['copy_num']
        origin_len = len(copy_num)
        assert origin_len > 0
        interaction_action_loss = nn.BCELoss(reduction='none')(interaction_action_score, interaction_action_labels)
        assert np.sum(copy_num) == interaction_action_loss.size()[0]
        interaction_action_loss = torch.sum(interaction_action_loss) / 117 / origin_len

        # get interaction branch predict action accuracy
        interaction_action_preds = \
            (interaction_action_score.sigmoid() > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        interaction_action_accuray_cls = interaction_action_preds.eq(interaction_action_labels).float().mean()

        #####################################################################################
        interaction_affinity_loss = torch.zeros(interaction_action_loss.shape).cuda(device_id)
        interaction_affinity_cls = torch.zeros(interaction_action_accuray_cls.shape).cuda(device_id)  # add
        human_embedding_score_loss = torch.zeros(interaction_action_loss.shape).cuda(device_id)
        union_embedding_score_loss = torch.zeros(interaction_action_loss.shape).cuda(device_id)
        object_embedding_score_loss = torch.zeros(interaction_action_loss.shape).cuda(device_id)

        # human_embedding_score = hoi_blob['human_embedding_score']
        # object_embedding_score = hoi_blob['object_embedding_score']
        # union_embedding_score = hoi_blob['union_embedding_score']
        # interaction_action_labels_1 = torch.from_numpy(hoi_blob['interaction_action_labels_1']).float().cuda(device_id)
        #
        # human_embedding_score_loss = nn.BCELoss()(human_embedding_score, interaction_action_labels_1)
        # union_embedding_score_loss = nn.BCELoss()(union_embedding_score, interaction_action_labels_1)
        # object_embedding_score_loss = nn.BCELoss()(object_embedding_score, interaction_action_labels_1)

        return interaction_action_loss, interaction_affinity_loss, \
               interaction_action_accuray_cls, interaction_affinity_cls
        #  human_embedding_score_loss, union_embedding_score_loss, object_embedding_score_loss


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

        object_mask = torch.from_numpy(hoi_blob["object_mask"]).float().cuda(device_id)
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

        interaction_action_score = verb_prob * object_mask

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

        # criterion = nn.BCELoss()
        # interaction_action_loss = criterion(interaction_action_score, interaction_action_labels)

        copy_num = hoi_blob['copy_num']
        origin_len = len(copy_num)
        assert origin_len > 0
        interaction_action_loss = nn.BCELoss(reduction='none')(interaction_action_score, interaction_action_labels)
        assert np.sum(copy_num) == interaction_action_loss.size()[0]
        interaction_action_loss = torch.sum(interaction_action_loss) / 117 / origin_len
        # print(np.where(interaction_action_score.detach().cpu().numpy() > 0)[0])
        # print(np.where(interaction_action_labels.cpu().numpy() > 0)[0])
        # print(interaction_action_loss)
        # tensor(0.0655, device='cuda:0', grad_fn=<DivBackward0>)
        # tensor(0.0405, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0508, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0473, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0213, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0461, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0131, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0191, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0397, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0185, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0234, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0249, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0197, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0254, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0313, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0135, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0138, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0131, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0226, device='cuda:0', grad_fn= < DivBackward0 >)

        # get interaction branch predict action accuracy
        interaction_action_preds = \
            (interaction_action_score.sigmoid() > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        interaction_action_accuray_cls = interaction_action_preds.eq(interaction_action_labels).float().mean()

        #####################################################################################
        interaction_affinity_loss = torch.zeros(interaction_action_loss.shape).cuda(device_id)
        interaction_affinity_cls = torch.zeros(interaction_action_accuray_cls.shape).cuda(device_id)  # add

        return interaction_action_loss, interaction_affinity_loss, \
               interaction_action_accuray_cls, interaction_affinity_cls


class SH_VCM_PAM(nn.Module):

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

        self.factor_attention = FactorAttention(600, 3, use_sigmoid_or_softmax='sigmoid')

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
        object_mask = torch.from_numpy(hoi_blob["object_mask"]).float().cuda(device_id)
        verb_object_vec = torch.from_numpy(hoi_blob["verb_object_vec"]).float().cuda(device_id)
        pam_socre = self.factor_attention(verb_object_vec)
        print(pam_socre[0])
        print(pam_socre[-1])

        factor_scores = 0
        factor_scores += x_human * pam_socre[:, 0].view(pam_socre.size()[0], -1)
        factor_scores += x_object * pam_socre[:, 1].view(pam_socre.size()[0], -1)
        factor_scores += x_union
        factor_scores += x_pose
        factor_scores += x_pose_line * pam_socre[:, 2].view(pam_socre.size()[0], -1)
        verb_prob = nn.Sigmoid()(factor_scores)

        interaction_action_score = verb_prob * object_mask

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
        interaction_action_labels = torch.from_numpy(hoi_blob['interaction_action_labels']).float().cuda(
            device_id)

        copy_num = hoi_blob['copy_num']
        origin_len = len(copy_num)
        assert origin_len > 0
        interaction_action_loss = nn.BCELoss(reduction='none')(interaction_action_score, interaction_action_labels)
        assert np.sum(copy_num) == interaction_action_loss.size()[0]
        interaction_action_loss = torch.sum(interaction_action_loss) / 117 / origin_len

        # get interaction branch predict action accuracy
        interaction_action_preds = \
            (interaction_action_score.sigmoid() > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        interaction_action_accuray_cls = interaction_action_preds.eq(interaction_action_labels).float().mean()

        #####################################################################################
        interaction_affinity_loss = torch.zeros(interaction_action_loss.shape).cuda(device_id)
        interaction_affinity_cls = torch.zeros(interaction_action_accuray_cls.shape).cuda(device_id)  # add

        return interaction_action_loss, interaction_affinity_loss, \
               interaction_action_accuray_cls, interaction_affinity_cls


class CSP_VCM(nn.Module):

    def __init__(self, roi_xform_func, spatial_scale, box_head,
                 interaction_num_action_classes=187):
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

        object_mask = torch.from_numpy(hoi_blob["object_mask"]).float().cuda(device_id)
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

        interaction_action_score = verb_prob * object_mask

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
        interaction_action_labels = torch.from_numpy(hoi_blob['interaction_action_labels']).float().cuda(
            device_id)

        # criterion = nn.BCELoss()
        # interaction_action_loss = criterion(interaction_action_score, interaction_action_labels)

        copy_num = hoi_blob['copy_num']
        origin_len = len(copy_num)
        assert origin_len > 0
        interaction_action_loss = nn.BCELoss(reduction='none')(interaction_action_score, interaction_action_labels)

        assert np.sum(copy_num) == interaction_action_loss.size()[0]
        interaction_action_loss = torch.sum(interaction_action_loss) / 117 / origin_len

        # print(np.where(interaction_action_score.detach().cpu().numpy() > 0)[0])
        # print(np.where(interaction_action_labels.cpu().numpy() > 0)[0])
        # [9  10  11  24  29  37  38  39  43  51  52  53  57  65  66  67  71  79
        #  80  81  95 108 300 322 336 350 356 364 365 366]
        # print(interaction_action_loss)
        # tensor(0.0426, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0557, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0526, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0286, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0497, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0193, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0243, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0410, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0252, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0311, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0284, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0193, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0330, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0408, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0247, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0094, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0148, device='cuda:0', grad_fn= < DivBackward0 >)
        # tensor(0.0226, device='cuda:0', grad_fn= < DivBackward0 >)

        # tensor(0.0647, device='cuda:0', grad_fn=<DivBackward0>)

        # get interaction branch predict action accuracy
        interaction_action_preds = \
            (interaction_action_score.sigmoid() > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        interaction_action_accuray_cls = interaction_action_preds.eq(interaction_action_labels).float().mean()

        #####################################################################################
        interaction_affinity_loss = torch.zeros(interaction_action_loss.shape).cuda(device_id)
        interaction_affinity_cls = torch.zeros(interaction_action_accuray_cls.shape).cuda(device_id)  # add

        return interaction_action_loss, interaction_affinity_loss, \
               interaction_action_accuray_cls, interaction_affinity_cls


class CSP_VCM_PAM(nn.Module):

    def __init__(self, roi_xform_func, spatial_scale, box_head,
                 interaction_num_action_classes=187):
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

        self.factor_attention = FactorAttention(600, 3, use_sigmoid_or_softmax='sigmoid')

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
        object_mask = torch.from_numpy(hoi_blob["object_mask"]).float().cuda(device_id)
        verb_object_vec = torch.from_numpy(hoi_blob["verb_object_vec"]).float().cuda(device_id)

        pam_socre = self.factor_attention(verb_object_vec)

        factor_scores = 0
        factor_scores += x_human * pam_socre[:, 0].view(pam_socre.size()[0], -1)
        factor_scores += x_object * pam_socre[:, 1].view(pam_socre.size()[0], -1)
        factor_scores += x_union
        factor_scores += x_pose
        factor_scores += x_pose_line * pam_socre[:, 2].view(pam_socre.size()[0], -1)
        verb_prob = nn.Sigmoid()(factor_scores)

        interaction_action_score = verb_prob * object_mask

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
        interaction_action_labels = torch.from_numpy(hoi_blob['interaction_action_labels']).float().cuda(
            device_id)

        # criterion = nn.BCELoss()
        # interaction_action_loss = criterion(interaction_action_score, interaction_action_labels)

        copy_num = hoi_blob['copy_num']
        origin_len = len(copy_num)
        assert origin_len > 0
        interaction_action_loss = nn.BCELoss(reduction='none')(interaction_action_score, interaction_action_labels)
        assert np.sum(copy_num) == interaction_action_loss.size()[0]
        interaction_action_loss = torch.sum(interaction_action_loss) / 187 / origin_len

        # get interaction branch predict action accuracy
        interaction_action_preds = \
            (interaction_action_score.sigmoid() > cfg.HICO.ACTION_THRESH).type_as(interaction_action_labels)
        interaction_action_accuray_cls = interaction_action_preds.eq(interaction_action_labels).float().mean()

        #####################################################################################
        interaction_affinity_loss = torch.zeros(interaction_action_loss.shape).cuda(device_id)
        interaction_affinity_cls = torch.zeros(interaction_action_accuray_cls.shape).cuda(device_id)  # add

        return interaction_action_loss, interaction_affinity_loss, \
               interaction_action_accuray_cls, interaction_affinity_cls


class fc_block_layer(nn.Module):
    def __init__(self, in_channel, out_channel, activation=True, bn=False, drop_out=False):
        super(fc_block_layer, self).__init__()

        self.layers = nn.ModuleList()
        fc = nn.Linear(in_channel, out_channel)
        self.layers.append(fc)
        if bn:
            self.layers.append(nn.BatchNorm1d(out_channel))
        if activation:
            self.layers.append(nn.ReLU())
        if drop_out:
            self.layers.append(nn.Dropout())

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel_list, activation_list, bn_list, drop_out_list):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i, out_channel in enumerate(out_channel_list):
            activation = activation_list[i]
            bn = bn_list[i]
            drop_out = drop_out_list[i]
            fc_block = fc_block_layer(in_channel, out_channel, activation, bn, drop_out)
            self.layers.append(fc_block)
            in_channel = out_channel

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class WordVecProjectionNet(nn.Module):

    def __init__(self, in_channel, out_channel_list, activation_list=[True, False], bn_list=[False, False],
                 drop_out_list=[False] * 2, normal=True):
        super(WordVecProjectionNet, self).__init__()
        self.normal = normal
        self.ProjectFunction = MLP(in_channel, out_channel_list, activation_list, bn_list, drop_out_list)

    def forward(self, feats):
        out = self.ProjectFunction(feats)
        if self.normal:
            out = nn.functional.normalize(out)
        return out


class FactorAttention(nn.Module):
    def __init__(self, obj_channel, out_channel, use_sigmoid_or_softmax, use_L1=False):
        super(FactorAttention, self).__init__()
        self.use_sigmoid_or_softmax = use_sigmoid_or_softmax
        mid_channel = int(math.sqrt(obj_channel / out_channel) * out_channel)
        self.fc1 = nn.Linear(obj_channel, mid_channel)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(mid_channel, out_channel)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        self.use_L1 = use_L1

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        if self.use_sigmoid_or_softmax == 'sigmoid':
            out = self.sigmoid(self.fc2(out))
            if self.use_L1:
                out = out / (torch.sum(out, 1)).view(-1, 1)
        elif self.use_sigmoid_or_softmax == 'softmax':
            out = self.softmax(self.fc2(out))
        else:
            raise ValueError
        return out


class Analogies(nn.Module):

    def __init__(self, roi_xform_func, spatial_scale, box_head):
        super().__init__()

        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.crop_size = cfg.HICO.PART_CROP_SIZE
        self.box_head = box_head

        self.human_embedding = WordVecProjectionNet(in_channel=1024, out_channel_list=[1024, 1024],
                                                    drop_out_list=[True, False], normal=False)
        self.object_embedding = WordVecProjectionNet(in_channel=1024, out_channel_list=[1024, 1024],
                                                     drop_out_list=[True, False], normal=False)
        self.predicate_embedding = WordVecProjectionNet(in_channel=1024 + 8, out_channel_list=[1024, 1024],
                                                        drop_out_list=[True, False], normal=False)
        self.union_embedding = WordVecProjectionNet(in_channel=1024 + 8, out_channel_list=[1024, 1024],
                                                    drop_out_list=[True, False], normal=False)

        self.human_word_embedding = WordVecProjectionNet(in_channel=300, out_channel_list=[1024, 1024])
        self.object_word_embedding = WordVecProjectionNet(in_channel=300, out_channel_list=[1024, 1024])
        self.predicate_word_embedding = WordVecProjectionNet(in_channel=300, out_channel_list=[1024, 1024])
        self.union_word_embedding = WordVecProjectionNet(in_channel=900, out_channel_list=[1024, 1024])

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
        person_verb_object_vec = torch.from_numpy(hoi_blob["person_verb_object_vec"]).float().cuda(device_id)
        sp = torch.from_numpy(hoi_blob["sp"]).float().cuda(device_id)

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
        x_object = x_object.view(x_object.size(0), -1)
        x_human = x_human.view(x_human.size(0), -1)
        #########################################################

        interaction_human_inds = torch.from_numpy(
            hoi_blob['interaction_human_inds']).long().cuda(device_id)
        interaction_object_inds = torch.from_numpy(
            hoi_blob['interaction_object_inds']).long().cuda(device_id)

        human_embedding = self.human_word_embedding(person_verb_object_vec[:, 0:300])
        x_human = self.human_embedding(x_human)[interaction_human_inds]
        human_embedding_score = nn.Sigmoid()(torch.sum(x_human * human_embedding, 1))

        object_embedding = self.object_word_embedding(person_verb_object_vec[:, 600:900])
        x_object = self.object_embedding(x_object)[interaction_object_inds]
        object_embedding_score = nn.Sigmoid()(torch.sum(x_object * object_embedding, 1))

        union_embedding = self.union_word_embedding(person_verb_object_vec)
        x_union = self.union_embedding(torch.cat((x_union, sp), 1))
        union_embedding_score = nn.Sigmoid()(torch.sum(x_union * union_embedding, 1))

        pred_embedding = self.predicate_word_embedding(person_verb_object_vec[:, 300:600])
        x_pred = self.predicate_embedding(torch.cat((x_union, sp), 1))
        pred_embedding_score = nn.Sigmoid()(torch.sum(x_pred * pred_embedding, 1))

        hoi_blob['union_embedding_score'] = union_embedding_score
        hoi_blob['pred_embedding_score'] = pred_embedding_score
        hoi_blob['object_embedding_score'] = object_embedding_score
        hoi_blob['human_embedding_score'] = human_embedding_score

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
        union_embedding_score = hoi_blob['union_embedding_score']
        pred_embedding_score = hoi_blob['pred_embedding_score']
        object_embedding_score = hoi_blob['object_embedding_score']
        human_embedding_score = hoi_blob['human_embedding_score']

        device_id = union_embedding_score.get_device()
        interaction_action_labels_1 = torch.from_numpy(hoi_blob['interaction_action_labels_1']).float().cuda(device_id)

        union_embedding_loss = nn.BCELoss()(union_embedding_score, interaction_action_labels_1)
        pred_embedding_loss = nn.BCELoss()(pred_embedding_score, interaction_action_labels_1)
        object_embedding_loss = nn.BCELoss()(object_embedding_score, interaction_action_labels_1)
        human_embedding_loss = nn.BCELoss()(human_embedding_score, interaction_action_labels_1)

        return human_embedding_loss, object_embedding_loss, union_embedding_loss, pred_embedding_loss


class Analogies_H_O(nn.Module):

    def __init__(self, roi_xform_func, spatial_scale, box_head):
        super().__init__()

        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.crop_size = cfg.HICO.PART_CROP_SIZE
        self.box_head = box_head

        self.human_embedding = WordVecProjectionNet(in_channel=1024, out_channel_list=[1024, 1024],
                                                    drop_out_list=[True, False], normal=False)
        self.object_embedding = WordVecProjectionNet(in_channel=1024, out_channel_list=[1024, 1024],
                                                     drop_out_list=[True, False], normal=False)

        self.human_word_embedding = WordVecProjectionNet(in_channel=300, out_channel_list=[1024, 1024])
        self.object_word_embedding = WordVecProjectionNet(in_channel=300, out_channel_list=[1024, 1024])

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
        person_vec = torch.from_numpy(hoi_blob["person_vec"]).float().cuda(device_id)
        object_vec = torch.from_numpy(hoi_blob["object_vec"]).float().cuda(device_id)

        x_human = self.box_head(
            x, hoi_blob,
            blob_rois='human_boxes',
        )
        x_object = self.box_head(
            x, hoi_blob,
            blob_rois='object_boxes'
        )

        x_object = x_object.view(x_object.size(0), -1)
        x_human = x_human.view(x_human.size(0), -1)
        #########################################################

        interaction_human_inds = torch.from_numpy(
            hoi_blob['interaction_human_inds']).long().cuda(device_id)
        interaction_object_inds = torch.from_numpy(
            hoi_blob['interaction_object_inds']).long().cuda(device_id)

        human_embedding = self.human_word_embedding(person_vec)
        x_human = self.human_embedding(x_human)[interaction_human_inds]
        human_embedding_score = nn.Sigmoid()(torch.sum(x_human * human_embedding, 1))

        object_embedding = self.object_word_embedding(object_vec)
        x_object = self.object_embedding(x_object)[interaction_object_inds]
        object_embedding_score = nn.Sigmoid()(torch.sum(x_object * object_embedding, 1))

        hoi_blob['object_embedding_score'] = object_embedding_score
        hoi_blob['human_embedding_score'] = human_embedding_score

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
        object_embedding_score = hoi_blob['object_embedding_score']
        human_embedding_score = hoi_blob['human_embedding_score']

        device_id = human_embedding_score.get_device()
        human_labels = torch.from_numpy(hoi_blob['human_labels']).float().cuda(device_id)
        object_labels = torch.from_numpy(hoi_blob['object_labels']).float().cuda(device_id)

        object_embedding_loss = nn.BCELoss()(object_embedding_score, object_labels) * 0.1
        human_embedding_loss = nn.BCELoss()(human_embedding_score, human_labels) * 0.1

        human_preds = \
            (human_embedding_score > cfg.HICO.ACTION_THRESH).type_as(human_labels)
        human_accuray_cls = human_preds.eq(human_labels).float().mean()
        object_preds = \
            (object_embedding_score > cfg.HICO.ACTION_THRESH).type_as(object_labels)
        object_accuray_cls = object_preds.eq(object_labels).float().mean()

        return human_embedding_loss, object_embedding_loss, \
               torch.zeros(human_embedding_loss.shape).cuda(device_id), \
               torch.zeros(human_embedding_loss.shape).cuda(device_id), \
               human_accuray_cls, object_accuray_cls
