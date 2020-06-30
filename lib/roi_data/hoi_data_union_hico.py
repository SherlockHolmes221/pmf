from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import numpy.random as npr
import ipdb

from core.config import cfg
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.keypoints as keypoint_utils
from utils.fpn import distribute_rois_over_fpn_levels
import cv2


def get_hoi_union_blob_names(is_training=True):
    """

    :param is_training:
    :return:
    """
    blob_names = ['human_boxes', 'object_boxes', 'union_boxes', 'rescale_kps', 'union_mask', 'spatial_info',
                  'interaction_human_inds', 'interaction_object_inds', 'interaction_affinity',
                  'gt_union_heatmap', 'interaction_init_part_attens', "human_scores", "object_scores", "sp",
                  'poseconfig', 'part_boxes', 'flag', 'object_mask', 'verb_object_vec', 'person_verb_object_vec',
                  "person_vec", "object_vec"]

    if is_training:
        blob_names.extend(
            ['interaction_action_labels', "copy_num_vec", "copy_num", 'interaction_action_labels_1', "object_labels",
             "human_labels"])

    blob_names = {k: [] for k in blob_names}
    return blob_names


def sample_for_hoi_branch(rpn_net, roidb, im_info,
                          cls_score=None, bbox_pred=None, is_training=True):
    hoi_blob_names = get_hoi_union_blob_names(is_training=is_training)
    if is_training:
        # list of sample result
        blobs_list = sample_for_hoi_branch_train(rpn_net, roidb, im_info)
    else:
        raise NotImplementedError
    hoi_blob_in = merge_hoi_blobs(hoi_blob_names, blobs_list)
    return hoi_blob_in


def merge_hoi_blobs(hoi_blob_in, blobs_list):
    '''
    Merge blob of each image
    :param hoi_blob_in: hoi blob names dict
    :param blobs_list: blob of each image
    :return:
    '''
    # support mini-batch
    human_boxes_count = 0
    object_boxes_count = 0
    for i in range(len(blobs_list)):
        blob_this_im = blobs_list[i]
        # ensure interaction_*_inds only index correct image's human/target_object feature
        blob_this_im['interaction_human_inds'] += human_boxes_count
        blob_this_im['interaction_object_inds'] += object_boxes_count

        # count human/object rois num
        human_boxes_count += blob_this_im['human_boxes'].shape[0]
        object_boxes_count += blob_this_im['object_boxes'].shape[0]
        # Append to blob list
        for k, v in blob_this_im.items():
            hoi_blob_in[k].append(v)

    # Concat the training blob lists into tensors
    # np.concatenate default axis=0
    for k, v in hoi_blob_in.items():
        if len(v) > 0:
            hoi_blob_in[k] = np.concatenate(v)

    # Add FPN multilevel training RoIs, if configured
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        distribute_rois_over_fpn_levels(hoi_blob_in, 'human_boxes')
        distribute_rois_over_fpn_levels(hoi_blob_in, 'object_boxes')
        distribute_rois_over_fpn_levels(hoi_blob_in, 'union_boxes')

    return hoi_blob_in


def sample_for_hoi_branch_train(rpn_ret, roidb, im_info):
    '''
    hoi: human-object interaction
    Sampling for human-centric branch and interaction branch
    :param rpn_ret:
    :param roidb:
    :return:
    '''
    # Select proposals(rois) that IoU with gt >= 0.5 for human-centric branch
    # and interaction branch
    print('cfg.TRAIN.FG_THRESH: ', cfg.TRAIN.FG_THRESH)
    keep_rois_inds = np.where(rpn_ret['rois_max_overlaps'] >= cfg.TRAIN.FG_THRESH)
    rois = rpn_ret['rois'][keep_rois_inds]
    rois_to_gt_ind = rpn_ret['rois_to_gt_ind_map'][keep_rois_inds]

    train_hoi_blobs = []
    # get blobs of each image
    for i, entry in enumerate(roidb):
        inds_of_this_image = np.where(rois[:, 0] == i)
        blob_this_im = _sample_human_union_boxes(rois[inds_of_this_image],
                                                 rois_to_gt_ind[inds_of_this_image],
                                                 entry,
                                                 im_info[i],
                                                 i)
        train_hoi_blobs.append(blob_this_im)

    return train_hoi_blobs


def sample_for_hoi_branch_precomp_box_train(roidb, im_info, is_training=True):
    '''
    hoi: human-object interaction
    Sampling for human-centric branch and interaction branch
    :param rpn_ret:
    :param roidb:
    :return:
    '''

    hoi_blob_names = get_hoi_union_blob_names(is_training=is_training)
    scales = im_info.data.numpy()[:, 2]

    train_hoi_blobs = []
    # get blobs of each image
    for i, entry in enumerate(roidb):
        # print(entry['image_id'])
        rois = entry['precomp_boxes'] * scales[i]
        rois = np.concatenate([np.full((len(entry['precomp_boxes']), 1), i), rois], 1).astype(np.float32)

        blob_this_im = _sample_human_union_boxes(entry['precomp_boxes'],
                                                 rois,
                                                 entry,
                                                 im_info[i],
                                                 i)

        if blob_this_im is not None:
            train_hoi_blobs.append(blob_this_im)

    if len(train_hoi_blobs) == 0:
        return None

    hoi_blob_in = merge_hoi_blobs(hoi_blob_names, train_hoi_blobs)
    return hoi_blob_in


def _sample_human_union_boxes(origin_rois, rois, roidb, im_info, batch_idx):
    """
    :param rois: fg rois(gt boxes have been added to rois, see roi_data/fast_rcnn)
    :param roidb:
    :param im_info:
    :return:
    """

    rois_human_inds = np.where(roidb['precomp_cate'] == 1)[0]
    human_rois = rois[rois_human_inds]
    # print(rois_human_inds) # [0 1 2 3 4 5]
    # print(human_rois.shape) # (6, 5)

    rois_object_inds = np.arange(rois.shape[0])
    object_rois = rois
    obj_cats = roidb['precomp_cate']
    obj_mask = roidb['precomp_prob_mask']

    if len(rois_human_inds) == 0 or len(rois_object_inds) == 0:
        return None

    triplets_info = \
        generate_triplets(origin_rois, rois, rois_human_inds, rois_object_inds, obj_cats,
                          roidb['gt_human_boxes'], roidb['gt_object_boxes'], roidb['gt_verb_index'],
                          roidb['gt_obj_cat'],
                          batch_idx=batch_idx)

    rois_keypoints = roidb['precomp_keypoints']
    human_keypoints = rois_keypoints[rois_human_inds]

    part_boxes, flag = generate_part_box_from_kp17(human_keypoints, human_rois, float(im_info[2]),
                                                   body_ratio=cfg.HICO.BODY_RATIO, head_ratio=1.5)

    union_gt_kps = rois_keypoints[rois_human_inds[triplets_info['human_inds']]]

    gt_union_heatmap, union_mask, rescale_kps = \
        generate_joints_heatmap(union_gt_kps,
                                triplets_info['union_boxes'][:, 1:] / float(im_info[2]),
                                rois[rois_human_inds[triplets_info['human_inds']]][:, 1:] / float(im_info[2]), \
                                rois[rois_object_inds[triplets_info['object_inds']]][:, 1:] / float(im_info[2]), \
                                gaussian_kernel=(cfg.HICO.HEATMAP_KERNEL_SIZE,
                                                 cfg.HICO.HEATMAP_KERNEL_SIZE))

    poseconfig = generate_pose_configmap(union_gt_kps, triplets_info['union_boxes'][:, 1:] / float(im_info[2]), \
                                         rois[rois_human_inds[triplets_info['human_inds']]][:, 1:] / float(im_info[2]), \
                                         rois[rois_object_inds[triplets_info['object_inds']]][:, 1:] / float(
                                             im_info[2]))

    # print(poseconfig)
    # print(human_rois.shape, object_rois.shape, triplets_info['union_boxes'].shape,
    #       triplets_info['spatial_info'].shape, triplets_info['human_inds'].shape,
    #       triplets_info['object_inds'].shape)
    # assert False
    # (4, 5)(6, 5)(16, 5)(16, 4)(16, )(16, )
    # obj_cat = obj_cats[triplets_info['object_inds']]
    object_mask = obj_mask[triplets_info['object_inds']]
    labels = triplets_info['action_labels']
    index1, index2 = np.where(labels == 1)
    for i in range(len(index1)):
        assert object_mask[index1[i]][index2[i]] == 1

    # # todo for test
    # triplets_info['human_inds'] = triplets_info['human_inds'][:, np.newaxis][0]
    # triplets_info['object_inds'] = triplets_info['object_inds'][:, np.newaxis][0]
    # triplets_info['union_boxes'] = triplets_info['union_boxes'][0][np.newaxis, :]
    # triplets_info['action_labels'] = triplets_info['action_labels'][0][np.newaxis, :]
    # poseconfig = poseconfig[0][np.newaxis, :]
    # obj_mask = obj_mask[triplets_info['object_inds']][0]
    # print(np.where(obj_mask > 0))
    # print(obj_cats[0])
    # print(np.where(triplets_info['action_labels'] > 0))
    # print(triplets_info['union_boxes'])
    # print(triplets_info['human_inds'])
    # print(triplets_info['object_inds'])
    # (array([0]), array([36]))
    # [[0.        23.937649 563.2082   488.78192  782.74603]]
    # [2]
    # [31]
    # todo end testing
    # print(triplets_info['interaction_affinity'])
    return_dict = dict(
        human_boxes=human_rois,
        object_boxes=object_rois,
        union_boxes=triplets_info['union_boxes'],
        interaction_human_inds=triplets_info['human_inds'],
        interaction_object_inds=triplets_info['object_inds'],
        interaction_action_labels=triplets_info['action_labels'],
        interaction_affinity=triplets_info['interaction_affinity'][:, np.newaxis].astype(np.float32),
        part_boxes=part_boxes,
        flag=flag,
        poseconfig=poseconfig,
        object_mask=obj_mask,
        # gt_union_heatmap=gt_union_heatmap,
        # rescale_kps=rescale_kps,
        # spatial_info=triplets_info['spatial_info'],
        # union_mask=union_mask
    )

    return return_dict


def generate_joints_heatmap(union_kps, union_rois, human_rois, obj_rois, gaussian_kernel=(7, 7)):
    # ipdb.set_trace()
    num_triplets, _, kp_num = union_kps.shape
    ret = np.zeros((num_triplets, kp_num + 2, cfg.KRCNN.HEATMAP_SIZE, cfg.KRCNN.HEATMAP_SIZE))
    union_mask = np.zeros((num_triplets, 5, cfg.KRCNN.HEATMAP_SIZE, cfg.KRCNN.HEATMAP_SIZE))
    rescale_kps = np.zeros((num_triplets, kp_num, 2)).astype(np.int32)

    ux0 = union_rois[:, 0]
    uy0 = union_rois[:, 1]
    ux1 = union_rois[:, 2]
    uy1 = union_rois[:, 3]

    scale_x = cfg.KRCNN.HEATMAP_SIZE / (ux1 - ux0)
    scale_y = cfg.KRCNN.HEATMAP_SIZE / (uy1 - uy0)

    for i in range(num_triplets):
        for j in range(kp_num):
            vis = union_kps[i, -1, j]
            if vis > 0:
                kpx, kpy = union_kps[i, :2, j]
                if kpx < ux0[i] or kpy < uy0[i] or kpx > ux1[i] or kpy > uy1[i]:
                    continue
                kpx = np.clip(np.round((kpx - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
                kpy = np.clip(np.round((kpy - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
                rescale_kps[i, j] = np.array([kpx, kpy])
                ret[i, j, kpy, kpx] = 1.  # 1.0
                ret[i, j] = cv2.GaussianBlur(ret[i, j], gaussian_kernel, 0)
                am = np.amax(ret[i, j])
                ret[i, j] /= am

        ox0, oy0, ox1, oy1 = human_rois[i]
        ox0 = np.clip(np.round((ox0 - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
        oy0 = np.clip(np.round((oy0 - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
        ox1 = np.clip(np.round((ox1 - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
        oy1 = np.clip(np.round((oy1 - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
        ret[i, -2, oy0:oy1, ox0:ox1] = 1.0
        ret[i, -2] -= 0.5
        union_mask[i, 0, oy0:oy1, ox0:ox1] = 1.

        ox0, oy0, ox1, oy1 = obj_rois[i]
        ox0 = np.clip(np.round((ox0 - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
        oy0 = np.clip(np.round((oy0 - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
        ox1 = np.clip(np.round((ox1 - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
        oy1 = np.clip(np.round((oy1 - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
        ret[i, -1, oy0:oy1, ox0:ox1] = 1.0
        ret[i, -1] -= 0.5
        union_mask[i, 1, oy0:oy1, ox0:ox1] = 1.
        union_mask[i, 2] = np.maximum(union_mask[i, 0], union_mask[i, 1])

        anoise = 0.1 * np.random.random((cfg.KRCNN.HEATMAP_SIZE, cfg.KRCNN.HEATMAP_SIZE))
        union_mask[i, 3] = np.maximum(union_mask[i, 2], anoise)

        anoise2 = 0.1 * np.ones((cfg.KRCNN.HEATMAP_SIZE, cfg.KRCNN.HEATMAP_SIZE))
        union_mask[i, 4] = np.maximum(union_mask[i, 2], anoise2)

    return ret.astype(np.float32), union_mask.astype(np.float32), rescale_kps


def generate_pose_configmap(union_kps, union_rois, human_rois, obj_rois, gaussian_kernel=(7, 7)):
    # generate spatial configuration of pose map: 64 x 64
    # draw lines with different values
    skeletons = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], \
                 [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], \
                 [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    num_triplets, _, kp_num = union_kps.shape
    ret = np.zeros((num_triplets, 1 + 2, cfg.KRCNN.HEATMAP_SIZE, cfg.KRCNN.HEATMAP_SIZE))

    ux0 = union_rois[:, 0]
    uy0 = union_rois[:, 1]
    ux1 = union_rois[:, 2]
    uy1 = union_rois[:, 3]

    scale_x = cfg.KRCNN.HEATMAP_SIZE / (ux1 - ux0)
    scale_y = cfg.KRCNN.HEATMAP_SIZE / (uy1 - uy0)
    for i in range(1, num_triplets):
        cur_kps = np.zeros((kp_num, 2)).astype(np.int32)
        vis = union_kps[i, -1]
        for j in range(kp_num):
            # vis = union_kps[i, -1, j]
            # if vis>0 and vis<1:
            #    print(union_kps[i,:,j])
            if vis[j] > 0:
                kpx, kpy = union_kps[i, :2, j]
                kpx = np.round((kpx - ux0[i]) * scale_x[i]).astype(np.int)
                kpy = np.round((kpy - uy0[i]) * scale_y[i]).astype(np.int)
                cur_kps[j] = np.array([kpx, kpy])

        for j, sk in enumerate(skeletons):
            sk0 = sk[0] - 1
            sk1 = sk[1] - 1
            if vis[sk0] > 0 and vis[sk1] > 0:
                ret[i, 0] = cv2.line(ret[i, 0], tuple(cur_kps[sk0]), tuple(cur_kps[sk1]), 0.05 * (j + 1), 3)

        ox0, oy0, ox1, oy1 = human_rois[i]
        ox0 = np.clip(np.round((ox0 - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
        oy0 = np.clip(np.round((oy0 - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
        ox1 = np.clip(np.round((ox1 - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
        oy1 = np.clip(np.round((oy1 - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
        ret[i, 1, oy0:oy1, ox0:ox1] = 1.0

        ox0, oy0, ox1, oy1 = obj_rois[i]
        ox0 = np.clip(np.round((ox0 - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
        oy0 = np.clip(np.round((oy0 - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
        ox1 = np.clip(np.round((ox1 - ux0[i]) * scale_x[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
        oy1 = np.clip(np.round((oy1 - uy0[i]) * scale_y[i]).astype(np.int), 0, cfg.KRCNN.HEATMAP_SIZE - 1)
        ret[i, 2, ox0:ox1, oy0:oy1] = 1.0
        #
        # skeleton = ret[i, 0, :, :].reshape([64, 64, 1])
        # cv2.imshow("Joints", skeleton)
        # cv2.waitKey()
        # skeleton = ret[i, 1, :, :].reshape([64, 64, 1])
        # cv2.imshow("human", skeleton)
        # cv2.waitKey()
        # skeleton = ret[i, 2, :, :].reshape([64, 64, 1])
        # cv2.imshow("O
        # bject", skeleton)
        # cv2.waitKey()

    return ret.astype(np.float32)





def generate_part_box_from_kp(all_kps, human_rois, scale, body_ratio=0.1, head_ratio=1.5):
    """
    :param kps: human_roi_num*17*3
    :param human_roi: human_roi_num*5
    :return: human_roi_num*13*5
    """
    assert all_kps.shape[0] == human_rois.shape[0]
    human_num, _, kp_num = all_kps.shape
    # 13 for head and 12 other parts, head in in last channel
    ret = -np.ones([human_num, 13, 4]).astype(human_rois.dtype)
    flag = np.zeros([human_num, 13])

    for h in range(human_num):
        x0, y0, x1, y1 = human_rois[h, 1:]
        width = x1 - x0
        height = y1 - y0
        length = max(width, height)

        ### TODO: for vis: kps = all_kps[h, :2]
        kps = all_kps[h, :2] * scale

        vis = all_kps[h, -1]
        valid_ind = np.where(vis > 0)[0]
        head_ind = []
        for ind in valid_ind:
            if ind < 5:
                head_ind.append(ind)
            else:
                flag[h, ind - 5] = 1
                kp_x, kp_y = kps[:, ind]
                # x_min = np.clip(kp_x - ratio_x*width, x0, x1)
                # x_max = np.clip(kp_x + ratio_x*width, x0, x1)
                # y_min = np.clip(kp_y - ratio_y*height, y0, y1)
                # y_max = np.clip(kp_y + ratio_y*height, y0, y1)
                x_min = np.clip(kp_x - body_ratio * length, x0, x1)
                x_max = np.clip(kp_x + body_ratio * length, x0, x1)
                y_min = np.clip(kp_y - body_ratio * length, y0, y1)
                y_max = np.clip(kp_y + body_ratio * length, y0, y1)
                ret[h, ind - 5] = np.array([x_min, y_min, x_max, y_max])

        if len(head_ind) > 0:
            flag[h, -1] = 1
            head_ind = np.array(head_ind)
            # ipdb.set_trace()
            x_head_min = np.min(kps[0, head_ind])
            x_head_max = np.max(kps[0, head_ind])
            y_head_min = np.min(kps[1, head_ind])
            y_head_max = np.max(kps[1, head_ind])
            head_width = x_head_max - x_head_min
            head_height = y_head_max - y_head_min
            x_center = (x_head_max + x_head_min) / 2.0
            y_center = (y_head_max + y_head_min) / 2.0
            x_min = np.clip(x_center - head_width * head_ratio / 2.0, x0, x1)
            x_max = np.clip(x_center + head_width * head_ratio / 2.0, x0, x1)
            y_min = np.clip(y_center - head_height * head_ratio / 2.0, y0, y1)
            y_max = np.clip(y_center + head_height * head_ratio / 2.0, y0, y1)
            ret[h, -1] = np.array([x_min, y_min, x_max, y_max])
    ret = np.concatenate([human_rois[:, [0]].repeat(13, 1)[:, :, None], ret], -1)
    return ret, flag


def generate_part_box_from_kp17(all_kps, human_rois, scale, body_ratio=0.1, head_ratio=1.5):
    """
    :param kps: human_roi_num*17*3
    :param human_roi: human_roi_num*5
    :return: human_roi_num*13*5
    """
    #  print(all_kps.shape, human_rois.shape)
    assert all_kps.shape[0] == human_rois.shape[0]
    human_num, _, kp_num = all_kps.shape
    ret = -np.ones([human_num, 17, 4]).astype(human_rois.dtype)
    flag = np.zeros([human_num, 17])

    for h in range(human_num):
        x0, y0, x1, y1 = human_rois[h, 1:]
        width = x1 - x0
        height = y1 - y0
        length = max(width, height)

        kps = all_kps[h, :2] * scale
        vis = all_kps[h, -1]
        valid_ind = np.where(vis > 0)[0]

        for ind in valid_ind:
            flag[h, ind] = 1
            kp_x, kp_y = kps[:, ind]
            x_min = np.clip(kp_x - body_ratio * length, x0, x1)
            x_max = np.clip(kp_x + body_ratio * length, x0, x1)
            y_min = np.clip(kp_y - body_ratio * length, y0, y1)
            y_max = np.clip(kp_y + body_ratio * length, y0, y1)
            ret[h, ind] = np.array([x_min, y_min, x_max, y_max])
    ret = np.concatenate([human_rois[:, [0]].repeat(17, 1)[:, :, None], ret], -1)

    return ret, flag


def get_action_labels(pred_human_boxes, pred_obj_boxes, pred_obj_cats,
                      gt_human_boxes, gt_object_boxes, gt_verb_index, gt_obj_cat):
    human_pred_to_gt = box_utils.bbox_overlaps(
        pred_human_boxes.astype(dtype=np.float32, copy=False),
        gt_human_boxes.astype(dtype=np.float32, copy=False)
    )

    object_pred_to_gt = box_utils.bbox_overlaps(
        pred_obj_boxes.astype(dtype=np.float32, copy=False),
        gt_object_boxes.astype(dtype=np.float32, copy=False)
    )
    # print(human_pred_to_gt.shape) # (60, 2)
    # print(object_pred_to_gt.shape) # (60, 2)
    # print(object_pred_to_gt)
    # print(human_pred_to_gt.shape, object_pred_to_gt.shape)  # (4, 2) (4, 2)
    # print(pred_obj_cats.shape, gt_obj_cat.shape)  # (4,) (2, 1)

    pred_obj_cats = pred_obj_cats[:, np.newaxis].repeat(gt_obj_cat.shape[0], axis=1)
    gt_obj_cat = gt_obj_cat.reshape((1, gt_obj_cat.shape[0])).repeat(pred_obj_cats.shape[0], axis=0)

    # print(pred_obj_cats)
    # print(gt_obj_cat)
    # print(pred_obj_cats.shape)
    # print(gt_obj_cat.shape)

    obj_mat = pred_obj_cats - gt_obj_cat
    # print(obj_mat)
    indexs = np.where((human_pred_to_gt > 0.5) & (object_pred_to_gt > 0.5) & (obj_mat == 0))
    # print(indexs)
    # print(gt_verb_index)

    action_labels = np.zeros([len(pred_human_boxes), 117])
    # interaction_affinity = np.zeros([len(pred_human_boxes)])

    for i in range(len(indexs[0])):
        pred_index = indexs[0][i]
        gt_index = indexs[1][i]
        gt_verb = gt_verb_index[gt_index][0]
        action_labels[pred_index][gt_verb] = 1
        # interaction_affinity[pred_index] = 1
    # print(np.where(action_labels == 1))

    # action_labels[indexs[0]][gt_verb_index[indexs[1]]] = 1
    # print(np.where(action_labels == 1))
    interaction_affinity = np.any(action_labels.reshape(action_labels.shape[0], -1) > 0, 1)
    # index1 = np.where(action_labels > 0)[0]
    # interaction_affinity = np.zeros([len(pred_human_boxes)])
    # interaction_affinity[index1] = 1

    return action_labels, interaction_affinity


def get_HO_labels(pred_human_boxes, pred_obj_boxes, pred_obj_cats,
                  gt_human_boxes, gt_object_boxes, gt_obj_cat):
    human_pred_to_gt = box_utils.bbox_overlaps(
        pred_human_boxes.astype(dtype=np.float32, copy=False),
        gt_human_boxes.astype(dtype=np.float32, copy=False)
    )

    object_pred_to_gt = box_utils.bbox_overlaps(
        pred_obj_boxes.astype(dtype=np.float32, copy=False),
        gt_object_boxes.astype(dtype=np.float32, copy=False)
    )
    # print(human_pred_to_gt.shape, object_pred_to_gt.shape) # (10, 12) (63, 12)

    pred_obj_cats = pred_obj_cats[:, np.newaxis].repeat(gt_obj_cat.shape[0], axis=1)
    gt_obj_cat = gt_obj_cat.reshape((1, gt_obj_cat.shape[0])).repeat(pred_obj_cats.shape[0], axis=0)
    # print(pred_obj_cats.shape, gt_obj_cat.shape) # (63, 12) (63, 12)

    obj_mat = pred_obj_cats - gt_obj_cat
    index_human = np.unique(np.where((human_pred_to_gt > 0.5))[0])
    index_object = np.unique(np.where((object_pred_to_gt > 0.5) & (obj_mat == 0))[0])
    # print(index_human, index_object) # [0 1 2 3] [11 12 13 15 17]

    human_labels = np.zeros([len(pred_human_boxes), 1])
    object_labels = np.zeros([len(pred_obj_boxes), 1])

    for i in range(len(index_human)):
        pred_index = index_human[i]
        human_labels[pred_index][0] = 1

    for i in range(len(index_object)):
        pred_index = index_object[i]
        object_labels[pred_index][0] = 1

    # print(np.where(human_labels > 0)[0])
    # print(np.where(object_labels > 0)[0])
    # [0 1 2 3]
    # [11 12 13 15 17]

    return human_labels, object_labels


def generate_triplets(origin_rois, rois, rois_human_inds, rois_object_inds, obj_cats,
                      gt_human_boxes, gt_object_boxes, gt_verb_index, gt_obj_cat, batch_idx):
    # print(len(box_index_to_gt_index))
    # print(box_index_to_gt_index)
    """
    :param rois:
    :param rois_human_inds: human ind to rois index
    :param rois_object_inds:
    :param batch_idx:
    :return:
    """
    triplets_num_per_image = cfg.HICO.TRIPLETS_NUM_PER_IM
    # print(triplets_num_per_image) # 32

    # generate combinations
    human_rois_inds, object_rois_inds = np.meshgrid(np.arange(rois_human_inds.size),
                                                    np.arange(rois_object_inds.size),
                                                    indexing='ij')
    human_rois_inds, object_rois_inds = human_rois_inds.reshape(-1), object_rois_inds.reshape(-1)
    # print(len(human_rois_inds)) # 60
    # print(len(object_rois_inds))# 60
    # # # [0  0  0... 17 17 17]
    # [0  1  2... 97 98 99]

    union_boxes = box_utils.get_union_box(rois[rois_human_inds[human_rois_inds]][:, 1:],
                                          rois[rois_object_inds[object_rois_inds]][:, 1:])
    union_boxes = np.concatenate(
        (batch_idx * np.ones((union_boxes.shape[0], 1), dtype=union_boxes.dtype),
         union_boxes), axis=1)

    # print(union_boxes)
    # [[0.        358.73096    12.991829  738.20575   587.8608]
    #  [0.         42.26664    12.991829 1054.974     691.46246]]

    relative_location = box_utils.bbox_transform_inv(rois[rois_human_inds[human_rois_inds]][:, 1:],
                                                     rois[rois_object_inds[object_rois_inds]][:, 1:])

    # gt_len = len(gt_human_boxes)
    # assert len(gt_human_boxes) == len(gt_object_boxes)

    # print(gt_len)
    # box_index_to_gt_index = np.array([i - gt_len if i >= gt_len else i
    #                                   for i in box_index_to_gt_index])

    # human_boxes_to_gt_index = box_index_to_gt_index[rois_human_inds[human_rois_inds]]
    # object_boxes_to_gt_index = box_index_to_gt_index[rois_object_inds[object_rois_inds]]
    # object_boxes_to_gt_index = np.array([i - gt_len for i in object_boxes_to_gt_index])
    # # print(object_boxes_to_gt_index)
    # print(object_boxes_to_gt_index)

    # indexs = np.where(human_boxes_to_gt_index == object_boxes_to_gt_index)
    # print(gt_len)
    # print(indexs)
    # assert False
    # (array([0, 17, 28, 34, 51, 68, 85, 102, 106, 119, 136, 153, 158,
    #         159, 166, 170, 187, 193, 204]),)
    # pair_index = indexs[0]
    # assert (human_boxes_to_gt_index[pair_index] == object_boxes_to_gt_index[pair_index]).all()
    # gt_indexs = human_boxes_to_gt_index[pair_index]
    # print(gt_len) # 2
    # print(gt_indexs) # [0 0]
    # [15 14 14  9 10  5  3  7  7 11 12  8  8  8  7  7  6 14 14]

    # print(gt_verb_vec.shape) # (16, 117)

    # action_labels = np.zeros([len(human_rois_inds), 117])
    # action_labels[pair_index] = gt_verb_vec[gt_indexs]
    # interaction_affinity = np.zeros([len(human_rois_inds), 1])
    # interaction_affinity[pair_index] = 1
    # print(action_labels.shape) # (208, 117)
    # print(interaction_affinity.shape) # (208, 1)
    # print(np.where(action_labels > 0))
    # print(np.where(interaction_affinity > 0))
    # (array([0, 17, 28, 34, 51, 68, 85, 102, 106, 119, 136, 153, 158,
    #         159, 166, 170, 187, 193, 204]), array([86, 86, 86, 86, 86, 24, 24, 24, 24, 86, 86, 86, 86, 86, 24, 24, 24,
    #                                                86, 86]))
    # (array([0, 17, 28, 34, 51, 68, 85, 102, 106, 119, 136, 153, 158,
    #         159, 166, 170, 187, 193, 204]), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    action_labels, interaction_affinity = \
        get_action_labels(origin_rois[rois_human_inds[human_rois_inds]],
                          origin_rois[rois_object_inds[object_rois_inds]],
                          obj_cats[rois_object_inds[object_rois_inds]],
                          gt_human_boxes, gt_object_boxes, gt_verb_index, gt_obj_cat)

    # print(np.where(action_labels > 0))
    # print(np.where(interaction_affinity > 0))
    fg_triplets_inds = np.where(np.sum(action_labels, axis=1) > 0)[0]
    bg_triplets_inds = np.setdiff1d(np.arange(action_labels.shape[0]), fg_triplets_inds)
    # print(fg_triplets_inds, bg_triplets_inds)
    # [ 4 10 17 22] [ 0  1  2  3  5  6  7  8  9 11 12 13 14 15 16 18 19 20 21 23]

    fg_triplets_num_this_image = min(int(triplets_num_per_image * 1 / 4.), fg_triplets_inds.size)

    if fg_triplets_inds.size > 0:
        fg_triplets_inds = npr.choice(
            fg_triplets_inds, size=fg_triplets_num_this_image, replace=False)

    bg_triplets_num_this_image = max(fg_triplets_num_this_image * 3, 1)
    bg_triplets_num_this_image = min(bg_triplets_num_this_image, bg_triplets_inds.size)
    # print(fg_triplets_num_this_image, bg_triplets_num_this_image)

    if bg_triplets_inds.size > 0 and bg_triplets_num_this_image > 0:
        bg_triplets_inds = npr.choice(
            bg_triplets_inds, size=bg_triplets_num_this_image, replace=False)

        keep_triplets_inds = np.concatenate((fg_triplets_inds, bg_triplets_inds))
    else:
        keep_triplets_inds = fg_triplets_inds
    # print(keep_triplets_inds)
    # [10 17 22  4  7 11  5 16 23 14  2 19 18 21 12  6]

    # print(human_rois_inds.shape, object_rois_inds.shape, union_boxes.shape, action_labels.shape,
    #       relative_location.shape, interaction_affinity.shape)
    # (24,)(24, )(24, 5)(24, 117)(24, 4)(24, 1)
    assert (rois_human_inds[human_rois_inds[keep_triplets_inds]] == human_rois_inds[keep_triplets_inds]).all()

    return_dict = dict(
        human_inds=rois_human_inds[human_rois_inds[keep_triplets_inds]],
        object_inds=rois_object_inds[object_rois_inds[keep_triplets_inds]],
        union_boxes=union_boxes[keep_triplets_inds],
        action_labels=action_labels[keep_triplets_inds],
        spatial_info=relative_location[keep_triplets_inds],
        interaction_affinity=interaction_affinity[keep_triplets_inds],
    )

    return return_dict


def get_location_info(human_boxes, object_boxes, union_boxes):
    assert human_boxes.shape[1] == object_boxes.shape[1] == union_boxes.shape[1] == 4
    human_object_loc = box_utils.bbox_transform_inv(human_boxes, object_boxes)
    human_union_loc = box_utils.bbox_transform_inv(human_boxes, union_boxes)
    object_union_loc = box_utils.bbox_transform_inv(object_boxes, union_boxes)
    return np.concatenate((human_object_loc, human_union_loc, object_union_loc), axis=1)


def generate_union_mask(human_rois, object_rois, union_rois, human_inds, object_inds):
    union_mask = np.zeros((human_inds.size, 2, cfg.KRCNN.HEATMAP_SIZE, cfg.KRCNN.HEATMAP_SIZE))
    pooling_size = cfg.KRCNN.HEATMAP_SIZE
    for i in range(human_inds.size):
        union_left_top = np.tile(union_rois[i, 1:3], 2)
        w, h = union_rois[i, 3:5] - union_rois[i, 1:3]
        weights_t = pooling_size / np.array([w, h, w, h])
        human_coord = ((human_rois[human_inds][i, 1:] - union_left_top) * weights_t).astype(np.int32)
        object_coord = ((object_rois[object_inds][i, 1:] - union_left_top) * weights_t).astype(np.int32)
        union_mask[i, 0, human_coord[1]:human_coord[3] + 1, human_coord[0]:human_coord[2] + 1] = 1
        union_mask[i, 1, object_coord[1]:object_coord[3] + 1, object_coord[0]:object_coord[2] + 1] = 1
    return union_mask.astype(np.float32)
