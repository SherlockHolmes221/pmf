import os
import numpy as np
import sys

sys.path.insert(0, "lib")
from utils.bbox_utils import compute_iou
import json
import utils.io_utils as io
import h5py


def match_hoi(pred_det, gt_dets):
    is_match = False
    remaining_gt_dets = [gt_det for gt_det in gt_dets]
    for i, gt_det in enumerate(gt_dets):
        human_iou = compute_iou(pred_det['human_box'], gt_det['human_box'])
        if human_iou > 0.5:
            object_iou = compute_iou(pred_det['object_box'], gt_det['object_box'])
            if object_iou > 0.5:
                is_match = True
                del remaining_gt_dets[i]
                break

    return is_match, remaining_gt_dets


def compute_ap(precision, recall):
    if np.any(np.isnan(recall)):
        return np.nan

    ap = 0
    for t in np.arange(0, 1.1, 0.1):  # 0, 0.1, 0.2, ..., 1.0
        selected_p = precision[recall >= t]
        if selected_p.size == 0:
            p = 0
        else:
            p = np.max(selected_p)
        ap += p / 11.

    return ap


def compute_pr(y_true, y_score, npos):
    sorted_y_true = [y for y, _ in
                     sorted(zip(y_true, y_score), key=lambda x: x[1], reverse=True)]
    tp = np.array(sorted_y_true)

    if len(tp) == 0:
        return 0, 0, False

    fp = ~tp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    if npos == 0:
        recall = np.nan * tp
    else:
        recall = tp / npos
    precision = tp / (tp + fp)
    return precision, recall, True


def compute_normalized_pr(y_true, y_score, npos, N=196.45):
    sorted_y_true = [y for y, _ in
                     sorted(zip(y_true, y_score), key=lambda x: x[1], reverse=True)]
    tp = np.array(sorted_y_true)
    fp = ~tp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    if npos == 0:
        recall = np.nan * tp
    else:
        recall = tp / npos
    precision = recall * N / (recall * N + fp)
    nap = np.sum(precision[sorted_y_true]) / (npos + 1e-6)
    return precision, recall, nap


def load_gt_dets():
    # Load anno_list
    print('Loading anno_list.json ...')
    anno_list_json = 'data/anno_list.json'
    anno_list = json.load(open(anno_list_json, "r"))

    gt_dets = {}
    for anno in anno_list:
        if "test" not in anno['global_id']:
            continue

        global_id = anno['global_id']
        gt_dets[global_id] = {}
        for hoi in anno['hois']:
            hoi_id = hoi['id']
            gt_dets[global_id][hoi_id] = []
            for human_box_num, object_box_num in hoi['connections']:
                human_box = hoi['human_bboxes'][human_box_num]
                object_box = hoi['object_bboxes'][object_box_num]
                det = {
                    'human_box': human_box,
                    'object_box': object_box,
                }
                gt_dets[global_id][hoi_id].append(det)

    return gt_dets


class hico_eval():
    def __init__(self, model_path):
        self.out_dir = model_path
        self.nis = False
        self.annotations = load_gt_dets()
        print(len(self.annotations))
        self.hoi_list = json.load(open('data/hoi_list_new.json', 'r'))
        self.file_name_to_obj_cat = json.load(
            open('data/file_name_to_obj_cat.json', "r"))

        self.global_ids = self.annotations.keys()
        print(len(self.global_ids))
        self.hoi_id_to_num = json.load(
            open('data/hoi_id_to_num.json', "r"))
        self.rare_id_json = [key for key, item in self.hoi_id_to_num.items() if item['rare']]
        print(len(self.rare_id_json))
        self.pred_anno = {}

        self.verb_obj_index_to_hoi_index = io.load_json_object("data/verb_obj_index_to_hoi_index.json")

    def evaluation_default(self, predict_annot):
        if self.pred_anno == {}:
            pred_anno = {}
            for global_id in predict_annot:
                global_id_ = "HICO_test2015_" + str(global_id).zfill(8)
                # print(global_id)
                pred_anno[global_id_] = {}

                data = predict_annot[global_id]
                human_obj_boxes_scores_hoi_idx = data['human_obj_boxes_scores_hoi_idx']

                for data in human_obj_boxes_scores_hoi_idx:
                    obj_cat_index = int(data[-2])
                    verb_index = int(data[-1])
                    assert 0 <= obj_cat_index < 80
                    assert 0 <= verb_index < 117
                    # print(obj_cat_index)
                    # print("verb_index", verb_index)
                    # if str(verb_index) not in self.verb_obj_index_to_hoi_index:
                    #     continue
                    # if str(obj_cat_index) not in self.verb_obj_index_to_hoi_index[str(verb_index)]:
                    #     continue

                    hoi_index = self.verb_obj_index_to_hoi_index[str(verb_index)][str(obj_cat_index)]
                    assert 0 <= hoi_index < 600
                    hoi = str(hoi_index + 1).zfill(3)
                    # if hoi not in pred_anno[global_id_]:
                    #     pred_anno[global_id_][hoi] = np.empty([1, 10])
                    # data[8] = data[8] * data[10] * data[11]
                    # data[9] = data[9] * data[10] * data[11]
                    if hoi not in pred_anno[global_id_]:
                        pred_anno[global_id_][hoi] = data[:10][np.newaxis, :]
                    else:
                        pred_anno[global_id_][hoi] = np.concatenate([pred_anno[global_id_][hoi],
                                                                     data[:10][np.newaxis, :]])

                # print(pred_anno[global_id_].keys())
            self.pred_anno = pred_anno

        outputs = []
        for hoi in self.hoi_list:
            o = self.eval_hoi(hoi['id'], self.global_ids, self.annotations, self.pred_anno, self.out_dir)
            outputs.append(o)

        mAP = {
            'AP': {},
            'mAP': 0,
            'invalid': 0,
            'mAP_rare': 0,
            'mAP_non_rare': 0,
        }
        map_ = 0
        map_rare = 0
        map_non_rare = 0
        count = 0
        count_rare = 0
        count_non_rare = 0
        for ap, hoi_id in outputs:
            mAP['AP'][hoi_id] = ap
            if not np.isnan(ap):
                count += 1
                map_ += ap
                if hoi_id in self.rare_id_json:
                    count_rare += 1
                    map_rare += ap
                else:
                    count_non_rare += 1
                    map_non_rare += ap

        mAP['mAP'] = map_ / count
        print(mAP['mAP'])
        mAP['invalid'] = len(outputs) - count
        mAP['mAP_rare'] = map_rare / count_rare
        mAP['mAP_non_rare'] = map_non_rare / count_non_rare

        mAP_json = os.path.join(
            self.out_dir,
            f'mAP_default_{self.nis}.json')
        io.dump_json_object(mAP, mAP_json)

        # print(f'APs have been saved to {self.out_dir}')

    def evaluation_ko(self):

        outputs = []
        for hoi in self.hoi_list:
            o = self.eval_hoi(hoi['id'], self.global_ids, self.annotations,
                              self.pred_anno, mode="ko",
                              obj_cate=hoi['object_cat'])
            outputs.append(o)

        mAP = {
            'AP': {},
            'mAP': 0,
            'invalid': 0,
            'mAP_rare': 0,
            'mAP_non_rare': 0,
        }
        map_ = 0
        map_rare = 0
        map_non_rare = 0
        count = 0
        count_rare = 0
        count_non_rare = 0
        for ap, hoi_id in outputs:
            mAP['AP'][hoi_id] = ap
            if not np.isnan(ap):
                count += 1
                map_ += ap
                if hoi_id in self.rare_id_json:
                    count_rare += 1
                    map_rare += ap
                else:
                    count_non_rare += 1
                    map_non_rare += ap

        mAP['mAP'] = map_ / count
        mAP['invalid'] = len(outputs) - count
        print(count_rare, count_non_rare)
        mAP['mAP_rare'] = map_rare / count_rare
        mAP['mAP_non_rare'] = map_non_rare / count_non_rare

        mAP_json = os.path.join(
            self.out_dir,
            f'mAP_ko_{self.nis}.json')
        io.dump_json_object(mAP, mAP_json)

        # print(f'APs have been saved to {self.out_dir}')

    def eval_hoi(self, hoi_id, global_ids, gt_dets, pred_anno,
                 mode='default', obj_cate=None):
        # print(f'Evaluating hoi_id: {hoi_id} ...')
        y_true = []
        y_score = []
        det_id = []
        npos = 0
        for global_id in global_ids:

            if mode == 'ko':
                if global_id + ".jpg" not in self.file_name_to_obj_cat:
                    continue
                obj_cats = self.file_name_to_obj_cat[global_id + ".jpg"]
                if int(obj_cate) not in obj_cats:
                    continue

            if hoi_id in gt_dets[global_id]:
                candidate_gt_dets = gt_dets[global_id][hoi_id]
            else:
                candidate_gt_dets = []

            npos += len(candidate_gt_dets)

            if global_id not in pred_anno or hoi_id not in pred_anno[global_id]:
                hoi_dets = np.empty([0, 10])
            else:
                hoi_dets = pred_anno[global_id][hoi_id]

            num_dets = hoi_dets.shape[0]

            if self.nis:
                sorted_idx = [idx for idx, _ in sorted(
                    zip(range(num_dets), (hoi_dets[:, 8] * hoi_dets[:, 9]).tolist()),
                    key=lambda x: x[1],
                    reverse=True)]
            else:
                sorted_idx = [idx for idx, _ in sorted(
                    zip(range(num_dets), hoi_dets[:, 8].tolist()),
                    key=lambda x: x[1],
                    reverse=True)]
            for i in sorted_idx:
                pred_det = {
                    'human_box': hoi_dets[i, :4],
                    'object_box': hoi_dets[i, 4:8],
                    'score': hoi_dets[i, 9] * hoi_dets[i, 8] if self.nis else hoi_dets[i, 8]
                }
                is_match, candidate_gt_dets = match_hoi(pred_det, candidate_gt_dets)
                y_true.append(is_match)
                y_score.append(pred_det['score'])
                det_id.append((global_id, i))

        # Compute PR
        precision, recall, mark = compute_pr(y_true, y_score, npos)
        if not mark:
            ap = 0
        else:
            ap = compute_ap(precision, recall)
        # nprecision,nrecall,nap = compute_normalized_pr(y_true,y_score,npos)

        # Compute AP
        print(f'AP:{ap}')

        # Plot PR curve
        # plt.figure()
        # plt.step(recall,precision,color='b',alpha=0.2,where='post')
        # plt.fill_between(recall,precision,step='post',alpha=0.2,color='b')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.ylim([0.0, 1.05])
        # plt.xlim([0.0, 1.0])
        # plt.title('Precision-Recall curve: AP={0:0.4f}'.format(ap))
        # plt.savefig(
        #     os.path.join(out_dir,f'{hoi_id}_pr.png'),
        #     bbox_inches='tight')
        # plt.close()

        # Save AP data
        # ap_data = {
        #     'y_true': y_true,
        #     'y_score': y_score,
        #     'det_id': det_id,
        #     'npos': npos,
        #     'ap': ap,
        # }
        # np.save(
        #     os.path.join(out_dir, f'{hoi_id}_ap_data.npy'),
        #     ap_data)

        return (ap, hoi_id)


if __name__ == '__main__':
    dir = "PMFNet_Baseline_HICO/"
    num = "299999/"
    file = h5py.File("Outputs/exp/" + dir + num + "pred_hoi_dets_test.hdf5", "r")

    hoi_eval = hico_eval(f"Outputs/exp/{dir}{num}")

    hoi_eval.evaluation_default(file)
    hoi_eval.evaluation_ko()
    hoi_eval.nis = True
    hoi_eval.evaluation_default(file)
    hoi_eval.evaluation_ko()
