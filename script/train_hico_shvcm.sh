#!/usr/bin/env bash
set -x

model_name="e2e_pmf_net_dR-50-FPN_1x"
EXP="SH_VCM"
mkdir -p ./Outputs/e2e_pmfnet_R-50-FPN_1x/${EXP}

CUDA_VISIBLE_DEVICES=1 python -u tools/train_net_step.py \
--dataset hico_trainval \
--cfg /home/xian/Documents/code/PMF/configs/baselines/e2e_pmf_net_R-50-FPN_1x.yaml \
--use_precomp_box \
--hico_use_union_feat \
--lr 4e-2 \
--bs 2 \
--nw 4 \
--disp_interval 200 \
--freeze_at 5 \
--mlp_head_dim 256 \
--part_crop_size 5 \
--use_kps17 \
--max_iter 300000 \
--solver_steps 0 150000 250000 \
--net_name SH_VCM \
--triplets_num_per_im 32 \
--expID ${EXP} \
--copy \
--load_ckpt data/pretrained_model/e2e_faster_rcnn_R-50-FPN_1x_step119999.pth \
--use_tfboard |tee ./Outputs/e2e_pmfnet_R-50-FPN_1x/${EXP}/train-log.out
