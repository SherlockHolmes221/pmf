#!/usr/bin/env bash

model_name="e2e_pmf_net_R-50-FPN_1x"
EXP="PMF_Baseline_HICO"
NUM="299999"

CUDA_VISIBLE_DEVICES=1 python tools/test_net.py \
--dataset hico_test \
--cfg configs/baselines/$model_name.yaml \
--use_precomp_box \
--mlp_head_dim 256 \
--part_crop_size 5 \
--use_kps17 \
--net_name PMFNet_Baseline \
--load_ckpt Outputs/exp/${EXP}/ckpt/model_step${NUM}.pth \
--output_dir Outputs/exp/${EXP}/${NUM}


