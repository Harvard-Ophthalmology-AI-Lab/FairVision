#!/bin/bash
DATASET_DIR=/shared/ssd_30T/luoy/project/python/datasets/harvard/FairVision/
RESULT_DIR=.
MODEL_TYPE=( ViT-B ) # Options: efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext
MODALITY_TYPE='slo_fundus' # Options: 'oct_bscans_3d' | 'slo_fundus'
ATTRIBUTE_TYPE=( race gender hispanic ) # Options: race | gender | hispanic

# Args (slo_fundus)
VIT_WEIGHTS=imagenet
BATCH_SIZE=64
BLR=5e-4
WD=0.01
LD=0.55
DP=0.1
EXP_NAME=${VIT_WEIGHTS}_slo_fundus

SCALE_COEF=0.5
SCALE_BLUR=.1
SCALE_TEMP=1.
SCALE_ASYMPTOTE=1

PERF_FILE=AMD_${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}.csv
python scripts/train_amd_fair_fis.py \
        --epochs 50 \
        --batch_size ${BATCH_SIZE} \
        --blr ${BLR} \
        --min_lr 1e-6 \
        --warmup_epochs 5 \
        --weight_decay ${WD} \
        --layer_decay ${LD} \
        --drop_path ${DP} \
        --data_dir ${DATASET_DIR}/AMD/ \
        --result_dir ${RESULT_DIR}/results/AMD_${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE} \
        --model_type ${MODEL_TYPE} \
        --modality_types ${MODALITY_TYPE} \
        --perf_file ${PERF_FILE} \
        --vit_weights ${VIT_WEIGHTS} \
        --attribute_type ${ATTRIBUTE_TYPE} \
        --fair_scaling_coef ${SCALE_COEF} \
        --fair_scaling_sinkhorn_blur ${SCALE_BLUR} \
        --fair_scaling_temperature ${SCALE_TEMP} \
        --fair_right_asymptote ${SCALE_ASYMPTOTE}
