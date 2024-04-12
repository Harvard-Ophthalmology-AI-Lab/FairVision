#!/bin/bash
DATASET_DIR=/shared/ssd_30T/luoy/project/python/datasets/harvard/FairVision/
RESULT_DIR=.

LR=5e-5
NUM_EPOCH=50 
BATCH_SIZE=2
MODALITY_TYPE='oct_bscans_3d'
ATTRIBUTE_TYPE=( race gender hispanic ) # race|gender|hispanic
EXPR=train_predictor_amd

MODEL_TYPE=resnet18
CONV_TYPE=Conv3d

SCALE_COEF=.5
SCALE_BLUR=.1
SCALE_TEMP=10.
SCALE_BZ=6


PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_3D_fis.csv
python ./scripts/train_amd_3d_fair_fis.py \
		--data_dir ${DATASET_DIR}/AMD/ \
		--result_dir ${RESULT_DIR}/results_3D/amd_fis_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_${MODEL_TYPE}_${CONV_TYPE}_3D_lr${LR}_temp${SCALE_TEMP} \
		--model_type ${MODEL_TYPE} \
		--image_size 200 \
		--loss_type ${LOSS_TYPE} \
		--lr ${LR} --weight-decay 0. --momentum 0.1 \
		--batch_size ${BATCH_SIZE} \
		--task ${TASK} \
		--epochs ${NUM_EPOCH} \
		--modality_types ${MODALITY_TYPE} \
		--perf_file ${PERF_FILE} \
		--attribute_type ${ATTRIBUTE_TYPE} \
        --conv_type ${CONV_TYPE} \
		--fair_scaling_coef ${SCALE_COEF} \
		--fair_scaling_sinkhorn_blur ${SCALE_BLUR} \
		--fair_scaling_temperature ${SCALE_TEMP} \
		--fair_scaling_batchsize ${SCALE_BZ}