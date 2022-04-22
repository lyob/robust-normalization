#!/bin/bash

SAVE_FOLDER=$1
FRONTEND=$2
NORM_POSITION=$3
MODEL=$4
SEED=$5
MODE=$6
# LEARNING_RATE=$7
NORMALIZE=$7
EP=$8
WEIGHT_DECAY=$9
WIDTH_SCALE=${10}

cd /mnt/home/blyo1/ceph/syLab/robust-normalization/code/;
source /mnt/home/blyo1/miniconda3/etc/profile.d/conda.sh
conda activate pyenv36;

python3 wider_convnet_norm.py \
	--save_folder ${SAVE_FOLDER} \
	--frontend ${FRONTEND} \
	--norm_position ${NORM_POSITION} \
	--model_name ${MODEL} \
	--seed ${SEED} \
	--mode ${MODE} \
	--eps ${EP} \
	--normalize ${NORMALIZE} \
	--weight_decay ${WEIGHT_DECAY} \
	--width_scale ${WIDTH_SCALE} \
	# --learning_rate ${LEARNING_RATE} \
