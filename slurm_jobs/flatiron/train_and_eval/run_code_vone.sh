#!/bin/bash

SAVE_FOLDER=$1
FRONTEND=$2
NORM_POSITION=$3
MODEL=$4
SEED=$5
MODE=$6
# LEARNING_RATE=$7
# WEIGHT_DECAY=$8
NORMALIZE=$7
EP=$8

cd /mnt/home/blyo1/ceph/syLab/robust-normalization/code/;
source /mnt/home/blyo1/miniconda3/etc/profile.d/conda.sh
conda activate pyenv36;

python3 vone_convnet_norm.py \
	--save_folder ${SAVE_FOLDER} \
	--frontend ${FRONTEND} \
	--norm_position ${NORM_POSITION} \
	--model_name ${MODEL} \
	--seed ${SEED} \
	--mode ${MODE} \
	--eps ${EP} \
	--normalize ${NORMALIZE} \
	# --learning_rate ${LEARNING_RATE} \
	# --weight_decay ${WEIGHT_DECAY} \
