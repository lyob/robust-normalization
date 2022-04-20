#!/bin/bash

SAVE_FOLDER=$1
MODEL=$2
SEED=$3
MODE=$4
NORMALIZE=$5
EP=$6
WEIGHT_DECAY=$7

cd /mnt/home/blyo1/ceph/syLab/robust-normalization/code/;
source /mnt/home/blyo1/miniconda3/etc/profile.d/conda.sh
conda activate pyenv36;

python3 resnet_normalize.py \
	--cluster "flatiron" \
	--save_folder ${SAVE_FOLDER} \
	--model_name ${MODEL} \
	--seed ${SEED} \
	--mode ${MODE} \
	--normalize ${NORMALIZE} \
	--eps ${EP} \
	--weight_decay ${WEIGHT_DECAY} \
