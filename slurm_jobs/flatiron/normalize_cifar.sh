#!/bin/bash

SAVE_FOLDER=$1
MODEL=$2
SEED=$3
MODE=$4
LEARNING_RATE=$5
WEIGHT_DECAY=$6
NORMALIZE=$7
EP=$8

cd /mnt/home/blyo1/ceph/syLab/robust-normalization/code/;
source /mnt/home/blyo1/miniconda3/etc/profile.d/conda.sh
conda activate pyenv36;

python3 cifar_normalize.py \
	--cluster "flatiron" \
	--save_folder ${SAVE_FOLDER} \
	--model_name ${MODEL} \
	--seed ${SEED} \
	--learning_rate ${LEARNING_RATE} \
	--weight_decay ${WEIGHT_DECAY} \
	--mode ${MODE} \
	--eps ${EP} \
	--normalize ${NORMALIZE} \
