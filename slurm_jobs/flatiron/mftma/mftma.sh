#!/bin/bash

SAVE_FOLDER=$1
DATASET=$2
SEED=$3
NORM_METHOD=$4

cd /mnt/home/blyo1/ceph/syLab/robust-normalization/code/analysis/;
source /mnt/home/blyo1/miniconda3/etc/profile.d/conda.sh
conda activate pyenv36

python -m parallel_mftma \
	--save_folder ${SAVE_FOLDER} \
	--dataset_name ${DATASET} \
	--seed ${SEED} \
	--norm_method ${NORM_METHOD} 

