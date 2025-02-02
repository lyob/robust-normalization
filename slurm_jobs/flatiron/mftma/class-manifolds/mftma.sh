#!/bin/bash

SAVE_FOLDER=$1
DATASET=$2
SEED=$3
NORM_METHOD=$4
EP=$5


cd /mnt/home/blyo1/ceph/syLab/robust-normalization/code/analysis/class-manifolds/;
source /mnt/home/blyo1/miniconda3/etc/profile.d/conda.sh
conda activate pyenv36

echo "EP is ${EP}"
echo "NORM_METHOD is ${NORM_METHOD}"
echo "SEED is ${SEED}"
echo "DATASET is ${DATASET}"

python -m parallel_mftma \
	--save_folder ${SAVE_FOLDER} \
	--dataset_name ${DATASET} \
	--seed ${SEED} \
	--eps ${EP} \
	--norm_method ${NORM_METHOD}
