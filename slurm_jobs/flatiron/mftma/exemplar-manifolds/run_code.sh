#!/bin/bash

SAVE_FOLDER=$1
SEED=$2
NORM_METHOD=$3
EP=$4


cd /mnt/home/blyo1/ceph/syLab/robust-normalization/code/analysis/exemplar-manifolds/;
source /mnt/home/blyo1/miniconda3/etc/profile.d/conda.sh
conda activate pyenv36

echo "EP is ${EP}"
echo "NORM_METHOD is ${NORM_METHOD}"
echo "SEED is ${SEED}"

python exemplar_analysis_parallel.py \
	--save_folder ${SAVE_FOLDER} \
	--seed ${SEED} \
	--eps ${EP} \
	--norm_method ${NORM_METHOD}
