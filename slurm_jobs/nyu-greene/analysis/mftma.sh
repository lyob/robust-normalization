#!/bin/bash

SAVE_FOLDER=$1
DATASET=$2
SEED=$3
NORM_METHOD=$4


module purge;
bash /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
cd /scratch/bl3021/research/sy-lab/;
conda activate pyenv36;

cd robust-normalization/code/analysis/

python3 parallel_mftma.py \
	--save_folder ${SAVE_FOLDER} \
	--dataset_name ${DATASET}
	--seed ${SEED} \
	--norm_method ${NORM_METHOD} \

