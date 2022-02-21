#!/bin/bash

SAVE_FOLDER=$1
DATASET=$2
SEED=$3
NORM_METHOD=$4


module purge;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
# conda activate pyenv36;
conda activate /scratch/bl3021/.conda/envs/pyenv36

cd /scratch/bl3021/research/sy-lab/;
cd robust-normalization/code/analysis/

python -m parallel_mftma \
	--save_folder ${SAVE_FOLDER} \
	--dataset_name ${DATASET}
	--seed ${SEED} \
	--norm_method ${NORM_METHOD} \

