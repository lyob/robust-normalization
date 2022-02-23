#!/bin/bash

SAVE_FOLDER=$1
MODEL=$2
SEED=$3
MODE=$4
LEARNING_RATE=$5
WEIGHT_DECAY=$6
NORMALIZE=$7
EP=$8

#module purge;
#module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;

conda activate /scratch/bl3021/.conda/envs/py36;
# conda activate py36
cd /scratch/bl3021/research/sy-lab/;
cd robust-normalization/code/

python cifar_normalize.py \
	--cluster "nyu" \
	--save_folder ${SAVE_FOLDER} \
	--model_name ${MODEL} \
	--seed ${SEED} \
	--learning_rate ${LEARNING_RATE} \
	--weight_decay ${WEIGHT_DECAY} \
	--mode ${MODE} \
	--eps ${EP} \
	--normalize ${NORMALIZE} \

