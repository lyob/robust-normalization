#!/bin/bash

SAVE_FOLDER=$1
MODEL=$2
SEED=$3
MODE=$4
LEARNING_RATE=$5
WEIGHT_DECAY=$6
NORMALIZE=$7
EP=$8

module purge;
module load cuda/11.3.1;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;

cd /scratch/bl3021/research/sy-lab/;
conda activate ./penv;
export PATH=./penv/bin:$PATH;

cd robust-normalization/code/

python3 mnist_normalize.py \
	--save_folder ${SAVE_FOLDER} \
	--model_name ${MODEL} \
	--seed ${SEED} \
	--learning_rate ${LEARNING_RATE} \
	--weight_decay ${WEIGHT_DECAY} \
	--mode ${MODE} \
	--eps ${EP} \
	--normalize ${NORMALIZE} \
