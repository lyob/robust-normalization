#!/bin/bash

SINGULARITY_IMG=/om2/user/hangle/torchvision_art.simg
SAVE_FOLDER=$1
MODEL=$2
SEED=$3
MODE=$4
LEARNING_RATE=$5
WEIGHT_DECAY=$6
NORMALIZE=$7
EP=$8

if [ "$MODE" = "mft" ] || [ "$MODE" = "mft_ball" ]; then
    SINGULARITY_IMG=/om2/user/hangle/low_proj.simg
else
    SINGULARITY_IMG=/om2/user/hangle/torchvision_art.simg
fi

export SINGULARITY_BINDPATH="$SINGULARITY_IMG","$SAVE_FOLDER"
singularity exec --nv $SINGULARITY_IMG \
		python3 cifar_normalize.py \
		--save_folder ${SAVE_FOLDER} \
		--model_name ${MODEL} \
		--seed ${SEED} \
        --learning_rate ${LEARNING_RATE} \
		--weight_decay ${WEIGHT_DECAY} \
		--mode ${MODE} \
		--eps ${EP} \
		--normalize ${NORMALIZE} \
