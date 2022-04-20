#!/bin/bash
PARAM_FILE="resnet.txt"
SAVE_FOLDER="results/"
MODEL_NAME=("resnet1") #mode of the model, can be "standard" or "convnet" or "convnet2" //// "vone_convnet-layer1_norm" or vone_convnet-layer1_norm-relu_last
MODES=("train") #mode can be 'train' or 'val'
WEIGHT_DECAYS=("0.0005")
# SEEDS=('1' '2' '3' '4' '5' '6' '7' '8' '9' '10')
# SEEDS=("1 2 3 4 5")
SEEDS=("1")
# NORMALIZES=('bn' 'gn' 'in' 'ln' 'lrnb' 'lrnc' 'lrns' 'nn')
NORMALIZES=('bn gn in ln lrnb lrnc lrns nn')

EPS=("1.0_2.0_4.0_6.0_8.0") # cifar

> ${PARAM_FILE}
for MODEL in ${MODEL_NAME[@]};do
	for SEED in ${SEEDS[@]};do
		for MODE in ${MODES[@]};do
			# for FRONTEND in ${FRONTENDS[@]};do
			# for NORM_POSITION in ${NORM_POSITIONS[@]};do
			for NORMALIZE in ${NORMALIZES[@]};do
				for EP in ${EPS[@]};do
					for WEIGHT_DECAY in ${WEIGHT_DECAYS[@]};do
						echo "${SAVE_FOLDER} ${MODEL} ${SEED} ${MODE} ${NORMALIZE} ${EP} ${WEIGHT_DECAY}" >> ${PARAM_FILE}
					done
				done
			done
		done
	done
done

# 2. Push all the Jobs through Job Arrays
LINE_COUNT=0
ARRAY_INDEX=0
while read line; do
	LINE_COUNT=$(expr ${LINE_COUNT} + 1)
done < ${PARAM_FILE}
sbatch --array=1-${LINE_COUNT} submit_jobs_resnet.sbatch ${ARRAY_INDEX} ${PARAM_FILE}