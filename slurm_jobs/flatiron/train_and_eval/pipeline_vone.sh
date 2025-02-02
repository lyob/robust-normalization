#!/bin/bash
PARAM_FILE="mnist_param_2.txt"
SAVE_FOLDER="results/"
MODEL_NAME=("convnet4") #mode of the model, can be "standard" or "convnet" or "convnet2" //// "vone_convnet-layer1_norm" or vone_convnet-layer1_norm-relu_last
FRONTENDS=("learned_conv") # vone_filterbank or learned_conv or frozen_conv
NORM_POSITIONS=("both") # 1, 2, or both
MODES=("val") #mode can be 'train' or 'val'
# WEIGHT_DECAYS=("0.0 0.0001 0.0002 0.0003 0.0004 0.0005 0.001")
# WEIGHT_DECAYS=("0.002 0.005 0.01 0.02 0.05")
WEIGHT_DECAYS=("0.0 0.0005 0.005")
# SEEDS=('1' '2' '3' '4' '5' '6' '7' '8' '9' '10')
# SEEDS=("1 2 3 4 5")
SEEDS=("3")
NORMALIZES=('bn' 'gn' 'in' 'ln' 'lrnb' 'lrnc' 'lrns' 'nn')
# NORMALIZES=('in ln')

EPS=("0.01_0.03_0.05_0.07_0.1_0.15_0.2")

> ${PARAM_FILE}
for MODEL in ${MODEL_NAME[@]};do
	for SEED in ${SEEDS[@]};do
		for MODE in ${MODES[@]};do
			for FRONTEND in ${FRONTENDS[@]};do
				for NORM_POSITION in ${NORM_POSITIONS[@]};do
					for NORMALIZE in ${NORMALIZES[@]};do
						for EP in ${EPS[@]};do
							for WEIGHT_DECAY in ${WEIGHT_DECAYS[@]};do
								echo "${SAVE_FOLDER} ${FRONTEND} ${NORM_POSITION} ${MODEL} ${SEED} ${MODE} ${NORMALIZE} ${EP} ${WEIGHT_DECAY}" >> ${PARAM_FILE}
							done
						done
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
sbatch --array=1-${LINE_COUNT} submit_jobs_vone.sbatch ${ARRAY_INDEX} ${PARAM_FILE}