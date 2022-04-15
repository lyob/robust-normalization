#!/bin/bash
PARAM_FILE="mnist_param.txt"

SAVE_FOLDER="results/"

MODEL_NAME=("convnet4") #mode of the model, can be "standard" or "convnet" or "convnet2" //// "vone_convnet-layer1_norm" or vone_convnet-layer1_norm-relu_last
FRONTENDS=("learned_conv") # vone_filterbank or learned_conv or frozen_conv
NORM_POSITIONS=("both") # 1, 2, or both
SEEDS=("1 2 3 4 5")
MODES=("val") #mode can be 'train' or 'val'
NORMALIZES=("bn gn in ln lrnb lrnc lrns nn")
WEIGHT_DECAYS=("0.0 0.001 0.002 0.005 0.01 0.02 0.05 0.1")

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

# 2.a Iterate through parameter lines.
# For every 100 lines we should create a new job array
echo "2a Iterate through parameter lines"
LINE_COUNT=0
START_INDEX=1
ARRAY_INDEX=0
while read line; do
        LINE_COUNT=$(expr ${LINE_COUNT} + 1)

        if [ "$(expr ${LINE_COUNT} % 100)" = "0" ]
        then
                echo "New Array For Parameters from ${START_INDEX} to ${LINE_COUNT}"
                sbatch --array=1-100 submit_jobs_vone.sbatch ${ARRAY_INDEX} ${PARAM_FILE}
                START_INDEX=$(expr ${LINE_COUNT} + 1)
                ARRAY_INDEX=$(expr ${ARRAY_INDEX} + 1)
        fi

done < ${PARAM_FILE}

# 2.b Check to make sure that all parameters were tested
echo "2b Check to make sure that all parameters were tested"
if [ "${LINE_COUNT}" -ge "${START_INDEX}" ]
then
	echo ${LINE_COUNT}
	echo ${START_INDEX}
        DIFF=$(expr ${LINE_COUNT} - ${START_INDEX} + 1)
        echo "New Array For Parameters from ${START_INDEX} to ${LINE_COUNT}"
        #sbatch --array=1-${DIFF} --mem 16G normalize.sbatch ${ARRAY_INDEX} ${PARAM_FILE}
		sbatch --array=1-${DIFF} submit_jobs_vone.sbatch ${ARRAY_INDEX} ${PARAM_FILE}
fi

