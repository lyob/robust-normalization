#!/bin/bash
PARAM_FILE="mnist_param.txt"

SAVE_FOLDER="./"
MODEL_NAME=("convnet") #mode of the model, can be "standard" or "convnet"  //// "vone_convnet-layer1_norm" or vone_convnet-layer1_norm-relu_last
FRONTENDS=("vone_filterbank") # vone_filterbank or learned_conv
# NORM_POSITIONS=("2 both") # 1, 2, or both
NORM_POSITIONS=("both") # 1, 2, or both
# SEEDS=("1 2 3 4 5")
SEEDS=("4")
MODES=("val") #mode can be 'train', 'val', 'extract', 'mft', 'pairwise', 'meanvar'
# LEARNING_RATES=("0.01")
# WEIGHT_DECAYS=("0.0005")
# NORMALIZES=("nn" "bn" "ln" "in" "gn" "lrnc" "lrns" "lrnb")
# NORMALIZES=("bn gn in ln lrnb lrnc")
# NORMALIZES=("nn" "bn" "ln" "in" "gn" "lrnc" "lrns")
NORMALIZES=("in")
EPS=("0.01_0.03_0.05_0.07_0.1_0.15_0.2")
#EPS=("1.0")

> ${PARAM_FILE}
for MODEL in ${MODEL_NAME[@]};do
	for SEED in ${SEEDS[@]};do
		for MODE in ${MODES[@]};do
			for FRONTEND in ${FRONTENDS[@]};do
				for NORM_POSITION in ${NORM_POSITIONS[@]};do
					for NORMALIZE in ${NORMALIZES[@]};do
						for EP in ${EPS[@]};do
							echo "${SAVE_FOLDER} ${FRONTEND} ${NORM_POSITION} ${MODEL} ${SEED} ${MODE} ${NORMALIZE} ${EP}" >> ${PARAM_FILE}
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

