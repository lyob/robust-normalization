#!/bin/bash
PARAM_FILE="mftma.txt"

SAVE_FOLDER="/mnt/home/blyo1/ceph/syLab/robust-normalization/results/mftma/"
DATASETS=("cifar")
SEEDS=("17")
NORM_METHODS=("nn" "bn" "ln" "in" "gn" "lrnc" "lrns" "lrnb")
#NORM_METHODS=("nn" "bn" "ln" "in" "gn" "lrnc" "lrns")
#NORM_METHODS=("lrnb")
EPS=("0")

> ${PARAM_FILE}
for DATASET in ${DATASETS[@]};do
	for SEED in ${SEEDS[@]};do
		for NORM_METHOD in ${NORM_METHODS[@]};do
			for EP in ${EPS[@]};do
				echo "${SAVE_FOLDER} ${DATASET} ${SEED} ${NORM_METHOD} ${EP}" >> ${PARAM_FILE}
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
                sbatch --array=1-100 normalize.sbatch ${ARRAY_INDEX} ${PARAM_FILE}
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
	sbatch --array=1-${DIFF} mftma.sbatch ${ARRAY_INDEX} ${PARAM_FILE}
fi


