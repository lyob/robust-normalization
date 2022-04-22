#!/bin/bash
#SBATCH -n 1 # one node
#SBATCH -p ccn -c 1

#SBATCH -t 1:00:00
#SBATCH --mem=8G
#SBATCH --job-name=resnorm
#SBATCH --output ./out/resnet/out.%j.%a.%N.out 
#SBATCH -e ./err/resnet/err.%j.%a.%N.err

ARRAY_ID=$1
PARAMETER_FILE=$2

let FILE_LINE=($ARRAY_ID + $SLURM_ARRAY_TASK_ID)

echo "Line ${FILE_LINE}"
# Get the relevant line from the parameters
PARAMETERS=$(sed "${FILE_LINE}q;d" ${PARAMETER_FILE})
# PARAMETERS=${PARAMETER_FILE}
echo ${PARAMETERS}

bash run_code_resnet.sh ${PARAMETERS}
