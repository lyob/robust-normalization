#!/bin/bash
#SBATCH -n 1 # one node
#SBATCH -p ccn # CCN partition
#SBATCH -c 1 # one core

#SBATCH -t 1:00:00
#SBATCH --mem=2G
#SBATCH --job-name=normalize
#SBATCH --output ./out/mnist/out.%j.%a.%N.out 
#SBATCH -e ./err/mnist/err.%j.%a.%N.err

ARRAY_ID=$1
PARAMETER_FILE=$2

let FILE_LINE=($ARRAY_ID + $SLURM_ARRAY_TASK_ID)

echo "Line ${FILE_LINE}"
# Get the relevant line from the parameters
PARAMETERS=$(sed "${FILE_LINE}q;d" ${PARAMETER_FILE})
# PARAMETERS=${PARAMETER_FILE}
echo ${PARAMETERS}

bash run_code_vone.sh ${PARAMETERS}
