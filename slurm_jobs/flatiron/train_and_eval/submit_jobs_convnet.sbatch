#!/bin/bash
#SBATCH -n 1 # one node
#SBATCH -p ccn # CCN partition
#SBATCH -c 1 # one core

#SBATCH -t 2:00:00
#SBATCH --mem=3G
#SBATCH --job-name=normalize
#SBATCH --output ./out/convnet/out.%j.%a.%N.out 
#SBATCH -e ./err/convnet/err.%j.%a.%N.err

ARRAY_ID=$1
PARAMETER_FILE=$2

let FILE_LINE=($ARRAY_ID + $SLURM_ARRAY_TASK_ID)

echo "Line ${FILE_LINE}"
# Get the relevant line from the parameters
PARAMETERS=$(sed "${FILE_LINE}q;d" ${PARAMETER_FILE})
# PARAMETERS=${PARAMETER_FILE}
echo ${PARAMETERS}

bash run_code_convnet.sh ${PARAMETERS}
