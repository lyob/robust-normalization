#!/bin/bash
#
#SBATCH -n 1 # one core
#SBATCH -N 1 # one node
#SBATCH -t 5000
#SBATCH --mem=16G
#SBATCH --gres=gpu:1 --constraint=high-capacity

ARRAY_ID=$1
PARAMETER_FILE=$2

let FILE_LINE=(100*$ARRAY_ID + $SLURM_ARRAY_TASK_ID)

echo "Line ${FILE_LINE}"
# Get the relevant line from the parameters
PARAMETERS=$(sed "${FILE_LINE}q;d" ${PARAMETER_FILE})
echo ${PARAMETERS}


bash normalize.sh ${PARAMETERS}
