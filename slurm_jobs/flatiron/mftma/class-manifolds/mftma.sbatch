#!/bin/bash
#SBATCH -n 1 # one core
#SBATCH -p gpu --gpus=1 -c 1 --gpus-per-task=1 --constraint=v100
#SBATCH -t 2:00:00
#SBATCH --mem=16G
#SBATCH --job-name=normalize
#SBATCH --output ./outputs/out.%j.%a.%N.out -e ./errors/err.%j.%a.%N.err

ARRAY_ID=$1
PARAMETER_FILE=$2

let FILE_LINE=(100*$ARRAY_ID + $SLURM_ARRAY_TASK_ID)

echo "Line ${FILE_LINE}"
# Get the relevant line from the parameters
PARAMETERS=$(sed "${FILE_LINE}q;d" ${PARAMETER_FILE})
echo ${PARAMETERS}

bash mftma.sh ${PARAMETERS}
