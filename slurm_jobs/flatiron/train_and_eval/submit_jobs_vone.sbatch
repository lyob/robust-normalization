#!/bin/bash
#SBATCH -n 1 # one node
#SBATCH -p genx # a small job with...
#SBATCH -c 1 # ...one core
#SBATCH -t 10:00:00
#SBATCH --mem=16G
#SBATCH --job-name=normalize
#SBATCH --output ./out/mnist/out.%j.%a.%N.out -e ./err/mnist/err.%j.%a.%N.err

ARRAY_ID=$1
PARAMETER_FILE=$2

let FILE_LINE=(100*$ARRAY_ID + $SLURM_ARRAY_TASK_ID)

echo "Line ${FILE_LINE}"
# Get the relevant line from the parameters
PARAMETERS=$(sed "${FILE_LINE}q;d" ${PARAMETER_FILE})
echo ${PARAMETERS}

bash run_code_vone.sh ${PARAMETERS}
