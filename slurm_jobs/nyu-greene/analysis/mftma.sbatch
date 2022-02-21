#!/bin/bash
#SBATCH --cpus-per-task=1 # one core
#SBATCH --nodes=1 # one node
#SBATCH --ntasks-per-node=1 # one task per node
#SBATCH --time=5000
#SBATCH --mem=16G
#SBATCH --job-name=normalize
#SBATCH --mail-type=END
#SBATCH --mail-use=bl3021@nyu.edu
#SBATCH --output ./outputs/out.%j.%a.%N.out -e ./errors/err.%j.%a.%N.err
#SBATCH --gres=gpu:v100:1

ARRAY_ID=$1
PARAMETER_FILE=$2

let FILE_LINE=(100*$ARRAY_ID + $SLURM_ARRAY_TASK_ID)

echo "Line ${FILE_LINE}"
# Get the relevant line from the parameters
PARAMETERS=$(sed "${FILE_LINE}q;d" ${PARAMETER_FILE})
echo ${PARAMETERS}

bash mftma.sh ${PARAMETERS}
