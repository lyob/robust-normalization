#!/bin/bash

#SBATCH --job-name=jupyter_cpu
#SBATCH -p genx
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --output=jupyter_cpu.out
#SBATCH --error=jupyter_cpu.err
#SBATCH --wait


sleep 1d