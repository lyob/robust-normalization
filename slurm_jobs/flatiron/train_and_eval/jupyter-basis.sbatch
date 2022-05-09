#!/bin/bash

#SBATCH --job-name=jupyter_cpu
#SBATCH -p genx
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=48:00:00
#SBATCH --output=jupyter_cpu.out
#SBATCH --error=jupyter_cpu.err








port=$(shuf -i 10000-65500 -n 1)

/usr/bin/ssh -N -f -R $port:localhost:$port rusty1
/usr/bin/ssh -N -f -R $port:localhost:$port rusty2
/usr/bin/ssh -N -f -R $port:localhost:$port rustyamd1
/usr/bin/ssh -N -f -R $port:localhost:$port rustyamd2

cat<<EOF

Jupyter server is running on: $(hostname)
Job starts at: $(date)

Step 1 :

If you are working in NYU campus, please open an iTerm window, run command
ssh -L $port:localhost:$port $USER@rusty.flatironinstitute.org

If you are working off campus, you should already have ssh tunneling setup through HPC bastion host,
that you can directly login to flatiron with command

ssh $USER@flatiron

Please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@flatiron

Step 2:

Keep the iTerm windows in the previouse step open. Now open browser, find the line with

The Jupyter Notebook is running at: $(hostname)

the URL is something: http://localhost:${port}/?token=XXXXXXXX (see your token below)

you should be able to connect to jupyter notebook running remotely on flatiron compute node with above url

EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

# module load slurm gcc python3
source /mnt/home/blyo1/miniconda3/bin/activate pyenv38

# source /mnt/home/blyo1/.bashrc
# source /mnt/home/blyo1/.bash_profile

# conda activate pyenv38
# DIR = '/mnt/home/blyo1/ceph/syLab/robust-normalization/code/analysis/fisher-info/'
jupyter lab --no-browser --ip=0.0.0.0 --port=$port --notebook-dir=$(pwd)
# jupyter notebook --no-browser --port $port
