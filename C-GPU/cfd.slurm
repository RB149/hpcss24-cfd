#!/bin/bash

# Slurm job options (name, compute nodes, job time)
#SBATCH --job-name=sharpen
#SBATCH --time=00:01:00
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
##SBATCH --qos=gpu
#SBATCH --qos=reservation
#SBATCH --reservation=tc069_1525356

module --silent load gcc
module --silent load nvidia/nvhpc

# Launch the parallel job
srun --unbuffered ./cfd 100 5000

