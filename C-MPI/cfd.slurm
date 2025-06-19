#!/bin/bash

# Slurm job options (name, compute nodes, job time)
#SBATCH --job-name=cfd
#SBATCH --time=0:10:0
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --partition=standard
#SBATCH --qos=reservation
#SBATCH --reservation=tc069_1525341


export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Load the default HPE MPI environment
module load mpt
module load intel-20.4/compilers

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Set the number of threads to 1
#   This prevents any threaded system libraries from automatically
#   using threading.
export OMP_NUM_THREADS=1

# Launch the parallel job
#   Using 4 MPI processes on 1 node
#   srun picks up the distribution from the sbatch options
srun --cpu-bind=cores ./cfd 4 5000

