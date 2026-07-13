#!/bin/bash
#$ -N kmc_co_ox
#$ -cwd
#$ -j y
#$ -o logs/
#$ -pe mpi 24

mkdir -p logs

module load gcc/10.3.0
module load openmpi/4.1.2

source .venv/bin/activate

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

mpirun -np 1 python -m sweeps.mpi "$@"
