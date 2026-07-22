#!/bin/bash
#$ -N kmc_co_ox
#$ -cwd
#$ -j y
#$ -pe orte 24

module purge
unset OMPI_MCA_btl
export OMPI_MCA_btl=self,vader,tcp

module load gcc/10.3.0
module load openmpi/4.1.2

source .venv/bin/activate

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

mpirun -np 24 python -m sweeps.mpi --no-coexistence 
