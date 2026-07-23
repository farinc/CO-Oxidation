#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe orte 24

module purge
unset OMPI_MCA_btl
export OMPI_MCA_btl=self,vader,tcp

module load petsc/3.25.3-real slepc/3.25.1-real openmpi/4.1.2

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# The $@ syntax in Bash forwards all parameters to sweep.mpi
#   qsub -N big_tile submit_sweep_job.sh --memkm-sites 12 --out big
mpirun -np 24 uv run python -m sweeps.mpi "$@"
