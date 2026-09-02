#!/bin/bash
#$ -cwd
#$ -j y

module purge
unset OMPI_MCA_btl
export OMPI_MCA_btl=self,vader,tcp

module load petsc/3.25.3-real slepc/3.25.1-real openmpi/4.1.2

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source .venv/bin/activate

# The $@ syntax in Bash forwards all parameters to sweep.mpi. $NSLOTS is set by SGE to
# whatever count follows -pe orte at submission time, so the rank count always matches
# what was actually granted instead of a number baked into this script.
#   qsub -N big_tile -pe orte 24 submit_sweep_job.sh --memkm-sites 12 --out big
mpirun -np "$NSLOTS" python -m sweeps.mpi "$@"
