#!/bin/bash
#PBS -q SQUID
#PBS --group=cm9029
#PBS -b 1
#PBS -l cpunum_job=1
#PBS -l elapstim_req=00:30:00
#PBS -T openmpi
#PBS -v NQSV_MPI_MODULE=BaseGCC
#PBS -v OMP_NUM_THREADS=1
#PBS -v OPENBLAS_NUM_THREADS=1
#PBS -v MKL_NUM_THREADS=1
#PBS -v NUMEXPR_NUM_THREADS=1
set -euo pipefail

cd "$PBS_O_WORKDIR"
module load BaseGCC
source "$HOME/miniforge3/bin/activate" evac_env
python scripts/check_squid_mpi_env.py --expected-flavor openmpi

mpirun $NQSV_MPIOPTS -np 1 \
  python src/spinodal_phase5_mpi.py \
  --N 1024 \
  --deltas 1e-3 \
  --modes 0 \
  --M-total 128 \
  --block-size 32 \
  --benchmark-only \
  --benchmark-block-sizes 16,32,64,128 \
  --benchmark-steps 50 \
  --output-dir results/runs/phase5_B2_R12
