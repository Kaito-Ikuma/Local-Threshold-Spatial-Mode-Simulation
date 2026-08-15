#!/bin/bash
#PBS -q SQUID
#PBS --group=cm9029
#PBS -b 1
#PBS -l cpunum_job=1
#PBS -l elapstim_req=00:30:00
#PBS -T intmpi
#PBS -v OMP_NUM_THREADS=1
#PBS -v OPENBLAS_NUM_THREADS=1
#PBS -v MKL_NUM_THREADS=1
#PBS -v NUMEXPR_NUM_THREADS=1
set -euo pipefail

cd "$PBS_O_WORKDIR"
source scripts/phase5_squid_env.sh
source scripts/phase5_squid_preflight.sh

mpirun ${NQSV_MPIOPTS} -np 1 \
  "$PHASE5_PY" src/spinodal_phase5_mpi.py \
  --N 1024 \
  --deltas 1e-3 \
  --modes 0 \
  --M-total 128 \
  --block-size 32 \
  --benchmark-only \
  --benchmark-block-sizes 16,32,64,128 \
  --benchmark-steps 50 \
  --no-figures \
  --output-dir results/runs/phase5_B2_R12
