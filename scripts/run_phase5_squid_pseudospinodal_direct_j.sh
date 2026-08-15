#!/bin/bash
#PBS -q SQUID
#PBS --group=cm9029
#PBS -b 1
#PBS -l cpunum_job=1
#PBS -l elapstim_req=00:10:00
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
  "$PHASE5_PY" src/spinodal_phase5_pseudospinodal_mpi.py \
  --deltas 0.01,0.02,0.03,0.05,0.10 \
  --N 1024 \
  --M-total 64 \
  --block-size 32 \
  --kernel direct_J \
  --epsilon-fraction 0.05 \
  --T-fixed 2 \
  --preparation-width 0.02 \
  --preparation-steps 6 \
  --burn-steps-per-stage 8 \
  --output-dir results/runs/phase5_pseudospinodal_direct_J
