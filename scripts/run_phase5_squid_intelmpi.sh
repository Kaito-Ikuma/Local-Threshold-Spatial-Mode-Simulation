#!/bin/bash
#PBS -q SQUID
#PBS --group=cm9029
#PBS -b 1
#PBS -l cpunum_job=76
#PBS -l elapstim_req=15:00:00
#PBS -T intmpi
#PBS -v OMP_NUM_THREADS=1
#PBS -v OPENBLAS_NUM_THREADS=1
#PBS -v MKL_NUM_THREADS=1
#PBS -v NUMEXPR_NUM_THREADS=1
set -euo pipefail

cd "$PBS_O_WORKDIR"
source scripts/phase5_squid_env.sh
source scripts/phase5_squid_preflight.sh

mpirun ${NQSV_MPIOPTS} -np 76 \
  "$PHASE5_PY" src/spinodal_phase5_mpi.py \
  --phase0-dir results/runs/phase0_B2_R12 \
  --phase12-dir results/runs/phase12_B2_R12 \
  --phase34-dir results/runs/phase34_B2_R12 \
  --N 1024 \
  --deltas 1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5 \
  --modes 0,1,2,3,4 \
  --epsilon-fraction 0.05 \
  --M-total 2048 \
  --block-size 32 \
  --kernel aggregated_exact \
  --initialization prepared_metastable \
  --stage production \
  --resume \
  --no-figures \
  --max-runtime-seconds 53400 \
  --output-dir results/runs/phase5_B2_R12
