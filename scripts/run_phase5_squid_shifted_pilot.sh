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

mpirun ${NQSV_MPIOPTS} -np 57 \
  "$PHASE5_PY" src/spinodal_phase5_mpi.py \
  --analytic-references \
  --B 2.0 \
  --R 12 \
  --sigma-J 1.0 \
  --sigma-phi 0.06 \
  --phi-bar 0.0 \
  --a 1.0 \
  --branch stay_to_evacuate \
  --N 1024 \
  --deltas 0.065,0.070,0.080 \
  --modes 0,1,2,3,4,5,6 \
  --epsilon-fraction 0.05 \
  --epsilon-fractions 0.025,0.05,0.10 \
  --M-total 8192 \
  --block-size 64 \
  --kernel aggregated_exact \
  --initialization prepared_metastable \
  --preparation-width 0.02 \
  --preparation-steps 6 \
  --burn-steps-per-stage 8 \
  --T-fixed 50 \
  --fit-start 0 \
  --fit-end 3 \
  --qR-max-fit 0.45 \
  --stage pilot \
  --M-convergence-candidates 1024,2048,4096,8192 \
  --resume \
  --no-figures \
  --max-runtime-seconds 53400 \
  --output-dir results/runs/phase5_B2_R12_shifted_dispersion_M8192