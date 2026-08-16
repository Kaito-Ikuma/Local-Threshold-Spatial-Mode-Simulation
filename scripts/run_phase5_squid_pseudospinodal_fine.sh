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

cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is not set}"
source scripts/phase5_squid_env.sh
source scripts/phase5_squid_preflight.sh

# This primary scan is epsilon=0.  Keep burn=8 until Phase5-B determines the
# preparation protocol; changing burn requires a new output directory.
mpirun ${NQSV_MPIOPTS} -np 57 \
  "$PHASE5_PY" src/spinodal_phase5_mpi.py \
  --analytic-references \
  --B 2.0 --R 12 --sigma-J 1.0 --sigma-phi 0.06 --phi-bar 0.0 --a 1.0 \
  --branch stay_to_evacuate \
  --N 1024 \
  --deltas 0.058,0.060,0.062,0.064,0.066,0.068,0.070 \
  --modes 0 \
  --epsilon-fraction 0.0 \
  --unperturbed \
  --M-total 8192 --block-size 64 \
  --kernel aggregated_exact \
  --initialization prepared_metastable \
  --preparation-width 0.02 --preparation-steps 6 --burn-steps-per-stage 8 \
  --T-fixed 50 --fit-start 0 --fit-end 3 \
  --track-survival \
  --stage pilot \
  --M-convergence-candidates 8192 \
  --bootstrap-replicates 2 \
  --resume --no-figures --max-runtime-seconds 53400 \
  --output-dir results/runs/phase5_B2_R12_pseudospinodal_fine

"$PHASE5_PY" src/spinodal_phase5_followup_analysis.py pseudospinodal \
  --input-dir results/runs/phase5_B2_R12_pseudospinodal_fine \
  --output-dir results/runs/phase5_B2_R12_followup \
  --criterion-probability 0.10 \
  --primary-T 50 \
  --observation-times 10,20,30,40,50
