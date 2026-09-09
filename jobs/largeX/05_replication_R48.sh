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

R24_SUMMARY="${LARGEX_R24_SUMMARY:-results/runs/phase5_largeX/R024/production/primary/largeX_analysis/largeX_validation_summary.json}"
test -r "$R24_SUMMARY"
"$PHASE5_PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["reached_X_0p5"], "R24 X>=0.5 quality gate has not passed"' "$R24_SUMMARY"
: "${LARGEX_R48_DELTA_PS:?Measure/provide operational R48,N8192 delta_ps(Tobs=50) first}"
: "${LARGEX_R48_OFFSET:?Provide one R48 matched offset selected after R24 success}"
: "${LARGEX_EPSILON:?Provide the validated epsilon fraction}"
: "${LARGEX_FIT_END:?Provide the pre-specified R48 fit end}"
: "${LARGEX_RANKS:?Select the fastest safe rank count from the R48 benchmark}"
DELTA_G="$($PHASE5_PY -c 'import sys; print(f"{float(sys.argv[1])+float(sys.argv[2]):.12g}")' "$LARGEX_R48_DELTA_PS" "$LARGEX_R48_OFFSET")"

mpirun ${NQSV_MPIOPTS:-} -np "$LARGEX_RANKS" \
  "$PHASE5_PY" src/spinodal_phase5_mpi.py \
  --analytic-references --B 2 --R 48 --N 8192 --deltas "$DELTA_G" --auto-q-grid \
  --calibration-qR-max 0.15 --validation-qR-max 0.35 \
  --epsilon-fraction "$LARGEX_EPSILON" --M-total "${LARGEX_M:-8192}" --block-size 16 \
  --kernel aggregated_exact --T-fixed 50 --fit-end "$LARGEX_FIT_END" \
  --survivor-cohort-end "$LARGEX_FIT_END" --track-survival \
  --epsilon-linearity-validated \
  --rng-coupling-mode common_modes --harmonic-orders 3,5 \
  --fit-protocol q0_tau_survival_frozen_v1 \
  --campaign-version 2026.09.09-phase5-largeX-v1 \
  --bootstrap-replicates 1000 --stage production --resume --no-figures \
  --max-runtime-seconds 53400 --task-id-prefix largeX_R048_replication_ \
  --output-dir results/runs/phase5_largeX/R048/production
