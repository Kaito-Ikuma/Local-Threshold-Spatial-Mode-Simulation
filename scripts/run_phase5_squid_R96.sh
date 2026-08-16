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

PHASE5_R96_STAGE="${PHASE5_R96_STAGE:-benchmark}"
PHASE5_R=96
PHASE5_N=8192
PHASE5_ROOT=results/runs/phase5_R_sweep/R096
PHASE5_PLAN="$PHASE5_ROOT/fine_scan_plan.csv"

case "$PHASE5_R96_STAGE" in
  benchmark)
    mpirun ${NQSV_MPIOPTS} -np 1 \
      "$PHASE5_PY" src/spinodal_phase5_mpi.py \
      --analytic-references --B 2.0 --R 96 --N 8192 \
      --sigma-J 1.0 --sigma-phi 0.06 --phi-bar 0.0 --a 1.0 \
      --branch stay_to_evacuate --deltas 0.02 --modes 0,1 \
      --epsilon-fraction 0.05 --M-total 64 --block-size 64 \
      --kernel aggregated_exact --initialization prepared_metastable \
      --preparation-width 0.02 --preparation-steps 6 --burn-steps-per-stage 8 \
      --T-fixed 5 --fit-start 0 --fit-end 3 --qR-max-fit 0.35 \
      --stage benchmark --benchmark-only --benchmark-block-sizes 16,32,64 \
      --benchmark-steps 5 --task-id-prefix V4_R096_ --no-figures \
      --output-dir "$PHASE5_ROOT/benchmark"
    exit 0
    ;;
  coarse)
    PHASE5_DELTAS="${PHASE5_R96_DELTAS:-0.005,0.008,0.010,0.012,0.015,0.020,0.030}"
    PHASE5_DELTAS="${PHASE5_DELTAS//:/,}"
    PHASE5_M=2048
    PHASE5_OUTPUT="$PHASE5_ROOT/pseudospinodal_coarse${PHASE5_R96_TAG:-}"
    PHASE5_UNPERTURBED=yes
    ;;
  fine)
    PHASE5_DELTAS="$("$PHASE5_PY" src/spinodal_R_sweep_analysis.py print-plan-value --plan "$PHASE5_PLAN" --R 96)"
    PHASE5_M=8192
    PHASE5_OUTPUT="$PHASE5_ROOT/pseudospinodal_fine"
    PHASE5_UNPERTURBED=yes
    ;;
  response)
    PHASE5_TIME_TABLE="$PHASE5_ROOT/pseudospinodal_fine/analysis/phase5_pseudospinodal_time_dependence.csv"
    PHASE5_DELTAS="$("$PHASE5_PY" src/spinodal_R_sweep_analysis.py print-matched-deltas --time-table "$PHASE5_TIME_TABLE" --offsets 0.010 --T 50)"
    PHASE5_M=8192
    PHASE5_OUTPUT="$PHASE5_ROOT/response_matched"
    PHASE5_UNPERTURBED=no
    ;;
  *) echo "ERROR: PHASE5_R96_STAGE must be benchmark, coarse, fine, or response" >&2; exit 1 ;;
esac

PHASE5_RESPONSE_ARGS=(--epsilon-fraction 0.05 --epsilon-fractions 0.05)
PHASE5_MODES=0
if [ "$PHASE5_UNPERTURBED" = yes ]; then
  PHASE5_RESPONSE_ARGS=(--epsilon-fraction 0.0 --unperturbed)
fi

mpirun ${NQSV_MPIOPTS} -np 57 \
  "$PHASE5_PY" src/spinodal_phase5_mpi.py \
  --analytic-references --B 2.0 --R 96 --N 8192 \
  --sigma-J 1.0 --sigma-phi 0.06 --phi-bar 0.0 --a 1.0 \
  --branch stay_to_evacuate --deltas "$PHASE5_DELTAS" --modes "$PHASE5_MODES" \
  "${PHASE5_RESPONSE_ARGS[@]}" --M-total "$PHASE5_M" --block-size 64 \
  --kernel aggregated_exact --initialization prepared_metastable \
  --preparation-width 0.02 --preparation-steps 6 --burn-steps-per-stage 8 \
  --T-fixed 50 --fit-start 0 --fit-end 3 --qR-max-fit 0.35 \
  --track-survival --stage production --M-convergence-candidates "$PHASE5_M" \
  --bootstrap-replicates 1000 --task-id-prefix "V4_R096_${PHASE5_R96_STAGE}_" \
  --resume --no-figures --max-runtime-seconds 53400 --output-dir "$PHASE5_OUTPUT"

if [ "$PHASE5_UNPERTURBED" = yes ]; then
  "$PHASE5_PY" src/spinodal_phase5_followup_analysis.py pseudospinodal \
    --input-dir "$PHASE5_OUTPUT" --output-dir "$PHASE5_OUTPUT/analysis" \
    --criterion-probability 0.10 --primary-T 50 --observation-times 20,30,40,50
else
  "$PHASE5_PY" src/spinodal_phase5_followup_analysis.py time \
    --input-dir "$PHASE5_OUTPUT" --output-dir "$PHASE5_OUTPUT/analysis" \
    --gamma-eff-min-snr 5 --bootstrap-replicates 1000
  "$PHASE5_PY" src/spinodal_phase5_followup_analysis.py survival \
    --input-dir "$PHASE5_OUTPUT" --output-dir "$PHASE5_OUTPUT/analysis"
fi

if [ "$PHASE5_R96_STAGE" = coarse ]; then
  "$PHASE5_PY" src/spinodal_R_sweep_analysis.py plan-fine \
    --micro-root results/runs/phase5_R_sweep --R-list 96 \
    --criterion-probability 0.10 --primary-T 50 --max-step 0.002 \
    --extension-factor 1.5 --output "$PHASE5_PLAN"
  cat "$PHASE5_PLAN"
fi
