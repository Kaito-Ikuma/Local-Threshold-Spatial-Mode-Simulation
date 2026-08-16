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

PHASE5_TIME_TAG="${PHASE5_TIME_TAG:-_extension1}"
PHASE5_PLAN=results/runs/phase5_R_sweep/observation_time_extension_plan.csv
"$PHASE5_PY" src/spinodal_phase5_final_validation.py plan-time-extension \
  --r-sweep-dir results/runs/phase5_R_sweep --output "$PHASE5_PLAN"
cat "$PHASE5_PLAN"

for PHASE5_R in 6 12; do
  if ! PHASE5_DELTAS="$("$PHASE5_PY" src/spinodal_phase5_final_validation.py print-time-extension --plan "$PHASE5_PLAN" --R "$PHASE5_R")"; then
    echo "R=$PHASE5_R already bracketed for requested observation times"
    continue
  fi
  case "$PHASE5_R" in
    6) PHASE5_N=512 ;;
    12) PHASE5_N=1024 ;;
  esac
  printf -v PHASE5_R_LABEL 'R%03d' "$PHASE5_R"
  PHASE5_OUTPUT="results/runs/phase5_R_sweep/${PHASE5_R_LABEL}/time_extension${PHASE5_TIME_TAG}"

  mpirun ${NQSV_MPIOPTS} -np 57 \
    "$PHASE5_PY" src/spinodal_phase5_mpi.py \
    --analytic-references --B 2.0 --R "$PHASE5_R" --N "$PHASE5_N" \
    --sigma-J 1.0 --sigma-phi 0.06 --phi-bar 0.0 --a 1.0 \
    --branch stay_to_evacuate --deltas "$PHASE5_DELTAS" --modes 0 \
    --epsilon-fraction 0.0 --unperturbed --M-total 8192 --block-size 64 \
    --kernel aggregated_exact --initialization prepared_metastable \
    --preparation-width 0.02 --preparation-steps 6 --burn-steps-per-stage 8 \
    --T-fixed 50 --fit-start 0 --fit-end 3 --qR-max-fit 0.35 \
    --track-survival --stage pilot --M-convergence-candidates 8192 \
    --bootstrap-replicates 2 --task-id-prefix "V3_${PHASE5_R_LABEL}${PHASE5_TIME_TAG}_" \
    --resume --no-figures --max-runtime-seconds 53400 --output-dir "$PHASE5_OUTPUT"
done

"$PHASE5_PY" src/spinodal_phase5_final_validation.py plan-time-extension \
  --r-sweep-dir results/runs/phase5_R_sweep --output "$PHASE5_PLAN"
cat "$PHASE5_PLAN"
