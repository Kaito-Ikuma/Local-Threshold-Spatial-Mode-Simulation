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

PHASE5_RESPONSE_KIND="${PHASE5_RESPONSE_KIND:-both}"
case "$PHASE5_RESPONSE_KIND" in
  fixed|matched|both) ;;
  *) echo "ERROR: PHASE5_RESPONSE_KIND must be fixed, matched, or both" >&2; exit 1 ;;
esac

run_response_case() {
  local phase5_coordinate="$1"
  local phase5_deltas="$2"
  local phase5_output="$3"

  mpirun ${NQSV_MPIOPTS} -np 57 \
    "$PHASE5_PY" src/spinodal_phase5_mpi.py \
    --analytic-references \
    --B 2.0 --R "$PHASE5_R" --N "$PHASE5_N" \
    --sigma-J 1.0 --sigma-phi 0.06 --phi-bar 0.0 --a 1.0 \
    --branch stay_to_evacuate \
    --deltas "$phase5_deltas" --modes 0,1,4 \
    --epsilon-fraction 0.05 --epsilon-fractions 0.05 \
    --M-total 8192 --block-size 64 \
    --kernel aggregated_exact --initialization prepared_metastable \
    --preparation-width 0.02 --preparation-steps 6 --burn-steps-per-stage 8 \
    --T-fixed 50 --fit-start 0 --fit-end 3 --qR-max-fit 0.35 \
    --track-survival --stage production --M-convergence-candidates 8192 \
    --task-id-prefix "${PHASE5_R_LABEL}_${phase5_coordinate}_" \
    --resume --no-figures --max-runtime-seconds 53400 \
    --output-dir "$phase5_output"

  "$PHASE5_PY" src/spinodal_phase5_followup_analysis.py time \
    --input-dir "$phase5_output" --output-dir "$phase5_output/analysis" \
    --gamma-eff-min-snr 5 --bootstrap-replicates 1000
  "$PHASE5_PY" src/spinodal_phase5_followup_analysis.py survival \
    --input-dir "$phase5_output" --output-dir "$phase5_output/analysis"
}

for PHASE5_R in 6 12 24 48; do
  case "$PHASE5_R" in
    6) PHASE5_N=512 ;;
    12) PHASE5_N=1024 ;;
    24) PHASE5_N=2048 ;;
    48) PHASE5_N=4096 ;;
    *) echo "ERROR: unsupported R=$PHASE5_R" >&2; exit 1 ;;
  esac
  printf -v PHASE5_R_LABEL 'R%03d' "$PHASE5_R"
  PHASE5_TIME_TABLE="results/runs/phase5_R_sweep/${PHASE5_R_LABEL}/pseudospinodal_fine/analysis/phase5_pseudospinodal_time_dependence.csv"
  PHASE5_MATCHED_DELTAS="$("$PHASE5_PY" src/spinodal_R_sweep_analysis.py print-matched-deltas --time-table "$PHASE5_TIME_TABLE" --offsets 0.005,0.010,0.020 --T 50)"

  if [ "$PHASE5_RESPONSE_KIND" = fixed ] || [ "$PHASE5_RESPONSE_KIND" = both ]; then
    run_response_case fixed 0.08,0.10,0.12 \
      "results/runs/phase5_R_sweep/${PHASE5_R_LABEL}/response_fixed_delta"
  fi
  if [ "$PHASE5_RESPONSE_KIND" = matched ] || [ "$PHASE5_RESPONSE_KIND" = both ]; then
    run_response_case matched "$PHASE5_MATCHED_DELTAS" \
      "results/runs/phase5_R_sweep/${PHASE5_R_LABEL}/response_matched"
  fi
done
