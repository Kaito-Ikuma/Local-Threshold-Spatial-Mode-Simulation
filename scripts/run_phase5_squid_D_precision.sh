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

PHASE5_M="${PHASE5_D_M_TOTAL:-${PHASE5_D_M:-32768}}"
PHASE5_R_LIST="${PHASE5_D_R_LIST:-12,24,48}"
PHASE5_R_LIST="${PHASE5_R_LIST//,/ }"
PHASE5_R_LIST="${PHASE5_R_LIST//:/ }"
case "$PHASE5_M" in
  32768) PHASE5_M_CANDIDATES=8192,16384,32768 ;;
  65536) PHASE5_M_CANDIDATES=8192,16384,32768,65536 ;;
  *) echo "ERROR: PHASE5_D_M_TOTAL must be 32768 or 65536" >&2; exit 1 ;;
esac

# Validate every upstream input before starting the first expensive run.
for PHASE5_R in $PHASE5_R_LIST; do
  printf -v PHASE5_R_LABEL 'R%03d' "$PHASE5_R"
  PHASE5_TIME_TABLE="results/runs/phase5_R_sweep/${PHASE5_R_LABEL}/pseudospinodal_fine/analysis/phase5_pseudospinodal_time_dependence.csv"
  test -r "$PHASE5_TIME_TABLE"
  "$PHASE5_PY" src/spinodal_R_sweep_analysis.py print-matched-deltas \
    --time-table "$PHASE5_TIME_TABLE" --offsets 0.010 --T 50 >/dev/null
done

for PHASE5_R in $PHASE5_R_LIST; do
  case "$PHASE5_R" in
    12) PHASE5_N=1024 ;;
    24) PHASE5_N=2048 ;;
    48) PHASE5_N=4096 ;;
    *) echo "ERROR: V2 supports R=12,24,48" >&2; exit 1 ;;
  esac
  printf -v PHASE5_R_LABEL 'R%03d' "$PHASE5_R"
  PHASE5_TIME_TABLE="results/runs/phase5_R_sweep/${PHASE5_R_LABEL}/pseudospinodal_fine/analysis/phase5_pseudospinodal_time_dependence.csv"
  PHASE5_DELTA="$("$PHASE5_PY" src/spinodal_R_sweep_analysis.py print-matched-deltas --time-table "$PHASE5_TIME_TABLE" --offsets 0.010 --T 50)"
  # Append to M6 itself so the existing M=8192 stable block IDs are reused.
  PHASE5_OUTPUT="results/runs/phase5_R_sweep/${PHASE5_R_LABEL}/dispersion"

  mpirun ${NQSV_MPIOPTS} -np 57 \
    "$PHASE5_PY" src/spinodal_phase5_mpi.py \
    --analytic-references --B 2.0 --R "$PHASE5_R" --N "$PHASE5_N" \
    --sigma-J 1.0 --sigma-phi 0.06 --phi-bar 0.0 --a 1.0 \
    --branch stay_to_evacuate --deltas "$PHASE5_DELTA" --modes 0,1,2,3,4,5,6 \
    --epsilon-fraction 0.05 --epsilon-fractions 0.05 \
    --M-total "$PHASE5_M" --block-size 64 \
    --kernel aggregated_exact --initialization prepared_metastable \
    --preparation-width 0.02 --preparation-steps 6 --burn-steps-per-stage 8 \
    --T-fixed 50 --fit-start 0 --fit-end 3 --qR-max-fit 0.35 \
    --track-survival --stage production --M-convergence-candidates "$PHASE5_M_CANDIDATES" \
    --bootstrap-replicates 1000 --task-id-prefix "${PHASE5_R_LABEL}_dispersion_" \
    --resume --no-figures --max-runtime-seconds 53400 --output-dir "$PHASE5_OUTPUT"

  "$PHASE5_PY" src/spinodal_phase5_followup_analysis.py time \
    --input-dir "$PHASE5_OUTPUT" --output-dir "$PHASE5_OUTPUT/analysis" \
    --gamma-eff-min-snr 5 --bootstrap-replicates 1000
  "$PHASE5_PY" src/spinodal_phase5_followup_analysis.py survival \
    --input-dir "$PHASE5_OUTPUT" --output-dir "$PHASE5_OUTPUT/analysis"
done

# Always rebuild V2 decisions from the just-completed production outputs.
"$PHASE5_PY" src/spinodal_phase5_final_validation.py analyze-D \
  --r-sweep-dir results/runs/phase5_R_sweep \
  --output-dir results/runs/phase5_final_validation
cat results/runs/phase5_final_validation/high_precision_D_over_kappa.csv
"$PHASE5_PY" -m json.tool \
  results/runs/phase5_final_validation/high_precision_D_validation_summary.json
