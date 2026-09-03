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

POSTER_B_R="${POSTER_B_R:-24}"
POSTER_B_M="${POSTER_B_M:-16384}"
POSTER_B_OFFSETS="${POSTER_B_OFFSETS:-0.001,0.002,0.003,0.005}"
POSTER_B_MODES="${POSTER_B_MODES:-0,1,2,3,4,5,6}"
POSTER_B_RANKS="${POSTER_B_RANKS:-57}"
POSTER_B_BLOCK_SIZE="${POSTER_B_BLOCK_SIZE:-64}"
POSTER_B_MAX_RUNTIME="${POSTER_B_MAX_RUNTIME:-53400}"

case "$POSTER_B_R" in
  24) POSTER_B_N=2048 ;;
  48) POSTER_B_N=4096 ;;
  *) echo "ERROR: refinement is staged for R=24 first and R=48 second" >&2; exit 1 ;;
esac

case "$POSTER_B_M" in
  16384) POSTER_B_M_CANDIDATES=4096,8192,16384 ;;
  32768) POSTER_B_M_CANDIDATES=8192,16384,32768 ;;
  65536) POSTER_B_M_CANDIDATES=8192,16384,32768,65536 ;;
  *) echo "ERROR: POSTER_B_M must be 16384, 32768, or 65536" >&2; exit 1 ;;
esac

printf -v POSTER_B_R_LABEL 'R%03d' "$POSTER_B_R"
POSTER_B_TIME_TABLE="results/runs/phase5_R_sweep/${POSTER_B_R_LABEL}/pseudospinodal_fine/analysis/phase5_pseudospinodal_time_dependence.csv"
POSTER_B_OUTPUT="results/runs/poster_ABCD/B_refinement/${POSTER_B_R_LABEL}"
test -r "$POSTER_B_TIME_TABLE"

POSTER_B_DELTAS="$(
  "$PHASE5_PY" src/spinodal_R_sweep_analysis.py print-matched-deltas \
    --time-table "$POSTER_B_TIME_TABLE" --offsets "$POSTER_B_OFFSETS" --T 50
)"

mpirun ${NQSV_MPIOPTS:-} -np "$POSTER_B_RANKS" \
  "$PHASE5_PY" src/spinodal_phase5_mpi.py \
  --analytic-references \
  --B 2.0 --R "$POSTER_B_R" --N "$POSTER_B_N" \
  --sigma-J 1.0 --sigma-phi 0.06 --phi-bar 0.0 --a 1.0 \
  --branch stay_to_evacuate \
  --deltas "$POSTER_B_DELTAS" --modes "$POSTER_B_MODES" \
  --epsilon-fraction 0.05 --epsilon-fractions 0.05 \
  --M-total "$POSTER_B_M" --block-size "$POSTER_B_BLOCK_SIZE" \
  --kernel aggregated_exact --initialization prepared_metastable \
  --preparation-width 0.02 --preparation-steps 6 --burn-steps-per-stage 8 \
  --T-fixed 50 --fit-start 0 --fit-end 3 --qR-max-fit 0.35 \
  --track-survival --stage production \
  --M-convergence-candidates "$POSTER_B_M_CANDIDATES" \
  --bootstrap-replicates 1000 \
  --task-id-prefix "poster_B_refine_${POSTER_B_R_LABEL}_aggregated_exact_" \
  --resume --no-figures --max-runtime-seconds "$POSTER_B_MAX_RUNTIME" \
  --output-dir "$POSTER_B_OUTPUT"

"$PHASE5_PY" -m json.tool "$POSTER_B_OUTPUT/phase5_run_state.json"
