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

POSTER_R="${POSTER_R:-24}"
POSTER_KERNEL="${POSTER_KERNEL:-aggregated_exact}"
if [ -z "${POSTER_M:-}" ]; then
  if [ "$POSTER_KERNEL" = "direct_J" ]; then
    POSTER_M=64
  else
    POSTER_M=8192
  fi
fi
POSTER_RANKS="${POSTER_RANKS:-57}"
POSTER_BLOCK_SIZE="${POSTER_BLOCK_SIZE:-64}"
POSTER_MAX_RUNTIME="${POSTER_MAX_RUNTIME:-53400}"

case "$POSTER_R" in
  6) POSTER_N=512 ;;
  12) POSTER_N=1024 ;;
  24) POSTER_N=2048 ;;
  48) POSTER_N=4096 ;;
  96) POSTER_N=8192 ;;
  *) echo "ERROR: POSTER_R must be one of 6,12,24,48,96" >&2; exit 1 ;;
esac

case "$POSTER_M" in
  64) POSTER_M_CANDIDATES=32,64 ;;
  128) POSTER_M_CANDIDATES=32,64,128 ;;
  256) POSTER_M_CANDIDATES=64,128,256 ;;
  8192) POSTER_M_CANDIDATES=2048,4096,8192 ;;
  16384) POSTER_M_CANDIDATES=4096,8192,16384 ;;
  32768) POSTER_M_CANDIDATES=8192,16384,32768 ;;
  65536) POSTER_M_CANDIDATES=8192,16384,32768,65536 ;;
  *) echo "ERROR: POSTER_M must be 64, 128, 256, 8192, 16384, 32768, or 65536" >&2; exit 1 ;;
esac

printf -v POSTER_R_LABEL 'R%03d' "$POSTER_R"
POSTER_TIME_TABLE="results/runs/phase5_R_sweep/${POSTER_R_LABEL}/pseudospinodal_fine/analysis/phase5_pseudospinodal_time_dependence.csv"
test -r "$POSTER_TIME_TABLE"

if [ "$POSTER_KERNEL" = "direct_J" ]; then
  # Small correctness reference only; it is not the production data source.
  POSTER_OFFSETS="${POSTER_OFFSETS:-0.020}"
  POSTER_MODES="${POSTER_MODES:-0,1}"
  POSTER_OUTPUT="results/runs/poster_ABCD/direct_J_reference/${POSTER_R_LABEL}"
else
  POSTER_OFFSETS="${POSTER_OFFSETS:-0.005,0.0075,0.010,0.015,0.020,0.030,0.040}"
  case "$POSTER_R" in
    24|48) POSTER_MODES="${POSTER_MODES:-0,1,2,3,4,5,6}" ;;
    *) POSTER_MODES="${POSTER_MODES:-0}" ;;
  esac
  POSTER_OUTPUT="results/runs/poster_ABCD/microscopic/${POSTER_R_LABEL}"
fi

POSTER_DELTAS="$(
  "$PHASE5_PY" src/spinodal_R_sweep_analysis.py print-matched-deltas \
    --time-table "$POSTER_TIME_TABLE" --offsets "$POSTER_OFFSETS" --T 50
)"

mpirun ${NQSV_MPIOPTS:-} -np "$POSTER_RANKS" \
  "$PHASE5_PY" src/spinodal_phase5_mpi.py \
  --analytic-references \
  --B 2.0 --R "$POSTER_R" --N "$POSTER_N" \
  --sigma-J 1.0 --sigma-phi 0.06 --phi-bar 0.0 --a 1.0 \
  --branch stay_to_evacuate \
  --deltas "$POSTER_DELTAS" --modes "$POSTER_MODES" \
  --epsilon-fraction 0.05 --epsilon-fractions 0.05 \
  --M-total "$POSTER_M" --block-size "$POSTER_BLOCK_SIZE" \
  --kernel "$POSTER_KERNEL" --initialization prepared_metastable \
  --preparation-width 0.02 --preparation-steps 6 --burn-steps-per-stage 8 \
  --T-fixed 50 --fit-start 0 --fit-end 3 --qR-max-fit 0.35 \
  --track-survival --stage production \
  --M-convergence-candidates "$POSTER_M_CANDIDATES" \
  --bootstrap-replicates 1000 \
  --task-id-prefix "poster_${POSTER_R_LABEL}_${POSTER_KERNEL}_" \
  --resume --no-figures --max-runtime-seconds "$POSTER_MAX_RUNTIME" \
  --output-dir "$POSTER_OUTPUT"

"$PHASE5_PY" -m json.tool "$POSTER_OUTPUT/phase5_run_state.json"
