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

PHASE5_M="${PHASE5_FS_M_TOTAL:-${PHASE5_FS_M:-4096}}"
PHASE5_TABLE=results/runs/phase5_final_validation/finite_size_pseudospinodal.csv
"$PHASE5_PY" src/spinodal_phase5_finite_size_analysis.py \
  --r-sweep-dir results/runs/phase5_R_sweep \
  --output-dir results/runs/phase5_final_validation

PHASE5_CASE_LIST="${PHASE5_FS_CASES:-12:512,12:2048,24:1024,24:4096}"
PHASE5_CASE_LIST="${PHASE5_CASE_LIST//,/ }"
for PHASE5_CASE in $PHASE5_CASE_LIST; do
  PHASE5_R="${PHASE5_CASE%%:*}"
  PHASE5_N="${PHASE5_CASE##*:}"
  printf -v PHASE5_LABEL 'R%03d_N%04d' "$PHASE5_R" "$PHASE5_N"
  PHASE5_DELTA="$("$PHASE5_PY" src/spinodal_phase5_final_validation.py print-finite-size-matched --table "$PHASE5_TABLE" --R "$PHASE5_R" --N "$PHASE5_N" --offset 0.010)"
  PHASE5_OUTPUT="results/runs/phase5_R_sweep/finite_size/${PHASE5_LABEL}/response_matched"

  mpirun ${NQSV_MPIOPTS} -np 57 \
    "$PHASE5_PY" src/spinodal_phase5_mpi.py \
    --analytic-references --B 2.0 --R "$PHASE5_R" --N "$PHASE5_N" \
    --sigma-J 1.0 --sigma-phi 0.06 --phi-bar 0.0 --a 1.0 \
    --branch stay_to_evacuate --deltas "$PHASE5_DELTA" --modes 0 \
    --epsilon-fraction 0.05 --epsilon-fractions 0.05 \
    --M-total "$PHASE5_M" --block-size 64 \
    --kernel aggregated_exact --initialization prepared_metastable \
    --preparation-width 0.02 --preparation-steps 6 --burn-steps-per-stage 8 \
    --T-fixed 50 --fit-start 0 --fit-end 3 --qR-max-fit 0.35 \
    --track-survival --stage production --M-convergence-candidates 4096,"$PHASE5_M" \
    --bootstrap-replicates 1000 --task-id-prefix "V1_${PHASE5_LABEL}_response_" \
    --resume --no-figures --max-runtime-seconds 53400 --output-dir "$PHASE5_OUTPUT"

  "$PHASE5_PY" src/spinodal_phase5_followup_analysis.py time \
    --input-dir "$PHASE5_OUTPUT" --output-dir "$PHASE5_OUTPUT/analysis" \
    --gamma-eff-min-snr 5 --bootstrap-replicates 1000
  "$PHASE5_PY" src/spinodal_phase5_followup_analysis.py survival \
    --input-dir "$PHASE5_OUTPUT" --output-dir "$PHASE5_OUTPUT/analysis"
done

"$PHASE5_PY" src/spinodal_phase5_finite_size_analysis.py \
  --r-sweep-dir results/runs/phase5_R_sweep \
  --output-dir results/runs/phase5_final_validation
