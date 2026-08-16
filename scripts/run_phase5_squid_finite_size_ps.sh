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

PHASE5_FS_STAGE="${PHASE5_FS_STAGE:-coarse}"
PHASE5_FS_TAG="${PHASE5_FS_TAG:-}"
PHASE5_FS_CASES="${PHASE5_FS_CASES:-12:512,12:2048,24:1024,24:4096}"
PHASE5_PLAN=results/runs/phase5_R_sweep/finite_size/finite_size_fine_plan.csv

IFS=',' read -r -a PHASE5_CASE_ARRAY <<< "$PHASE5_FS_CASES"
for PHASE5_CASE in "${PHASE5_CASE_ARRAY[@]}"; do
  PHASE5_R="${PHASE5_CASE%%:*}"
  PHASE5_N="${PHASE5_CASE##*:}"
  printf -v PHASE5_LABEL 'R%03d_N%04d' "$PHASE5_R" "$PHASE5_N"
  PHASE5_ROOT="results/runs/phase5_R_sweep/finite_size/${PHASE5_LABEL}"

  case "$PHASE5_FS_STAGE" in
    coarse)
      if [ "$PHASE5_R" -eq 12 ]; then
        PHASE5_DELTAS="${PHASE5_FS_DELTAS:-0.04,0.05,0.06,0.07,0.08,0.10}"
      else
        PHASE5_DELTAS="${PHASE5_FS_DELTAS:-0.0177777777778,0.0266666666667,0.04,0.05,0.06}"
      fi
      PHASE5_DELTAS="${PHASE5_DELTAS//:/,}"
      PHASE5_M=2048
      PHASE5_OUTPUT="${PHASE5_ROOT}/pseudospinodal_coarse${PHASE5_FS_TAG}"
      ;;
    fine)
      PHASE5_DELTAS="$("$PHASE5_PY" src/spinodal_phase5_final_validation.py print-finite-size-plan --plan "$PHASE5_PLAN" --R "$PHASE5_R" --N "$PHASE5_N")"
      PHASE5_M=4096
      PHASE5_OUTPUT="${PHASE5_ROOT}/pseudospinodal_fine"
      ;;
    *) echo "ERROR: PHASE5_FS_STAGE must be coarse or fine" >&2; exit 1 ;;
  esac

  mpirun ${NQSV_MPIOPTS} -np 57 \
    "$PHASE5_PY" src/spinodal_phase5_mpi.py \
    --analytic-references --B 2.0 --R "$PHASE5_R" --N "$PHASE5_N" \
    --sigma-J 1.0 --sigma-phi 0.06 --phi-bar 0.0 --a 1.0 \
    --branch stay_to_evacuate --deltas "$PHASE5_DELTAS" --modes 0 \
    --epsilon-fraction 0.0 --unperturbed --M-total "$PHASE5_M" --block-size 64 \
    --kernel aggregated_exact --initialization prepared_metastable \
    --preparation-width 0.02 --preparation-steps 6 --burn-steps-per-stage 8 \
    --T-fixed 50 --fit-start 0 --fit-end 3 --qR-max-fit 0.35 \
    --track-survival --stage pilot --M-convergence-candidates "$PHASE5_M" \
    --bootstrap-replicates 2 --task-id-prefix "V1_${PHASE5_LABEL}_" \
    --resume --no-figures --max-runtime-seconds 53400 --output-dir "$PHASE5_OUTPUT"

  "$PHASE5_PY" src/spinodal_phase5_followup_analysis.py pseudospinodal \
    --input-dir "$PHASE5_OUTPUT" --output-dir "$PHASE5_OUTPUT/analysis" \
    --criterion-probability 0.10 --primary-T 50 --observation-times 20,30,40,50
done

if [ "$PHASE5_FS_STAGE" = coarse ]; then
  "$PHASE5_PY" src/spinodal_phase5_final_validation.py plan-finite-size \
    --r-sweep-dir results/runs/phase5_R_sweep --output "$PHASE5_PLAN"
  cat "$PHASE5_PLAN"
else
  "$PHASE5_PY" src/spinodal_phase5_final_validation.py analyze-finite-size \
    --r-sweep-dir results/runs/phase5_R_sweep \
    --output-dir results/runs/phase5_final_validation
fi
