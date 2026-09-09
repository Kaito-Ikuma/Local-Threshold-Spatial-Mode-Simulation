#!/bin/bash
#PBS -q SQUID
#PBS --group=cm9029
#PBS -b 1
#PBS -l cpunum_job=76
#PBS -l elapstim_req=10:00:00
#PBS -T intmpi
#PBS -v OMP_NUM_THREADS=1
#PBS -v OPENBLAS_NUM_THREADS=1
#PBS -v MKL_NUM_THREADS=1
#PBS -v NUMEXPR_NUM_THREADS=1
set -euo pipefail

cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is not set}"
source scripts/phase5_squid_env.sh
source scripts/phase5_squid_preflight.sh
PS_TABLE="results/runs/phase5_R_sweep/finite_size/R024_N4096/pseudospinodal_fine/analysis/phase5_pseudospinodal_time_dependence.csv"
DELTA_PS="$($PHASE5_PY -c 'import pandas as pd,sys; d=pd.read_csv(sys.argv[1]); print(d.loc[d.T_obs.eq(50),"delta_ps_estimate"].iloc[0])' "$PS_TABLE")"
DELTAS="${LARGEX_EPSILON_DELTAS:-$($PHASE5_PY -c 'import sys; p=float(sys.argv[1]); print(f"{p+.00025:.12g},{p+.001:.12g}")' "$DELTA_PS")}" 
FIT_END="${LARGEX_FIT_END:-5}"
OUTPUT="results/runs/phase5_largeX/R024/epsilon_scout"

: "${LARGEX_RANKS:?Select 19, 38, or 76 from the DBG benchmark}"
mpirun ${NQSV_MPIOPTS:-} -np "$LARGEX_RANKS" \
  "$PHASE5_PY" src/spinodal_phase5_mpi.py \
  --analytic-references --B 2 --R 24 --N 4096 \
  --deltas "$DELTAS" --modes 0,4,9 \
  --epsilon-fraction 0.05 --epsilon-fractions 0.025,0.05,0.10 \
  --M-total "${LARGEX_M:-8192}" --block-size 32 \
  --kernel aggregated_exact --initialization prepared_metastable \
  --T-fixed 50 --fit-start 0 --fit-end "$FIT_END" --survivor-cohort-end "$FIT_END" \
  --track-survival --rng-coupling-mode common_modes --harmonic-orders 3,5 \
  --survival-criterion-primary 0.90 --survival-criterion-sensitivity 0.80 \
  --fit-protocol q0_tau_survival_frozen_v1 \
  --campaign-version 2026.09.09-phase5-largeX-v1 \
  --bootstrap-replicates 1000 --stage pilot --resume --no-figures \
  --max-runtime-seconds 35400 --task-id-prefix largeX_R024_epsilon_ \
  --output-dir "$OUTPUT"

"$PHASE5_PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["all_complete"], "epsilon scout incomplete: resubmit the same job to resume"' "$OUTPUT/phase5_run_state.json"
"$PHASE5_PY" src/spinodal_phase5_largex_analysis.py analyze \
  --input-dir "$OUTPUT" --output-dir "$OUTPUT/largeX_analysis" \
  --delta-ps "$DELTA_PS" --bootstrap-replicates 2000 --no-figures
