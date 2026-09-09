#!/bin/bash
#PBS -q DBG
#PBS --group=cm9029
#PBS -b 1
#PBS -l cpunum_job=76
#PBS -l elapstim_req=00:30:00
#PBS -T intmpi
#PBS -v OMP_NUM_THREADS=1
#PBS -v OPENBLAS_NUM_THREADS=1
#PBS -v MKL_NUM_THREADS=1
#PBS -v NUMEXPR_NUM_THREADS=1
set -euo pipefail

cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is not set}"
source scripts/phase5_squid_env.sh
source scripts/phase5_squid_preflight.sh

DBG_ROOT="results/runs/phase5_largeX/benchmarks"
mkdir -p "$DBG_ROOT"
"$PHASE5_PY" src/spinodal_phase5_largex_analysis.py memory \
  --N 4096 --block-size 32 --T 50 --ranks 76 \
  --output "$DBG_ROOT/largeX_memory_estimate.json"

for MPI_RANKS in 4 19 38 57 76; do
  OUTPUT="$DBG_ROOT/np_${MPI_RANKS}"
  mpirun ${NQSV_MPIOPTS:-} -np "$MPI_RANKS" \
    "$PHASE5_PY" src/spinodal_phase5_mpi.py \
    --analytic-references --B 2 --R 24 --N 512 \
    --deltas 0.04 --modes 0,1 --epsilon-fraction 0.05 \
    --M-total 608 --block-size 16 --T-fixed 5 --fit-end 3 \
    --preparation-steps 2 --burn-steps-per-stage 2 \
    --kernel aggregated_exact --track-survival \
    --rng-coupling-mode common_modes \
    --campaign-version 2026.09.09-phase5-largeX-v1 \
    --fit-protocol dbg_fixed_v1 --stage benchmark \
    --bootstrap-replicates 20 --no-resume --no-figures \
    --output-dir "$OUTPUT"
done

"$PHASE5_PY" -c '
import csv,json
from pathlib import Path
root=Path("results/runs/phase5_largeX/benchmarks")
rows=[]
for ranks in (4,19,38,57,76):
    state=json.loads((root/f"np_{ranks}"/"phase5_run_state.json").read_text())
    assignment=json.loads((root/f"np_{ranks}"/"mpi_assignment.json").read_text())
    wall=max(row["wall_seconds"] for row in state["rank_reports"])
    peak=max(row["peak_rss_mb"] for row in state["rank_reports"])
    costs=[row["estimated_cost"] for row in assignment["ranks"].values()]
    rows.append({"mpi_ranks":ranks,"wall_seconds":wall,"peak_rss_mb_per_rank":peak,
                 "trial_site_steps_per_sec":sum(costs)/wall,
                 "load_imbalance":max(costs)/max(sum(costs)/len(costs),1)})
with (root/"largeX_mpi_benchmark.csv").open("w",newline="") as handle:
    writer=csv.DictWriter(handle,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
' 
