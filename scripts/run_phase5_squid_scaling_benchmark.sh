#!/bin/bash
#PBS -q SQUID
#PBS --group=cm9029
#PBS -b 1
#PBS -l cpunum_job=76
#PBS -l elapstim_req=02:00:00
#PBS -T intmpi
#PBS -v OMP_NUM_THREADS=1
#PBS -v OPENBLAS_NUM_THREADS=1
#PBS -v MKL_NUM_THREADS=1
#PBS -v NUMEXPR_NUM_THREADS=1
set -euo pipefail

cd "$PBS_O_WORKDIR"
source scripts/phase5_squid_env.sh
source scripts/phase5_squid_preflight.sh

SCALING_ROOT=results/runs/phase5_scaling_benchmark
mkdir -p "$SCALING_ROOT"
for MPI_RANKS in 19 38 57 76; do
  OUTPUT_DIR="$SCALING_ROOT/np_$MPI_RANKS"
  mpirun ${NQSV_MPIOPTS} -np "$MPI_RANKS" \
    "$PHASE5_PY" src/spinodal_phase5_mpi.py \
    --N 256 \
    --deltas 1e-3 \
    --modes 0,1 \
    --epsilon-fraction 0.05 \
    --M-total 1216 \
    --block-size 16 \
    --kernel aggregated_exact \
    --initialization bernoulli_meanfield \
    --T-fixed 20 \
    --fit-end 8 \
    --bootstrap-replicates 20 \
    --stage benchmark \
    --no-resume \
    --no-figures \
    --output-dir "$OUTPUT_DIR"
done

"$PHASE5_PY" -c '
import csv, json
from pathlib import Path
root=Path("results/runs/phase5_scaling_benchmark")
rows=[]
for ranks in (19,38,57,76):
    run_dir=root/f"np_{ranks}"
    state=json.loads((run_dir/"phase5_run_state.json").read_text())
    assignment=json.loads((run_dir/"mpi_assignment.json").read_text())
    wall=max(item["wall_seconds"] for item in state["rank_reports"])
    work=sum(item["estimated_cost"] for item in assignment["ranks"].values())
    rows.append({
        "mpi_ranks":ranks,
        "wall_seconds":wall,
        "trial_site_steps_per_sec":work/wall,
    })
baseline=rows[0]["wall_seconds"]
for row in rows:
    row["speedup"]=baseline/row["wall_seconds"]
    row["parallel_efficiency"]=row["speedup"]/(row["mpi_ranks"]/19)
with (root/"phase5_mpi_scaling_benchmark.csv").open("w",newline="") as handle:
    writer=csv.DictWriter(handle,fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
'
