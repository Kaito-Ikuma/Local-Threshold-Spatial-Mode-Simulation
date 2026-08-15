#!/bin/bash
# Production wrapper for the BaseCPU / Intel MPI environment.
set -euo pipefail

cd "$(dirname "$0")/.."
module load BaseCPU
source "$HOME/miniforge3/bin/activate" evac_sim
python scripts/check_squid_mpi_env.py --expected-flavor intelmpi
echo "Submit scripts/run_phase5_squid_intelmpi.sh with qsub."
