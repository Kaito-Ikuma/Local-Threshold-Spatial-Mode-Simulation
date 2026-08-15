#!/bin/bash
# Production wrapper. Select the MPI-flavor script only after the environment check.
set -euo pipefail

cd "$(dirname "$0")/.."
python scripts/check_squid_mpi_env.py
echo "Submit run_phase5_squid_openmpi.sh or run_phase5_squid_intelmpi.sh with qsub."
