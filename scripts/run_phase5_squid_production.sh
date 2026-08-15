#!/bin/bash
# Production wrapper for the BasePy / BaseCPU / Intel MPI environment.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/phase5_squid_env.sh
source scripts/phase5_squid_preflight.sh
echo "Submit scripts/run_phase5_squid_intelmpi.sh with qsub."
