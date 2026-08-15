#!/bin/bash
# Log and validate the shared Phase5 Python/Intel-MPI environment.

phase5_preflight_main() (
set -euo pipefail

if [ -z "${PHASE5_VENV:-}" ] || [ -z "${PHASE5_PY:-}" ]; then
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/phase5_squid_env.sh"
fi

echo "===== Phase5 job context ====="
date
hostname
pwd
module list
echo "PBS_JOBID=${PBS_JOBID:-unset}"
echo "PBS_O_WORKDIR=${PBS_O_WORKDIR:-unset}"

echo "===== Phase5 Python environment ====="
echo "PHASE5_VENV=$PHASE5_VENV"
echo "PHASE5_PY=$PHASE5_PY"
"$PHASE5_PY" --version
"$PHASE5_PY" -c '
import sys
print("sys.executable =", sys.executable)
print("sys.prefix     =", sys.prefix)
'

echo "===== Python packages ====="
"$PHASE5_PY" -c '
import numpy
import scipy
import pandas
import mpi4py
from mpi4py import MPI

print("numpy  =", numpy.__version__)
print("scipy  =", scipy.__version__)
print("pandas =", pandas.__version__)
print("mpi4py =", mpi4py.__version__)
print("MPI library:")
print(MPI.Get_library_version())
'

echo "===== MPI launcher ====="
which mpirun
mpirun --version

"$PHASE5_PY" scripts/check_squid_mpi_env.py \
    --expected-flavor intelmpi \
    --expected-python "$PHASE5_PY"
)

if ! phase5_preflight_main; then
    echo "ERROR: Phase5 SQUID preflight failed." >&2
    unset -f phase5_preflight_main
    if [ "${BASH_SOURCE[0]}" != "$0" ]; then
        return 1
    fi
    exit 1
fi
unset -f phase5_preflight_main
