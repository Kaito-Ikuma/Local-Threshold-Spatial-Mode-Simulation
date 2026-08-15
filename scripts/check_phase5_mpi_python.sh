#!/bin/bash
# Verify that every MPI rank launches the same Phase5 venv interpreter.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"
source scripts/phase5_squid_env.sh
source scripts/phase5_squid_preflight.sh

MPI_TEST_RANKS=${MPI_TEST_RANKS:-2}
mpirun ${NQSV_MPIOPTS:-} -np "$MPI_TEST_RANKS" \
    "$PHASE5_PY" -c '
from mpi4py import MPI
import os
import socket
import sys

expected = os.path.realpath(sys.argv[1])
actual = os.path.realpath(sys.executable)
print(
    "rank=", MPI.COMM_WORLD.Get_rank(),
    "host=", socket.gethostname(),
    "python=", sys.executable,
    "python_realpath=", actual,
    "MPI=", MPI.Get_library_version().strip(),
    flush=True,
)
if actual != expected:
    raise SystemExit(f"ERROR: MPI rank Python mismatch: {actual} != {expected}")
' "$PHASE5_PY"
