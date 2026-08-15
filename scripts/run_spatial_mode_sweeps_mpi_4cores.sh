#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPIEXEC_BIN="${MPIEXEC_BIN:-mpiexec}"
MPI_PROCS="${MPI_PROCS:-4}"
PY_SCRIPT="$PROJECT_ROOT/src/spatial_mode_presentation_materials_sweeps_mpi.py"
BASE_SCRIPT="$PROJECT_ROOT/src/spatial_mode_ensemble_validation.py"
OUTPUT_DIR="$PROJECT_ROOT/results/presentation_materials"

# Avoid hidden nested threading inside each MPI process.  Four MPI ranks × one
# numerical-library thread is appropriate for a 16-GB Apple-silicon Mac.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if ! command -v "$MPIEXEC_BIN" >/dev/null 2>&1; then
  echo "ERROR: mpiexec was not found." >&2
  echo "Install mpi4py + Open MPI, then reopen the terminal." >&2
  exit 1
fi

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "ERROR: missing $PY_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$BASE_SCRIPT" ]]; then
  echo "ERROR: missing $BASE_SCRIPT" >&2
  exit 1
fi

"$PYTHON_BIN" -c "from mpi4py import MPI; print('mpi4py OK:', MPI.Get_library_version().splitlines()[0])"
"$PYTHON_BIN" "$PY_SCRIPT" --version

echo "Launching $MPI_PROCS MPI ranks..."

"$MPIEXEC_BIN" -n "$MPI_PROCS" "$PYTHON_BIN" "$PY_SCRIPT" \
  --base-script "$BASE_SCRIPT" \
  --output-dir "$OUTPUT_DIR" \
  --N 96 \
  --R 12 \
  --T 40 \
  --ensemble 2000 \
  --B 0.72 \
  --sigma-J 1.0 \
  --sigma-phi 0.06 \
  --phi-bar 0.0 \
  --h 0.0 \
  --epsilon 0.05 \
  --fit-steps 4 \
  --modes 0,1,2,4,6,8 \
  --selected-mode 2 \
  --negative-mode 6 \
  --n-seeds 12 \
  --seed0 20260801 \
  --scan-n-seeds 8 \
  --scan-tmax 4 \
  --R-scan 6,12,18,24 \
  --B-scan 0.4,0.6,0.8,0.95 \
  --delta-scan=-0.10,-0.05,0.0,0.05,0.10
