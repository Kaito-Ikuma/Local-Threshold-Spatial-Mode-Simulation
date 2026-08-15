#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPIEXEC_BIN="${MPIEXEC_BIN:-mpiexec}"
MPI_PROCS="${MPI_PROCS:-4}"

PY_SCRIPT="$PROJECT_ROOT/src/spinodal_phase12_mpi.py"
PHASE0_DIR="$PROJECT_ROOT/results/runs/phase0_B2_R12"
OUTPUT_DIR="$PROJECT_ROOT/results/runs/phase12_B2_R12"

# Each MPI rank runs independent NumPy tasks. Keep numerical-library threading
# at one thread per rank to avoid oversubscribing a four-core run.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if ! command -v "$MPIEXEC_BIN" >/dev/null 2>&1; then
  echo "ERROR: mpiexec was not found." >&2
  echo "Use serial execution or install Open MPI + mpi4py." >&2
  exit 1
fi

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "ERROR: missing $PY_SCRIPT" >&2
  exit 1
fi

"$PYTHON_BIN" -c "from mpi4py import MPI; print('mpi4py OK:', MPI.Get_library_version().splitlines()[0])"

echo "Launching $MPI_PROCS Phase1-2 task workers..."

"$MPIEXEC_BIN" -n "$MPI_PROCS" "$PYTHON_BIN" "$PY_SCRIPT" \
  --phase0-dir "$PHASE0_DIR" \
  --N 1024 \
  --modes 0,1,2,3,4,5,6 \
  --epsilon-fraction 0.05 \
  --tau-multiplier 6.0 \
  --T-min 50 \
  --qR-max-fit 0.35 \
  --output-dir "$OUTPUT_DIR"
