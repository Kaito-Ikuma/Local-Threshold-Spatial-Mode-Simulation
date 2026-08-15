#!/bin/bash
# Shared Phase5 SQUID runtime environment. Source this file from PBS scripts.

module purge
module load "${PHASE5_PY_MODULE:-BasePy/2026}"
module load "${PHASE5_MPI_MODULE:-BaseCPU/2026}"

export PHASE5_VENV="${PHASE5_VENV:-/sqfs/work/cm9029/${USER}/phase5_venv}"
export PHASE5_PY="${PHASE5_PY:-${PHASE5_VENV}/bin/python}"

case "$PHASE5_PY" in
    /*) ;;
    *)
        echo "ERROR: PHASE5_PY must be an absolute path: $PHASE5_PY" >&2
        if [ "${BASH_SOURCE[0]}" != "$0" ]; then
            return 1
        fi
        exit 1
        ;;
esac

if [ ! -x "$PHASE5_PY" ]; then
    echo "ERROR: Phase5 Python is not executable: $PHASE5_PY" >&2
    if [ "${BASH_SOURCE[0]}" != "$0" ]; then
        return 1
    fi
    exit 1
fi
