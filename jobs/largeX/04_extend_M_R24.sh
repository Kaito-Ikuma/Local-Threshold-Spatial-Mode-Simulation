#!/bin/bash
#PBS -q SQUID
#PBS --group=cm9029
#PBS -b 1
#PBS -l cpunum_job=76
#PBS -l elapstim_req=15:00:00
#PBS -T intmpi
#PBS -v OMP_NUM_THREADS=1
#PBS -v OPENBLAS_NUM_THREADS=1
#PBS -v MKL_NUM_THREADS=1
#PBS -v NUMEXPR_NUM_THREADS=1
set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is not set}"
source scripts/phase5_squid_env.sh
source scripts/phase5_squid_preflight.sh
: "${LARGEX_M:?Set the next M in 16384,32768,65536}"
case "$LARGEX_M" in 16384|32768|65536) ;; *) echo "LARGEX_M must be 16384, 32768, or 65536" >&2; exit 2;; esac
source jobs/largeX/_run_production_R24.sh

