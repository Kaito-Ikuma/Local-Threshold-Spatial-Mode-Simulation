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
export LARGEX_M="${LARGEX_M:-8192}"
source jobs/largeX/_run_production_R24.sh

