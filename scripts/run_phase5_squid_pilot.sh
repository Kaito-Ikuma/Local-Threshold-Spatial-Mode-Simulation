#!/bin/bash
#PBS -q SQUID
#PBS --group=cm9029
#PBS -b 1
#PBS -l cpunum_job=76
#PBS -l elapstim_req=15:00:00
#PBS -T openmpi
#PBS -v NQSV_MPI_MODULE=BaseGCC
#PBS -v OMP_NUM_THREADS=1
#PBS -v OPENBLAS_NUM_THREADS=1
#PBS -v MKL_NUM_THREADS=1
#PBS -v NUMEXPR_NUM_THREADS=1
set -euo pipefail

cd "$PBS_O_WORKDIR"
module load BaseGCC
source "$HOME/miniforge3/bin/activate" evac_env
python scripts/check_squid_mpi_env.py --expected-flavor openmpi

mpirun $NQSV_MPIOPTS -np 76 -npernode 76 \
  python src/spinodal_phase5_mpi.py \
  --N 1024 \
  --deltas 1e-3,1e-4,1e-5 \
  --modes 0,1,4 \
  --epsilon-fraction 0.05 \
  --epsilon-fractions 0.025,0.05,0.10 \
  --M-total 512 \
  --block-size 32 \
  --kernel aggregated_exact \
  --initialization prepared_metastable \
  --stage pilot \
  --M-convergence-candidates 128,256,512 \
  --resume \
  --max-runtime-seconds 53400 \
  --output-dir results/runs/phase5_B2_R12_pilot
