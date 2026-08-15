#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

python3 "$PROJECT_ROOT/src/spatial_mode_presentation_materials_sweeps_v2.py" \
  --base-script "$PROJECT_ROOT/src/spatial_mode_ensemble_validation.py" \
  --output-dir "$PROJECT_ROOT/results/presentation_materials" \
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
