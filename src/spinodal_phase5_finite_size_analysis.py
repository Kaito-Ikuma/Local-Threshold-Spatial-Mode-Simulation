#!/usr/bin/env python3
"""Compute-only V1 finite-size analysis wrapper.

The implementation lives in ``spinodal_phase5_final_validation`` so the V1
tables and the final V-A aggregation use exactly the same estimators.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spinodal_phase5_final_validation import analyze_finite_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-sweep-dir", type=Path, default=Path("results/runs/phase5_R_sweep"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs/phase5_final_validation"))
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    for path in analyze_finite_size(args):
        print(path)


if __name__ == "__main__":
    main()
