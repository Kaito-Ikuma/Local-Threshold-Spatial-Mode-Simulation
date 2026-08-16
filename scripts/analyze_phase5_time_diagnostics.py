#!/usr/bin/env python3
"""CLI wrapper for checkpoint-only Phase5 follow-up diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spinodal_phase5_followup_analysis import main


if __name__ == "__main__":
    main()
