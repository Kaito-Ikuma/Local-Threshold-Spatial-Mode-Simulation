#!/usr/bin/env python3
"""Print SQUID Python/MPI environment and reject an explicit flavor mismatch."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys

import numpy as np

try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None
    MPI = None


def detect_flavor(text: str) -> str:
    lowered = text.lower()
    if "open mpi" in lowered or "openmpi" in lowered:
        return "openmpi"
    if "intel" in lowered and "mpi" in lowered:
        return "intelmpi"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected-flavor",
        choices=("openmpi", "intelmpi"),
        default=None,
    )
    args = parser.parse_args()
    mpirun = shutil.which("mpirun")
    mpirun_version = ""
    if mpirun:
        completed = subprocess.run(
            [mpirun, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        mpirun_version = (completed.stdout + completed.stderr).strip()
    library = MPI.Get_library_version() if MPI is not None else ""
    payload = {
        "sys_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "mpi4py_version": mpi4py.__version__ if mpi4py is not None else None,
        "mpi_library_version": library or None,
        "mpi_library_flavor": detect_flavor(library),
        "hostname": socket.gethostname(),
        "mpi_world_size": MPI.COMM_WORLD.Get_size() if MPI is not None else 1,
        "mpirun_path": mpirun,
        "mpirun_version": mpirun_version or None,
        "mpirun_flavor": detect_flavor(mpirun_version),
        "pbs_variables": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith("PBS") or name.startswith("NQSV")
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if MPI is None:
        raise SystemExit("ERROR: mpi4py is not importable in the active Python environment.")
    if args.expected_flavor and payload["mpi_library_flavor"] != args.expected_flavor:
        raise SystemExit(
            "ERROR: mpi4py MPI library flavor "
            f"{payload['mpi_library_flavor']!r} does not match requested "
            f"{args.expected_flavor!r}."
        )
    if (
        payload["mpirun_flavor"] != "unknown"
        and payload["mpi_library_flavor"] != payload["mpirun_flavor"]
    ):
        raise SystemExit(
            "ERROR: mpirun and mpi4py are linked to different MPI implementations."
        )


if __name__ == "__main__":
    main()
