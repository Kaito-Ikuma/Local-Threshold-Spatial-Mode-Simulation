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


def validate_python_executable(
    expected_python: str | None,
    actual_python: str | None = None,
) -> tuple[str, str | None]:
    """Resolve both paths and reject an unexpected Python executable."""
    actual_realpath = os.path.realpath(actual_python or sys.executable)
    expected_realpath = (
        os.path.realpath(os.path.expanduser(expected_python))
        if expected_python
        else None
    )
    if expected_realpath and actual_realpath != expected_realpath:
        raise ValueError(
            "Python executable mismatch: "
            f"{actual_realpath!r} != {expected_realpath!r}."
        )
    return actual_realpath, expected_realpath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected-flavor",
        choices=("openmpi", "intelmpi"),
        default=None,
    )
    parser.add_argument(
        "--expected-python",
        default=None,
        help="required Python executable; compared with sys.executable by realpath",
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
    try:
        executable_realpath, expected_python_realpath = validate_python_executable(
            args.expected_python
        )
    except ValueError as error:
        raise SystemExit(f"ERROR: {error}") from error
    payload = {
        "sys_executable": sys.executable,
        "sys_executable_realpath": executable_realpath,
        "sys_prefix": sys.prefix,
        "expected_python": args.expected_python,
        "expected_python_realpath": expected_python_realpath,
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
