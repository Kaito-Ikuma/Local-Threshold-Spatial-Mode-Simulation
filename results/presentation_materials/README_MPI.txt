MPI presentation-material generation
====================================
script version: 2026.08.03-sweeps-mpi-v1
MPI ranks: 4

Parallelization
---------------
Independent seed indices are distributed cyclically across MPI ranks:
rank r computes r, r+P, r+2P, ... for P ranks.
Only compact per-seed mode amplitudes and fit statistics are gathered.
PNG/CSV writing is performed by rank 0 only.

Memory
------
A rank retains only one ModeResult at a time.  It does not retain all
spatiotemporal arrays from all seeds and modes.  This is intended for a
16-GB Apple-silicon Mac with four MPI ranks.
