# MPI CG work area

This directory contains Person B's distributed-memory MPI Conjugate Gradient implementation.

Files:
- `mpi_partition.cpp` — row-wise CSR matrix partitioning and gather/scatter
- `mpi_ops.cpp` — distributed SpMV (Allgatherv), dot products, vector operations
- `mpi_solver.cpp` — collective-safe MPI CG solver
- `benchmark_mpi.cpp` — MPI benchmark harness with MPI_Wtime and MPI_Reduce(MAX)
- `benchmark_mpi_csv.cpp` — CSV writer (rank 0 only)
- `run_mpi_benchmark.cpp` — command-line benchmark runner

Public interfaces are in the top-level `include/` directory:
- `include/mpi/mpi_partition.hpp`
- `include/mpi/mpi_ops.hpp`
- `include/mpi/mpi_solver.hpp`
- `include/mpi/mpi_benchmark.hpp`

Tests are in the top-level `tests/` directory:
- `tests/test_mpi_partition.cpp`
- `tests/test_mpi_spmv.cpp`
- `tests/test_mpi_cg.cpp`

See `REPORT_TASK_B.md` and `docs/task_b/` for design and benchmark methodology.