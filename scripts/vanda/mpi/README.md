# MPI Vanda Jobs

MPI PBS scripts are located in the parent directory `scripts/vanda/`:

- `test_mpi_correctness.pbs` — builds and runs all MPI test executables (1, 2, 4 ranks)
- `benchmark_mpi_scaling.pbs` — 2D Poisson 512×512 strong-scaling (1–72 ranks)
- `benchmark_mpi_scaling_1024.pbs` — 2D Poisson 1024×1024 strong-scaling (1–72 ranks)

See `docs/task_b/benchmark_methodology.md` for timing protocol and environment configuration.

Submit jobs from the project root on Vanda:
```bash
qsub scripts/vanda/test_mpi_correctness.pbs
qsub scripts/vanda/benchmark_mpi_scaling.pbs
qsub scripts/vanda/benchmark_mpi_scaling_1024.pbs
```
