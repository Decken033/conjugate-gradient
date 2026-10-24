# Person B — MPI Benchmark Methodology

## 1. Timing Protocol

### Timing boundaries

- **Excluded**: Matrix generation, Matrix Market loading, CSR partitioning and distribution, RHS construction (`b = A*ones`), warmup solves, solution error computation, CSV writing, console output
- **Included**: Full `solve_mpi_cg()` call including all MPI collectives (Allgatherv, Allreduce) and local computation

### Timer

`MPI_Wtime()` is used for all MPI solver timing. This is the standard wall-clock timer for MPI programs and is defined across all ranks.

### Max-time reduction

```cpp
MPI_Barrier(comm);           // synchronize before start
t_start = MPI_Wtime();
solve_mpi_cg(...);
t_end = MPI_Wtime();
local_elapsed = t_end - t_start;

MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
```

`max_elapsed` captures the slowest rank's time, representing the true wall-clock time the parallel solve took. The serial and OpenMP solvers use `std::chrono::steady_clock` for single-rank timing.

### Warmup protocol

Default: 4 warmups before 10 timed runs (matching Person A's methodology).

### Reported statistics

The final Person B report uses the arithmetic mean of the ten `solve_seconds` values as `T(P)`, together with sample standard deviation and observed minimum/maximum. The benchmark runner also prints the median for quick job monitoring. Speedup and efficiency in the report use the mean consistently for every rank count.

## 2. CSV Schema

Same 12-column schema as Person A (serial and OpenMP):

```
solver,matrix_name,n,nnz,ranks,threads,run_index,iterations,
converged,solve_seconds,true_relative_residual,relative_solution_error
```

For pure MPI: `solver=mpi`, `threads=1`.

17-digit floating-point precision.

## 3. Speedup Definitions

The final MPI scaling tables use the MPI one-rank run as their baseline:

**MPI 1-rank relative speedup:**
```
S_mpi(P) = T_mpi(1) / T_mpi(P)
```
This is the primary within-implementation scaling metric.

**Parallel efficiency:**
```
E(P) = S_mpi(P) / P * 100%
```

## 4. Workloads

| Matrix | N | NNZ | Notes |
|---|---|---|---|
| 2D Poisson 512×512 | 262,144 | 1,308,672 | Primary scaling experiment |
| 2D Poisson 1024×1024 | 1,048,576 | 5,238,784 | Secondary scaling experiment |
| 3D Poisson 4³ and 8³ | 64 and 512 | — | MPI correctness validation; no MPI performance campaign recorded |
| SuiteSparse thermal2 | 1,228,045 | 8,580,313 | CLI input supported; no MPI run result claimed here |

## 5. Cluster Configuration

| Component | Value |
|---|---|
| Cluster | NUS Vanda HPC |
| Modules in scaling jobs | `CMake/3.27.6-GCCcore-13.2.0` + `OpenMPI/4.1.6-GCC-13.2.0` |
| Compiler | GCCcore 13.2.0 |
| CMake build type | Release (`-O3 -DNDEBUG`) |
| MPI | Open MPI 4.1.6 in both recorded scaling jobs |
| Process binding | `--bind-to core` |

For strong scaling, request:
```
#PBS -l select=2:ncpus=36:mpiprocs=36:mem=32gb
```
This gives 2 nodes × 36 slots = 72 MPI slots.

## 6. Cross-Node vs Within-Node Scaling

For P ≤ 36: all ranks fit on one node, no inter-node communication.
For P > 36: ranks span two nodes, inter-node communication via Vanda's network.

The final report notes the within-node/cross-node distinction: up to 36 ranks fit within one allocated node and 72 ranks spans two nodes. These regimes are discussed separately because network overhead affects the comparison.

## 7. Reproducibility

Each PBS job generates a results file named:
```
mpi_2d<size>_p<ranks>_<JOBID>.csv
```

Record PBS job ID, compute node, allocation, and module versions for each run.

## 8. Recorded Results and Limitations

The selected 512² and 1024² campaigns each contain eight rank settings and ten records per setting. All 160 selected records have `converged=1`; the report recomputes speedup and efficiency from the mean solver-only time. The scaling figure is `docs/task_b/mpi_scaling.svg`.

- Latest correctness job: 10/10 CTest targets passed; partition, SpMV, and CG executables each passed at 1, 2, and 4 ranks.
- Scaling jobs: PBS `1482451.stdct-mgmt-02` (512²) and `1482491.stdct-mgmt-02` (1024²).
- Full results and cross-node discussion: `REPORT_TASK_B.md` and `REPORT_TASK_B_ZH.md`.

- Cross-solver comparisons with serial timing are omitted from the final tables; serial uses `std::chrono::steady_clock` while MPI uses `MPI_Wtime`.
- MPI_Allgatherv communication cost increases with P (each rank sends/receives full vector).
- UCX component warning observed on Vanda (ENVIRONMENT.md §5); may affect network performance.
- Results on different compute nodes are not directly comparable.
- No independent per-rank timing breakdown; load balance across ranks is not measured separately.
