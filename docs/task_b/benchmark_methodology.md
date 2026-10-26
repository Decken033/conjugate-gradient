# Person B — MPI Benchmark Methodology

## 1. Timing Protocol

### Timing boundaries

- **Excluded**: Matrix generation, Matrix Market loading, CSR partitioning and distribution, RHS construction (`b = A*ones`), warmup solves, solution error computation, CSV writing, console output
- **Included**: Full `solve_mpi_cg()` call including all MPI collectives (Allgatherv, Allreduce) and local computation

### Timer

`MPI_Wtime()` is used for all MPI solver timing. Each rank reads its local wall-clock timer; the elapsed intervals are reduced to the maximum after the solve.

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

The benchmark runner defaults to one warmup; the selected Poisson and thermal2 PBS campaigns explicitly pass four warmups and ten timed runs (matching Person A's campaign protocol).

### Reported statistics

The final Person B report uses the arithmetic mean of the ten `solve_seconds` values as `T(P)`, together with sample standard deviation and observed minimum/maximum. The benchmark runner also prints the median for quick job monitoring. Speedup and efficiency in the report use the mean consistently for every rank count.

## 2. CSV Schema

The MPI CSV has these 12 columns (the OpenMP CSV appends scheduling fields):

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
| SuiteSparse thermal2 | 1,228,045 | 8,580,313 | Real-world MPI scaling experiment at 1, 2, 4, 8, 16 ranks |

## 5. Cluster Configuration

| Component | Value |
|---|---|
| Cluster | NUS Vanda HPC |
| Modules in scaling jobs | `CMake/3.27.6-GCCcore-13.2.0` + `OpenMPI/4.1.6-GCC-13.2.0` |
| Compiler | GCCcore 13.2.0 |
| CMake build type | Release requested by the scaling scripts; historical cache is unavailable for thermal2 |
| MPI | Open MPI 4.1.6 loaded in all three recorded scaling jobs |
| Process binding | Poisson scripts request `--bind-to core`; thermal2 script does not specify binding |

The 512 Poisson script requests:
```
#PBS -l select=2:ncpus=36:mpiprocs=36:mem=32gb
```
The 1024 Poisson and thermal2 scripts request 64 GB per node instead. Each two-node request specifies 36 MPI slots per node (72 allocated slots), which differs from the MPI process count launched for any one configuration.

## 6. Cross-Node vs Within-Node Scaling

The Poisson scripts supply a two-node hostfile and request slot mapping and core binding; the thermal2 script launches `mpirun -np` without an explicit mapping report. The retained logs do not record process-to-node placement for any of these jobs. The thermal2 log is available at `results/mpi/thermal2/dsa5208_mpi_thermal2.o1495684`, but its `CN-115` component warnings do not identify solver-rank locations. Neither the allocation nor the launch options establish actual node occupancy.

## 7. Reproducibility

The Poisson PBS jobs generate results files named:
```
mpi_2d<size>_p<ranks>_<JOBID>.csv
```

The thermal2 job uses `results/mpi/thermal2/thermal2_p<ranks>_<JOBID>.csv`. Record job ID, allocation, module versions, and process mapping when available; the retained logs do not provide the latter.

## 8. Recorded Results and Limitations

The selected 512² and 1024² campaigns each contain eight rank settings and ten records per setting. All 160 selected records have `converged=1`; the report recomputes speedup and efficiency from the mean solver-only time. The scaling figure is `docs/task_b/mpi_scaling.svg`.

The later thermal2 campaign (PBS `1495684.stdct-mgmt-02`) contains five rank settings with four untimed warmups and ten timed solves per setting. All 50 CSV records converged to true relative residual below `1e-8`. Its summary table and Figures 2–3 are in `REPORT_TASK_B.md`; `scripts/analyze_mpi_thermal2.py` audits the CSVs and regenerates the figures. The retained execution log confirms completion of all five configurations but does not give rank-to-node placement.

- Latest correctness job: 10/10 CTest targets passed; partition, SpMV, and CG executables each passed at 1, 2, and 4 ranks.
- Scaling jobs: PBS `1482451.stdct-mgmt-02` (512²) and `1482491.stdct-mgmt-02` (1024²).
- Full results and cross-node discussion: `REPORT_TASK_B.md` and `REPORT_TASK_B_ZH.md`.

- Cross-solver comparisons with serial timing are omitted from the final tables; serial uses `std::chrono::steady_clock` while MPI uses `MPI_Wtime`.
- MPI_Allgatherv replicates a full input vector on each rank; each rank contributes its local vector segment. Collective cost was not separately measured.
- The thermal2 log contains ignored OpenMPI `libpbs.so.0` component-load warnings; their performance effect is unknown.
- Results on different compute nodes are not directly comparable.
- No independent per-rank timing breakdown; load balance across ranks is not measured separately.
