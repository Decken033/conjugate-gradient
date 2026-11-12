> Historical context: source paths and unified-build commands describe the original revisions. For current independent builds and experiments, use the [standalone submission README](../../README.md). Removed paths and archived snapshots are recorded in docs/archive/cleanup_manifest.json (repository-relative). Historical results are not verification of the standalone packages.

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

The benchmark runner defaults to one warmup. The selected 2D Poisson, thermal2, 3D, and new 256-rank PBS logs record four warmups and ten timed runs per configuration. A recovered historical 3D PBS snapshot is archived at `docs/task_b/evidence/scripts/benchmark_mpi_scaling_3d.pbs`; its submission-time identity and exact byte content are not independently verifiable. `docs/archive/vanda/benchmark_mpi_3d_reproduce.pbs` is a separate reproduction proposal, not the historical submission.

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

For the new fixed-workload strong series, `T(1)` is the ten-run mean of job `1509066.stdct-mgmt-02` alone. For the new weak series, every rank owns `16,384` unknowns and full-solve weak efficiency is `E_weak(P) = T_weak(1) / T_weak(P)`, using the ten-run one-rank mean from job `1509067.stdct-mgmt-02`. The weak `T(P)/iterations` metric is descriptive and does not replace full-solve efficiency. Means, sample SD (`n-1`), minimum, maximum, and all ratios are calculated before display rounding. The new figures place MPI ranks at `log2(P)` coordinates, including the non-power-of-two strong settings.

## 4. Workloads

| Matrix | N | NNZ | Notes |
|---|---|---|---|
| 2D Poisson weak series 128x128 to 2048x2048 | 16,384 to 4,194,304 | 81,408 to 20,963,328 | 16,384 unknowns per rank at P = 1, 4, 16, 64, 256 |
| 2D Poisson 512×512 | 262,144 | 1,308,672 | Primary scaling experiment |
| 2D Poisson 1024×1024 | 1,048,576 | 5,238,784 | Secondary scaling experiment |
| 3D Poisson 4³ and 8³ | 64 and 512 | — | MPI correctness validation |
| 3D Poisson 40³, 64³, 102³ | 64,000; 262,144; 1,061,208 | 438,400; 1,810,432; 7,366,032 | MPI scaling CSVs at 1, 2, 4, 8, 16, 32, 36, 72 ranks; ten timed runs each |
| SuiteSparse thermal2 | 1,228,045 | 8,580,313 | Real-world MPI scaling experiment at 1, 2, 4, 8, 16 ranks |

## 5. Cluster Configuration

The settings below refer to the logged 2D Poisson and thermal2 campaigns. The retained 3D output and two new extension outputs provide separate job-specific environment evidence.

| Component | Value |
|---|---|
| Cluster | NUS Vanda HPC |
| Modules in logged 2D/thermal2 scaling jobs | `CMake/3.27.6-GCCcore-13.2.0` + `OpenMPI/4.1.6-GCC-13.2.0` |
| Compiler | GCCcore 13.2.0 |
| CMake build type | Release requested by the scaling scripts; historical cache is unavailable for thermal2 |
| MPI | Open MPI 4.1.6 in the earlier logged 2D/thermal2 jobs, recovered 3D output, and the new strong/weak outputs |
| Process binding | 2D Poisson scripts request `--bind-to core`; thermal2 script does not specify binding |

The 512 Poisson script requests:
```
#PBS -l select=2:ncpus=36:mpiprocs=36:mem=32gb
```
The 1024 Poisson and thermal2 scripts request 64 GB per node instead. Each two-node request specifies 36 MPI slots per node (72 allocated slots), which differs from the MPI process count launched for any one configuration.

## 6. Cross-Node vs Within-Node Scaling

The 2D Poisson scripts supply a two-node hostfile and request slot mapping and core binding; the thermal2 script launches `mpirun -np` without an explicit mapping report. The retained logs do not record process-to-node placement for any of these jobs. The thermal2 log is available at `results/mpi/thermal2/dsa5208_mpi_thermal2.o1495684`, but its `CN-115` component warnings do not identify solver-rank locations. Neither the allocation nor the launch options establish actual node occupancy.

## 7. Reproducibility

The Poisson PBS jobs generate results files named:
```
mpi_2d<size>_p<ranks>_<JOBID>.csv
```

The thermal2 job uses `results/mpi/thermal2/thermal2_p<ranks>_<JOBID>.csv`. Record job ID, allocation, module versions, and process mapping when available; the retained logs do not provide the latter.

The 3D results use `results/mpi/3d/mpi_3d<size>_p<ranks>_1502451.stdct-mgmt-02.csv`. The recovered `dsa5208_mpi_scaling_3d.o1502451` records its allocation, software versions, warmups, launch sequence, and completion. The recovered historical PBS snapshot records the requested resources and launch settings, but its submission-time identity, exact process placement, build provenance and scheduler exit status remain unverified. `docs/archive/vanda/benchmark_mpi_3d_reproduce.pbs` is a separate proposed rerun configuration.

The new results are in `results/mpi/extension/strong_1509066.stdct-mgmt-02/` and `results/mpi/extension/weak_1509067.stdct-mgmt-02/`, with root-level PBS outputs `dsa5208_mpi_strong_256.o1509066` and `dsa5208_mpi_weak_256.o1509067`. Each output records eight allocated hosts with 36 slots each (288 slots), the CMake and Open MPI modules, core binding maps, all four warmups and ten timed solves per configuration, and a completion marker. The hostfiles retain eight hosts for each job. The strong and weak jobs ran in overlapping periods and had different host sets; they are separate baselines. Earlier jobs also differ in allocation and run conditions, so absolute times across jobs are not a controlled comparison. Neither CSV set has per-collective timings, so no communication fraction or causal attribution can be derived.

## 8. Recorded Results and Limitations

The selected 512² and 1024² campaigns each contain eight rank settings and ten records per setting. All 160 selected records have `converged=1`; the report recomputes speedup and efficiency from the mean solver-only time. The scaling figure is `docs/task_b/mpi_scaling.svg`.

The later thermal2 campaign (PBS `1495684.stdct-mgmt-02`) contains five rank settings with four untimed warmups and ten timed solves per setting. All 50 CSV records converged to true relative residual below `1e-8`. Its summary table and Figures 2–3 are in `REPORT_TASK_B.md`; `scripts/analyze_mpi_thermal2.py` audits the CSVs and regenerates the figures. The retained execution log confirms completion of all five configurations but does not give rank-to-node placement.

The 3D Poisson CSV series has 24 expected files and 240 timed records. All records converged, have true relative residual at most `1e-8`, and have finite relative solution error below `1e-4`. Section 7.5 of the English report and Section 5.5 of the Chinese report contain its statistics; the recovered log is checked by `scripts/validate_task_b_report.py`.

The extension adds 10 strong CSVs and 5 weak CSVs, each with ten timed records. `scripts/analyze_mpi_scaling_extension.py` audits schema, inventory, run indices, matrix metadata, convergence, residual and solution error, and regenerates the summary CSVs and SVGs in `results/mpi/extension/analysis/`. The report includes all 15 configurations, means, sample SDs, extrema, scale ratios, iterations, mean seconds per iteration, and numerical maxima. The validation script checks all 60 selected CSVs (600 timed records), both PBS outputs, the generated summaries and SVG coordinates, both Markdown tables, and DOCX key numbers.

- Latest correctness job: 10/10 CTest targets passed; partition, SpMV, and CG executables each passed at 1, 2, and 4 ranks.
- Scaling jobs: PBS `1482451.stdct-mgmt-02` (512²) and `1482491.stdct-mgmt-02` (1024²).
- Full results and cross-node discussion: `REPORT_TASK_B.md`.

- Cross-solver comparisons with serial timing are omitted from the final tables; serial uses `std::chrono::steady_clock` while MPI uses `MPI_Wtime`.
- MPI_Allgatherv replicates a full input vector on each rank; each rank contributes its local vector segment. Collective cost was not separately measured.
- The thermal2 log contains ignored OpenMPI `libpbs.so.0` component-load warnings; their performance effect is unknown.
- Results on different compute nodes are not directly comparable.
- No independent per-rank timing breakdown; load balance across ranks is not measured separately.
