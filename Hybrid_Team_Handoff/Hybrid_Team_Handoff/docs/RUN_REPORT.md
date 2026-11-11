# Hybrid MPI + OpenMP — Current Run Report

**Environment:** Ubuntu 24.04 on WSL2; GCC 13.3.0; OpenMPI 4.1.6; CMake 3.28.3; OpenMP 4.5 detected.

**Status:** Hybrid build and correctness tests passed. Exploratory Hybrid scheduling benchmarks completed for 2D Poisson grids 64 and 256. The planned 2D-512 comparison among MPI-only, OpenMP-only, and Hybrid has not yet been run.

## 1. Build and correctness validation

CMake configuration detected OpenMP 4.5 and MPI 3.1 and enabled the Hybrid targets. The full build reached 100%, including `run_hybrid_benchmark`, `test_hybrid_ops`, and `test_hybrid_cg`.

Reported test outcomes:

| Test | Configuration | Result |
|---|---|---|
| Hybrid Operations | 2 MPI ranks | PASS: SpMV and distributed dot product |
| Hybrid CG | 2 MPI ranks, 2 threads/rank | PASS; 29 iterations |
| Hybrid CG | 4 MPI ranks, 2 threads/rank | PASS; 29 iterations |

The two CG test runs reported relative residual `6.53759e-09` and solution error `9.97462e-10`.

## 2. Scheduling benchmark: 2D grid 64

Configuration: 2 MPI ranks × 4 OpenMP threads/rank; 4,096 rows; 20,224 nonzeros; 5 measured runs; 1 warm-up; tolerance `1e-8` as configured by the benchmark runner.

| Schedule / file | Run times (seconds) | Mean (seconds) |
|---|---|---:|
| Static, first file | 0.0224596, 0.0198081, 0.0279128, 0.0203398, 0.0219383 | 0.022492 |
| Static, second file | 0.0197489, 0.0182989, 0.0191142, 0.0194771, 0.0205222 | 0.019432 |
| Dynamic, chunk 0 | 0.0272446, 0.0284840, 0.0263563, 0.0276961, 0.0286977 | 0.027696 |
| Guided, chunk 4 | 0.0203611, 0.0189375, 0.0171316, 0.0179849, 0.0330647 | 0.021496 |

All runs reported 122 iterations, convergence, residual approximately `8.71368e-09`, and solution error approximately `3.09881e-09`.

The two Static series were separate executions; their means differ, indicating run-to-run environmental variation. Guided had one notably slower run. Treat these small-grid results as exploratory rather than definitive.

## 3. Scheduling benchmark: 2D grid 256

Configuration: 2 MPI ranks × 4 OpenMP threads/rank; 65,536 rows; 326,656 nonzeros; 5 measured runs; 1 warm-up.

| Schedule | Run times (seconds) | Mean (seconds) | Median (seconds) |
|---|---|---:|---:|
| Static, chunk 0 | 0.138653, 0.144682, 0.192681, 0.177222, 0.163235 | 0.163294 | 0.163235 |
| Dynamic, chunk 0 | 0.632870, 0.616740, 0.617022, 0.629434, 0.597982 | 0.618810 | 0.617022 |
| Guided, chunk 0 | 0.138263, 0.136679, 0.139256, 0.137948, 0.139459 | 0.138321 | 0.138263 |

All runs reported 454 iterations, convergence, relative residual approximately `9.91666e-09`, and solution error approximately `1.18589e-08`.

For these measurements, Guided's mean runtime was about 15.3% lower than Static's mean. Dynamic's mean was about 3.79 times Static's mean. These observations apply to this specific regular Poisson matrix, WSL2 environment, 2-rank/4-thread configuration, and default chunk settings; they should not be generalized to irregular sparse matrices or HPC nodes without further tests.

## 4. Preliminary interpretation

- Correctness appears stable across the tested scheduling modes: iteration counts and accuracy metrics are consistent.
- Dynamic with default chunk size is substantially slower in the 2D-256 experiment. A likely explanation is scheduling overhead from repeatedly dispatching small row chunks; this is a hypothesis, not a measured breakdown.
- Guided is fastest and most repeatable in the 2D-256 measurements.
- Static's 2D-256 runs show more variability than Guided. Host/WSL scheduling and background load may contribute; more repetitions are needed.

## 5. Next experiments agreed as the next stage

The next baseline is 2D Poisson grid 512, with identical matrix, tolerance, warm-ups, repeats, and CPU budget across:

1. MPI-only: 8 ranks × 1 thread.
2. OpenMP-only: 1 process × 8 threads.
3. Hybrid: 2 ranks × 4 threads.

After collecting those baselines, profile or instrument phase-level timings (SpMV communication, local SpMV, dot/reduction, vector updates) before modifying Hybrid kernels. Candidate optimizations include tuning Dynamic chunk size, reducing repeated OpenMP region overhead, improving communication strategy, and fusing vector operations. None of these code-level optimizations is claimed as completed in this report.
