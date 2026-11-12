# Person B MPI CG Implementation Summary

**Date:** October 24, 2026  
**Status:** Code complete; Vanda build, correctness verification and both scaling campaigns completed

## Files Implemented

### Headers (include/)
- `mpi_partition.hpp` — MPIPartitionedMatrix, partition_matrix(), gather/scatter
- `mpi_ops.hpp` — mpi_csr_matvec(), mpi_dot(), mpi_norm2(), vector updates, true residual
- `mpi_solver.hpp` — MPICGResult, solve_mpi_cg()
- `mpi_benchmark.hpp` — MPIBenchmarkRecord, benchmark_mpi_cg(), write_mpi_benchmark_csv()

### Source (src/mpi/)
- `mpi_partition.cpp` (342 lines) — Row-wise CSR partitioning, rank 0 distribution, gather/scatter
- `mpi_ops.cpp` (186 lines) — Distributed SpMV with Allgatherv, dot products with Allreduce
- `mpi_solver.cpp` (239 lines) — Collective-safe CG iteration with distributed true residual
- `benchmark_mpi.cpp` (145 lines) — Timing with MPI_Wtime, MPI_Reduce(MAX) for max elapsed time
- `benchmark_mpi_csv.cpp` (108 lines) — CSV writer using the common 12 columns
- `run_mpi_benchmark.cpp` (234 lines) — CLI benchmark runner for 2d/3d/thermal2
- `README.md` — MPI source directory documentation

### Tests (tests/)
- `test_mpi_partition.cpp` (307 lines) — 7 partition correctness tests
- `test_mpi_spmv.cpp` (266 lines) — 7 SpMV vs serial reference tests
- `test_mpi_cg.cpp` (351 lines) — 8 CG convergence and consistency tests

### PBS Scripts (scripts/vanda/)
- `test_mpi_correctness.pbs` (68 lines) — Build and run all MPI tests (1, 2, 4 ranks)
- `benchmark_mpi_scaling.pbs` (109 lines) — 2D Poisson 512² strong-scaling (1–72 ranks)
- `benchmark_mpi_scaling_1024.pbs` (74 lines) — 2D Poisson 1024² strong-scaling (1–72 ranks)
- `mpi/README.md` — PBS script documentation

### Documentation (docs/task_b/)
- `implementation_design.md` (150 lines) — Architecture, partitioning, SpMV, modularity
- `benchmark_methodology.md` (120 lines) — Timing protocol, CSV schema, speedup definitions

### Top-level
- `REPORT_TASK_B.md` — Complete English Person B technical report
- `mpi_scaling.svg` — Mean-time speedup and efficiency plot for both scaling workloads
- `README.md` — Updated with Person B contribution and MPI build instructions
- `CMakeLists.txt` — Added MPI targets with CG_ENABLE_MPI option

**Total:** Approximately 3,600 lines of code, tests, scripts, and documentation at implementation time

## Key Design Decisions

1. **Row-wise 1D partitioning** with `ceil(N/P)` rows per rank, handles non-divisible N/P
2. **MPI_Allgatherv for SpMV** replicates full x vector (O(N) communication per call)
3. **Distributed true residual** computed via local SpMV + Allreduce for norm (no gather to rank 0)
4. **Collective safety** ensured via MPI_Allreduce for all decision points (convergence, breakdown)
5. **Modular interface** separates cg_mpi_ops (reusable by Person C) from cg_mpi solver
6. **Optional CMake integration** via CG_ENABLE_MPI flag (default ON, gracefully degrades if MPI absent)

## Technical Highlights

- All MPI collective operations are called collectively by all ranks
- `collective_check()` uses MPI_Allreduce(MPI_MAX) to ensure all ranks see errors
- Residual restart path (recursive vs true residual drift) handled identically on all ranks
- MPI_Wtime timing with MPI_Reduce(MPI_MAX) captures slowest rank (true wall-clock time)
- CSV output uses the common 12 columns (solver="mpi", threads=1)
- Allgatherv displacement overflow is checked; counts and other MPI calls still use `int` arguments

## Known Limitations

1. MPI_Allgatherv memory: O(N) per rank (10 MB for thermal2 N=1.2M)
2. Int displacement limit: N < ~2^31 (MPI 4.0 large-count would extend)
3. No sparse halo exchange optimization (communication stays O(N) at all ranks)
4. No load balancing by NNZ (row-wise partitioning may cause imbalance)
5. Uses `MPI_UNSIGNED_LONG` for `std::size_t` metadata on the tested LP64 environment; portability to other data models needs validation

## Testing Strategy

- Three MPI test executables with the checks listed below
- Test coverage: partition (7), SpMV (7), CG solver (8)
- Each test runs with 1, 2, and 4 ranks
- CTest integrates 4-rank runs as default
- PBS correctness script runs all tests on allocated Vanda node

## Benchmark Protocol

- Workloads: 2D Poisson 512² and 1024²
- Rank counts: 1, 2, 4, 8, 16, 32, 36, 72
- Per configuration: 4 warmups + 10 timed runs
- Reported in the final report: arithmetic mean solve time, sample standard deviation and observed min/max across ten timed runs; the job output also prints medians
- Speedup: MPI 1-rank relative in final results; no serial cross-comparison in the result tables

## Completed Vanda Verification

- Correctness job: CTest 10/10 passed; partition, SpMV and CG executables passed at 1, 2 and 4 ranks.
- 512² scaling: ranks 1, 2, 4, 8, 16, 32, 36 and 72; 10/10 converged records at every rank (80/80).
- 1024² scaling: the same eight rank settings; 10/10 converged records at every rank (80/80).
- Final report: tables and `docs/task_b/mpi_scaling.svg` contain the recomputed statistics and scaling analysis.

## Integration with Person A's Code

- **Shared-code change:** `include/csr_matrix.hpp` adds the MPI-aware size-type mapping needed by shared validation; serial/OpenMP solver algorithms were not changed for Task B
- **CMake integration:** new MPI targets behind an optional MPI flag
- **Shared libraries reused**: cg_ops, cg_poisson, cg_matrix_market
- **Compatible CSV fields:** common 12-column schema; timer differences still apply to cross-solver comparisons
- **Same numerical spec**: tolerance 1e-8, max_iterations 10000, x0=0, x*=ones

## Preparation for Person C (Hybrid MPI+OpenMP)

- `cg_mpi_ops` library provides reusable partitioning and communication
- `MPIPartitionedMatrix` can be combined with OpenMP-parallel local operations
- `mpi_dot()` and `mpi_norm2()` can be reused as-is
- Replace `mpi_csr_matvec()` with hybrid version: Allgatherv + OpenMP-parallel local SpMV
- Replace local vector updates with OpenMP-parallel versions
- `solve_mpi_cg()` signature can be preserved with modified ops

## Verification Checklist

- [x] Headers created with complete API documentation
- [x] Source implementations complete with error handling
- [x] Three test executables with comprehensive coverage
- [x] CMake integration with optional MPI flag
- [x] PBS scripts for correctness and scaling
- [x] Design and methodology documentation
- [x] Technical report (REPORT_TASK_B.md)
- [x] README updated with MPI instructions
- [x] Serial/OpenMP solver algorithms unchanged; shared CSR compatibility updated
- [x] Vanda build verification
- [x] Automated test execution
- [x] Strong-scaling benchmark experiments

---

**Further project work:** MPI 3D and MPI `thermal2` performance campaigns are not part of the recorded Person B results. Hybrid MPI+OpenMP work remains with Person C.
