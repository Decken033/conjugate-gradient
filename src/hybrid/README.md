# Hybrid MPI + OpenMP implementation

This directory adds a CPU Hybrid CG path without changing the existing MPI-only or OpenMP-only implementations.

## Architecture

- `hybrid_ops.cpp`: MPI collectives are called by the rank's calling thread. Local CSR row loops, local dot-product accumulation, vector updates, and true-residual subtraction use OpenMP.
- `hybrid_solver.cpp`: the same CG recurrence and true-residual/restart safeguards as the existing MPI and OpenMP solvers.
- `benchmark_hybrid.cpp`: warm-ups, timed repeated solves, max-rank elapsed time, and solution-error validation.
- `benchmark_hybrid_csv.cpp`: CSV output with ranks, threads per rank, schedule, and chunk size.
- `run_hybrid_benchmark.cpp`: MPI-aware command-line entry point. Launch it with `mpirun`; the `threads` argument is per rank.

The existing `mpi_partition.cpp` is reused. Each rank keeps its local CSR rows and local vectors. `hybrid_csr_matvec` performs `MPI_Allgatherv` first, then OpenMP-parallelizes the local CSR rows. `hybrid_dot` performs an OpenMP reduction within each rank, then an `MPI_Allreduce` across ranks.

## Build

From the project root:

```bash
cmake -S . -B build-hybrid -DCMAKE_BUILD_TYPE=Release -DCG_ENABLE_MPI=ON
cmake --build build-hybrid --parallel 4
```

## Correctness tests

```bash
mpirun -np 2 ./build-hybrid/test_hybrid_ops
mpirun -np 2 ./build-hybrid/test_hybrid_cg
```

Or run the CTest entries:

```bash
ctest --test-dir build-hybrid --output-on-failure -R hybrid
```

## Benchmark

```bash
mkdir -p results/hybrid
OMP_PROC_BIND=true OMP_PLACES=cores \
mpirun -np 2 ./build-hybrid/run_hybrid_benchmark \
  2d 64 4 static 0 5 results/hybrid/hybrid_2d64_r2_t4.csv 1
```

Arguments:

`<2d|3d|thermal2> <grid_size|matrix.mtx> <threads_per_rank> <static|dynamic|guided> <chunk_size> <repeats> <output.csv> [warmups]`

For a Matrix Market input, use for example:

```bash
mpirun -np 2 ./build-hybrid/run_hybrid_benchmark \
  thermal2 data/suitesparse/thermal2/thermal2.mtx 4 static 0 5 \
  results/hybrid/thermal2_r2_t4.csv 1
```

## MPI thread level

The runner requests `MPI_THREAD_FUNNELED`. MPI calls occur outside OpenMP parallel regions, after the local OpenMP work has completed. The MPI implementation must provide at least `MPI_THREAD_FUNNELED`.

## Notes

- `threads` is the OpenMP thread count **per MPI rank**. Total worker threads are approximately `ranks * threads`.
- Keep the total worker count within the CPU resources actually available to the process/container.
- Floating-point reduction order differs from serial/MPI-only reductions, so compare residuals with tolerances rather than bitwise equality.
- This first Hybrid implementation intentionally preserves the existing all-gather communication strategy. It does not yet implement halo exchange or communication/computation overlap.
