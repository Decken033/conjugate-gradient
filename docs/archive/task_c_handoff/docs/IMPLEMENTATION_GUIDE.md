# Person C: Hybrid MPI + OpenMP implementation

## What is added

The Hybrid implementation is an additional execution path. It does not replace the existing MPI-only or OpenMP-only code.

### New headers

- `include/hybrid/hybrid_ops.hpp`
- `include/hybrid/hybrid_solver.hpp`
- `include/hybrid/hybrid_benchmark.hpp`

### New implementation files

- `src/hybrid/hybrid_ops.cpp`
- `src/hybrid/hybrid_solver.cpp`
- `src/hybrid/benchmark_hybrid.cpp`
- `src/hybrid/benchmark_hybrid_csv.cpp`
- `src/hybrid/run_hybrid_benchmark.cpp`

### New tests

- `tests/hybrid/test_hybrid_ops.cpp`
- `tests/hybrid/test_hybrid_cg.cpp`

The existing MPI partition implementation is reused; a separate Hybrid partitioner is not required because row-wise rank ownership is unchanged when OpenMP threads are added inside each rank.

## Execution model

For `P` MPI ranks and `T` OpenMP threads per rank:

1. MPI partitions the global CSR matrix by rows. Each rank owns its local CSR rows and local portions of `b`, `x`, `r`, and `p`.
2. For SpMV, all ranks call `MPI_Allgatherv` to replicate the current input vector. Each rank then uses an OpenMP row-parallel loop to compute only its local output rows.
3. For a dot product, each rank first performs an OpenMP reduction over its local vector elements. The calling thread then invokes `MPI_Allreduce` to combine the rank-local partial sums.
4. Vector updates (`x`, `r`, `p`) and true-residual subtraction are parallelized over local elements with OpenMP.
5. The outer CG iteration remains sequential because successive CG steps depend on the previous step's global scalar values.

MPI calls are outside OpenMP parallel regions. The runner requests `MPI_THREAD_FUNNELED`; the MPI implementation must provide at least that level.

## Build and run

```bash
cmake -S . -B build-hybrid -DCMAKE_BUILD_TYPE=Release -DCG_ENABLE_MPI=ON
cmake --build build-hybrid --parallel 4

mpirun -np 2 ./build-hybrid/test_hybrid_ops
mpirun -np 2 ./build-hybrid/test_hybrid_cg

mkdir -p results/hybrid
OMP_PROC_BIND=true OMP_PLACES=cores \
mpirun -np 2 ./build-hybrid/run_hybrid_benchmark \
  2d 64 4 static 0 5 results/hybrid/hybrid_2d64_r2_t4.csv 1
```

The command-line arguments are:

`<2d|3d|thermal2> <grid_size|matrix.mtx> <threads_per_rank> <static|dynamic|guided> <chunk_size> <repeats> <output.csv> [warmups]`

`threads_per_rank` is the OpenMP thread count inside each MPI process. Approximate total worker threads are `MPI ranks * threads_per_rank`.

## Numerical and performance notes

- OpenMP reductions can change floating-point summation order. Compare with tolerance-based checks, not bitwise equality.
- Benchmark time is the maximum solver elapsed time across ranks, so the slowest rank determines the reported parallel solve time.
- Matrix generation, matrix partitioning, RHS construction, warm-ups, solution-error calculation, and CSV output are outside the timed solver region.
- This is a correctness-first hybrid implementation retaining the current full-vector `MPI_Allgatherv` design. Halo exchange, nonblocking communication overlap, persistent parallel regions, and fused vector kernels are possible later optimizations.
- Local WSL timings are useful for correctness and exploratory scaling only; final HPC performance claims should be measured on the target cluster with controlled resources.
