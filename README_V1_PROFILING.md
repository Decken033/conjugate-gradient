# V1 Memory Reuse + Per-Rank Ops Profiling

This project combines the V1 Hybrid Memory Reuse implementation with the existing per-operation profiling instrumentation.

## What V1 changes

- `recvcounts` and `displs` are built once in `make_hybrid_workspace()` per CG solve.
- `x_global` is allocated once and reused for each SpMV Allgatherv.
- `spmv_output` is allocated once and overwritten by each SpMV.
- The Hybrid CG solver passes one `HybridWorkspace` through all iterations and true-residual checks.
- The compatibility SpMV overload remains available, but the CG solver uses the reusable-workspace overload.

## What profiling records

Each MPI rank writes its own sidecar file:

`<benchmark.csv>.profile.rank<RANK>.csv`

The profile rows include:

- `spmv_allgatherv`
- `spmv_compute`
- `dot_compute`
- `mpi_allreduce`
- `solution_residual_update`
- `direction_update`
- `true_residual_subtract`
- `collective_check`
- `workspace_init` (V1-specific)
- `solver_total`
- `other_uninstrumented`

The profile is accumulated over measured runs only; warm-up solves are excluded. `total_seconds` is the accumulated time across all measured runs on that rank, `calls` is the total call count, and `avg_seconds` is total/calls.

For MPI collectives, the measured duration is the rank-local wall time spent inside the collective call, including any waiting for other ranks. It is not pure network-transfer time.

## Build on Vanda

```bash
module purge
module load CMake/3.27.6-GCCcore-13.2.0
module load openmpi/gcc/64/4.1.8

cmake -S . -B build-v1-profile \
  -DCMAKE_BUILD_TYPE=Release \
  -DCG_ENABLE_MPI=ON \
  -DCG_ENABLE_OPENMP_GPU=OFF \
  -DBUILD_TESTING=ON
cmake --build build-v1-profile --parallel 4
```

## Correctness tests

```bash
mpirun -np 2 ./build-v1-profile/test_hybrid_ops
mpirun -np 2 ./build-v1-profile/test_hybrid_cg
mpirun -np 4 ./build-v1-profile/test_hybrid_cg
```

## Run the requested 2 MPI ranks x 4 OpenMP threads experiment

Run inside a PBS allocation matching 8 CPU cores, 2 MPI processes and 4 OpenMP threads per process:

```bash
mkdir -p results/v1-profile
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=true
export OMP_PLACES=cores

mpirun -np 2 --map-by slot:PE=4 --bind-to core \
  ./build-v1-profile/run_hybrid_benchmark \
  2d 512 4 static 0 10 \
  results/v1-profile/hybrid_2d512_r2_t4_v1_profile.csv 5
```

Expected outputs:

- `hybrid_2d512_r2_t4_v1_profile.csv`: unchanged benchmark result format.
- `hybrid_2d512_r2_t4_v1_profile.csv.profile.rank0.csv`
- `hybrid_2d512_r2_t4_v1_profile.csv.profile.rank1.csv`

Each rank profile contains the timings recorded by that rank. It does not contain per-OpenMP-thread rows; the Ops timers wrap the complete OpenMP parallel loop from the calling MPI rank's perspective.

## Summarize rank profiles

```bash
python3 scripts/summarize_profiles.py \
  results/v1-profile/hybrid_2d512_r2_t4_rank_summary.csv \
  'results/v1-profile/*.profile.rank*.csv'
```

The summary reports mean, maximum and minimum rank totals per operation. Use the per-rank files as the source of truth when investigating rank imbalance.

## V0 vs V1 comparison

Use the same node, PBS allocation, compiler/MPI modules, rank/thread binding, matrix, schedule, warm-ups and repeat count. V0 and V1 should be run separately with the same profiling build methodology. Compare:

1. `workspace_init` to see the one-time setup cost introduced/recorded by V1.
2. `spmv_allgatherv` to verify that communication algorithm and call count remain unchanged.
3. `spmv_compute` and `solver_total` to assess the effect of reusing `x_global` and `spmv_output`.
4. `calls` to confirm that iteration and operation counts remain consistent.

Memory reuse is intended to reduce allocation overhead; it does not reduce Allgatherv payload size or the number of collective calls.
