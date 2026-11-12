# MPI / OpenMP / Hybrid Ops Profiling Guide

## Purpose

This profiling build adds operation-level wall-clock timers to the existing MPI, OpenMP and Hybrid implementations. The numerical algorithms, operation order, scheduling choices, MPI collectives and vector-update formulas are unchanged. The instrumentation is observational only.

## Build on Vanda

Use the Hybrid-enabled CMake file supplied in this project (it is installed as the main `CMakeLists.txt` in this package):

```bash
module purge
module load CMake/3.27.6-GCCcore-13.2.0
module load openmpi/gcc/64/4.1.8

cmake -S . -B build-profile-vanda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCG_ENABLE_MPI=ON \
  -DCG_ENABLE_OPENMP_GPU=OFF \
  -DBUILD_TESTING=ON
cmake --build build-profile-vanda --parallel 4
```

## Run examples: 2D Poisson 512, 5 warmups, 10 measured runs

Create output directories first:

```bash
mkdir -p results/profiling/mpi results/profiling/openmp results/profiling/hybrid
```

MPI-only, for example 8 ranks:

```bash
mpirun -np 8 ./build-profile-vanda/run_mpi_benchmark \
  2d 512 10 results/profiling/mpi/mpi_2d512_r8.csv 5
```

OpenMP-only, 8 threads:

```bash
export OMP_NUM_THREADS=8 OMP_PROC_BIND=true OMP_PLACES=cores
./build-profile-vanda/run_openmp_benchmark \
  2d 512 8 static 0 10 results/profiling/openmp/openmp_2d512_t8.csv 5
```

Hybrid, 4 ranks x 2 threads:

```bash
export OMP_NUM_THREADS=2 OMP_PROC_BIND=true OMP_PLACES=cores
mpirun -np 4 --map-by slot:PE=2 --bind-to core \
  ./build-profile-vanda/run_hybrid_benchmark \
  2d 512 2 static 0 10 results/profiling/hybrid/hybrid_2d512_r4_t2.csv 5
```

Hybrid, 2 ranks x 4 threads:

```bash
export OMP_NUM_THREADS=4 OMP_PROC_BIND=true OMP_PLACES=cores
mpirun -np 2 --map-by slot:PE=4 --bind-to core \
  ./build-profile-vanda/run_hybrid_benchmark \
  2d 512 4 static 0 10 results/profiling/hybrid/hybrid_2d512_r2_t4.csv 5
```

## Output files

Each normal benchmark CSV remains unchanged. A profiling sidecar is also written:

- OpenMP: `<output.csv>.profile.csv`
- MPI: `<output.csv>.profile.rank<RANK>.csv` (one file per rank)
- Hybrid: `<output.csv>.profile.rank<RANK>.csv` (one file per rank)

Profile columns:

`solver,rank,ranks,threads,operation,calls,total_seconds,avg_seconds`

Operations:

- `spmv_allgatherv`: time inside the Allgatherv call only.
- `spmv_compute`: local CSR SpMV compute region.
- `dot_compute`: local serial/OpenMP dot-product accumulation.
- `mpi_allreduce`: dot-product global Allreduce time.
- `solution_residual_update`: fused x/r update.
- `direction_update`: p update.
- `true_residual_subtract`: b-Ax subtraction.
- `collective_check`: solver-side collective validation Allreduce time (MPI/Hybrid).
- `solver_total`: full solve call, measured separately; it overlaps the operation timers and MUST NOT be added to them.

Warm-up runs are excluded. Profile totals accumulate across the measured runs. `avg_seconds` is total time divided by calls.

## MPI rank aggregation

MPI and Hybrid write one profile CSV per rank so that no extra MPI collective is inserted into the measured loop. After the job finishes, aggregate them:

```bash
python3 scripts/summarize_profiles.py \
  results/profiling/hybrid/hybrid_4x2_summary.csv \
  'results/profiling/hybrid/*.profile.rank*.csv'
```

The summary reports mean, maximum and minimum rank totals per operation. For parallel wall-clock analysis, inspect `max_rank_total_s`; collective timings include waiting time inside the MPI call and can therefore reflect rank imbalance as well as communication latency.

## Interpretation cautions

1. Do not sum `solver_total` with the other rows; it is an enclosing measurement.
2. `spmv_allgatherv` and `mpi_allreduce` are collective-call wall times, not pure network-transfer times.
3. MPI/Hybrid rank files should be compared using the slowest-rank time as the critical-path estimate, while mean-rank time helps expose imbalance.
4. OpenMP timings include the parallel region and its implicit completion synchronization.
5. Profiling adds timer overhead. Use it to understand phase proportions; use the normal benchmark CSV (same code, same build) for final headline runtime comparisons.
6. The normal benchmark CSV schema and numerical algorithm are unchanged.

The profiler also emits `other_uninstrumented`, computed as `solver_total` minus the explicitly instrumented operation timers. This is a residual bucket for scalar CG control work, vector/container allocation, and any work not assigned to a named timer. `solver_total` is an enclosing timer and is not additive with the operation rows; the named operation rows plus `other_uninstrumented` approximately reconcile to the solver total.
