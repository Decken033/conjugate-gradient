# Conjugate Gradient – Ops Profiling Edition

This folder is based on the uploaded project source. It adds per-operation timing to the existing MPI-only, OpenMP-only and Hybrid MPI+OpenMP implementations without changing the CG recurrence, operation order, matrix partitioning, scheduling policy, MPI communication pattern or vector-update formulas.

## What changed

- Added `include/common/ops_profiler.hpp` (lightweight scoped timers).
- Added timers inside the existing MPI, OpenMP and Hybrid Ops functions.
- Added timers for solver-side collective checks and total CG solve time.
- Existing benchmark runners now write a profiling sidecar CSV in addition to the normal benchmark CSV.
- Added `scripts/summarize_profiles.py` to aggregate MPI/Hybrid per-rank profiles.
- The main `CMakeLists.txt` is the project's Hybrid-enabled CMake configuration. The previous main CMake file is retained as `CMakeLists_original.txt`.

No `build*` directories or benchmark result data are included. `results/` is intentionally empty except for `.gitkeep`.

## Build on Vanda

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

## Run

Create output folders:

```bash
mkdir -p results/profiling/{mpi,openmp,hybrid}
```

MPI-only (example: 8 ranks):

```bash
mpirun -np 8 ./build-profile-vanda/run_mpi_benchmark \
  2d 512 10 results/profiling/mpi/mpi_2d512_r8.csv 5
```

OpenMP-only (8 threads):

```bash
export OMP_NUM_THREADS=8 OMP_PROC_BIND=true OMP_PLACES=cores
./build-profile-vanda/run_openmp_benchmark \
  2d 512 8 static 0 10 results/profiling/openmp/openmp_2d512_t8.csv 5
```

Hybrid (4 ranks × 2 threads):

```bash
export OMP_NUM_THREADS=2 OMP_PROC_BIND=true OMP_PLACES=cores
mpirun -np 4 --map-by slot:PE=2 --bind-to core \
  ./build-profile-vanda/run_hybrid_benchmark \
  2d 512 2 static 0 10 results/profiling/hybrid/hybrid_2d512_r4_t2.csv 5
```

Hybrid (2 ranks × 4 threads):

```bash
export OMP_NUM_THREADS=4 OMP_PROC_BIND=true OMP_PLACES=cores
mpirun -np 2 --map-by slot:PE=4 --bind-to core \
  ./build-profile-vanda/run_hybrid_benchmark \
  2d 512 4 static 0 10 results/profiling/hybrid/hybrid_2d512_r2_t4.csv 5
```

## Profiling outputs

The normal benchmark CSV is unchanged. A sidecar profile is automatically generated:

- OpenMP: `<benchmark.csv>.profile.csv`
- MPI: `<benchmark.csv>.profile.rank0.csv`, `<benchmark.csv>.profile.rank1.csv`, ...
- Hybrid: one `.profile.rankN.csv` file per rank.

For MPI/Hybrid, aggregate rank files after the run:

```bash
python3 scripts/summarize_profiles.py \
  results/profiling/hybrid/hybrid_4x2_summary.csv \
  'results/profiling/hybrid/*.profile.rank*.csv'
```

See `docs/profiling/OPS_PROFILING_GUIDE.md` for operation definitions, interpretation and cautions.

## V1 Memory Reuse + Rank-level Profiling

For the combined V1 workspace reuse and per-rank profiling build, see `README_V1_PROFILING.md`. This version reuses Hybrid SpMV communication metadata and buffers while preserving per-rank profiling CSV output.
