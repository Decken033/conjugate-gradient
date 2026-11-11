# Hybrid MPI + OpenMP Run Guide (Ubuntu / WSL2)

Run commands from the root of the complete `conjugate-gradient-main` repository, not from this handoff subfolder.

## 1. Prerequisites

The tested local environment reported by the user:

- Ubuntu 24.04 on WSL2
- GCC 13.3.0
- OpenMPI 4.1.6
- CMake 3.28.3
- OpenMP 4.5 detected by CMake

Check tools:

```bash
g++ --version
mpicxx --showme
mpirun --version
cmake --version
```

## 2. Configure

Use the Hybrid-enabled CMake configuration. In the reported workflow, `CMakeLists_hybrid.txt` was copied to the canonical `CMakeLists.txt` after backing up the original.

```bash
cp CMakeLists.txt CMakeLists_original_backup.txt   # only if backup does not already exist
cp CMakeLists_hybrid.txt CMakeLists.txt

cmake -S . -B build-hybrid \
  -DCMAKE_BUILD_TYPE=Release \
  -DCG_ENABLE_MPI=ON \
  -DCG_ENABLE_OPENMP_GPU=OFF \
  -DBUILD_TESTING=ON
```

`-S .` selects the current directory as the source tree; `-B build-hybrid` places generated build files in a separate directory. `Release` enables compiler optimizations, MPI is enabled, GPU offload is disabled, and tests are enabled.

## 3. Build

```bash
cmake --build build-hybrid --parallel 4
```

`--parallel 4` allows up to four build jobs to compile/link concurrently. It affects build time only, not the number of MPI ranks or OpenMP threads used when the program runs.

## 4. Correctness tests

```bash
mpirun -np 2 ./build-hybrid/test_hybrid_ops
mpirun -np 2 ./build-hybrid/test_hybrid_cg
mpirun -np 4 ./build-hybrid/test_hybrid_cg
```

The reported run passed all three commands. The CG test output reported 29 iterations, relative residual about `6.53759e-09`, and solution error about `9.97462e-10`; its test configuration used 2 OpenMP threads per rank.

## 5. Hybrid benchmark CLI

The runner usage is:

```text
mpirun -np <ranks> ./build-hybrid/run_hybrid_benchmark \
  <2d|3d|thermal2> <grid_size|matrix.mtx> <threads_per_rank> \
  <static|dynamic|guided> <chunk_size> <repeats> <output.csv> [warmups]
```

Example, 2D Poisson grid 256, 2 MPI ranks, 4 OpenMP threads per rank, Static scheduling, 5 measured runs, 1 warm-up:

```bash
mkdir -p results/hybrid
OMP_PROC_BIND=true OMP_PLACES=cores \
mpirun -np 2 ./build-hybrid/run_hybrid_benchmark \
  2d 256 4 static 0 5 results/hybrid/hybrid_2d256_r2_t4_static.csv 1
```

The output CSV is written by rank 0. Use a unique filename for each configuration because the CSV writer opens the output in truncate mode.

## 6. Results and reproducibility

CSV columns include solver, matrix name, matrix size, nonzero count, ranks, threads, run index, iterations, convergence flag, solver time, true relative residual, solution error, schedule, and chunk size.

For fair comparisons, keep matrix, tolerance, max iterations, warm-up count, repeat count, and CPU allocation consistent. On WSL2, timings are exploratory and can be affected by host scheduling/background load. Use repeated measurements and report mean plus median/standard deviation. Final HPC performance claims should be measured on the target cluster with authorized, controlled resources.
