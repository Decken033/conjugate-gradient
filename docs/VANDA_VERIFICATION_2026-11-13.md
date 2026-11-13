# Vanda Standalone Verification — 13 November 2026

## Scope

This document records independent compilation, correctness testing,
benchmark smoke tests, and GPU-offloading verification of the standalone
Task A, Task B, and Task C implementations on NUS Vanda.

Verification was performed after repository restructuring. Historical
benchmark results are retained separately and were not reproduced in full.

## Environment

CPU compute node: CN-148
GPU compute node: GN-A40-034
GPU hardware: NVIDIA A40 (compute capability 8.6)

CPU compiler: GCC 13.2.0
CPU MPI: OpenMPI 4.1.6
CMake: 3.27.6
GPU compiler: NVHPC 24.7, CUDA 12.3

Tests were executed inside allocated Vanda compute jobs, not login nodes.

## Standalone CPU verification

All four standalone CPU builds completed successfully:

| Package | CTests | Result |
|---|---:|---|
| Task A (Serial/OpenMP, GPU disabled) | 7 | PASS |
| Task B (MPI) | 11 | PASS |
| Task C handoff (MPI/OpenMP) | 3 | PASS |
| Task C V1 profile (MPI/OpenMP) | 3 | PASS |
| **Total** | **24** | **PASS** |

All four packages were configured and built independently.

## CPU benchmark smoke tests

Representative execution results:

| Implementation | Workload | Configuration | Iterations | Result |
|---|---|---|---:|---|
| Task A Serial | Poisson 2D 128 | One CPU | 231 | Converged |
| Task A OpenMP | Poisson 2D 128 | Four threads | 231 | Converged |
| Task B MPI | Poisson 2D 128 | Two ranks | 231 | Converged |
| Task C handoff | Poisson 2D 64 | Two ranks, four threads/rank | 122 | Converged |
| Task C V1 | Poisson 2D 64 | Two ranks, four threads/rank | 122 | Converged |

The Task A Serial, OpenMP, and Task B MPI 2D128 runs reported a
true relative residual of approximately 9.91508e-09.

The Task C handoff and V1 2D64 runs reported a true relative residual
of approximately 8.71368e-09.

Task C V1 profiling CSV files and rank aggregation were also verified.

The Vanda hybrid smoke-test timings were much slower than the original
WSL2 observations. Their numerical correctness was confirmed, but these
timings must not be treated as a reproduction of historical speedups.

## Task A GPU build

GPU compilation succeeded using NVHPC 24.7 on the allocated A40 node.

The local NVHPC/GCC installation required an explicit GCC-toolchain
override to resolve GCC startup object paths.

From task_a, with the appropriate NVHPC, GCC, and CMake modules loaded:

```bash
cmake -S . -B build-gpu-fixed \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=nvc++ \
  -DCMAKE_CXX_FLAGS="--gcc-toolchain=$(which gcc)" \
  -DCG_ENABLE_OPENMP_GPU=ON \
  -DBUILD_TESTING=ON

cmake --build build-gpu-fixed --parallel 4
```

Both test_gpu_cg and run_gpu_benchmark were built successfully.

The toolchain override is specific to the observed Vanda environment.
A different installation may not require it.

## Task A GPU correctness

The following command passed:

```bash
OMP_TARGET_OFFLOAD=MANDATORY \
  ctest --test-dir build-gpu-fixed \
  -R '^gpu_cg$' --output-on-failure --no-tests=error -V
```

Result: 1/1 GPU CTests passed.

| Problem | GPU iterations | CPU iterations | CPU-GPU difference |
|---|---:|---:|---:|
| Poisson 2D 16 | 29 | 29 | 5.2231e-16 |
| Poisson 2D 64 | 122 | 122 | 8.36883e-16 |
| Poisson 3D 8 | 19 | 19 | 2.39919e-16 |

## GPU benchmark smoke test

Command:

```bash
OMP_TARGET_OFFLOAD=MANDATORY \
  ./build-gpu-fixed/run_gpu_benchmark \
  2d 128 2 results/verification/gpu_2d128.csv 1
```

Both measured runs converged in 231 iterations.

| Run | Solve time (seconds) | True relative residual |
|---|---:|---:|
| 1 | 0.014404213 | 9.91508456e-09 |
| 2 | 0.014500410 | 9.91508494e-09 |

These are smoke-test timings, not a complete controlled scaling study.

## Actual GPU hardware-offloading evidence

A separate diagnostic run used:

```bash
OMP_TARGET_OFFLOAD=MANDATORY NVCOMPILER_ACC_NOTIFY=1 \
  ./build-gpu-fixed/run_gpu_benchmark \
  2d 16 1 results/verification/gpu_offload_check.csv 1
```

NVHPC reported actual CUDA kernel launches on device 0 from
src/gpu/cg_gpu.cpp, including:

- Sparse CSR matrix-vector multiplication
- Dot-product computation
- Solution and residual update
- Search-direction update

The diagnostic benchmark converged in 29 iterations with a
true relative residual of approximately 6.53759e-09.

This confirms execution of GPU kernels rather than merely successful
compilation or an assumed device offload.

## Limitations and provenance

1. This was functional verification and representative benchmarking,
   not full reproduction of the historical performance campaigns.
2. The exact profiled Task C V0 source was not recovered. The unprofiled
   handoff is not an interchangeable V0 profiling baseline.
3. The recorded Task C handoff/V1 experiments do not establish execution
   on the real-world thermal2 matrix. CLI support alone is not test evidence.
4. Historical timing comparisons depend on original hardware, process
   placement, toolchains, workloads, and measurement protocols.
5. Source-level numerical algorithms were not modified as part of
   this documentation or verification work.
6. The observed tests do not certify every possible input, scaling
   configuration, or MPI failure path.

## Conclusion

All 24 registered CPU tests and the Task A GPU correctness test passed.
All four standalone CPU packages built independently. Representative
CPU/GPU benchmark execution, convergence, CSV generation, Task C V1
profiling, and actual A40 CUDA kernel launches were verified.

The tested standalone implementations are functionally operational on
NUS Vanda. Historical performance reproduction remains a separate scope.
