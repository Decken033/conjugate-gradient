# DSA5208 — Serial and OpenMP CG Benchmark Methodology

## 1. Experimental Platform

Experiments were conducted on the NUS Vanda HPC cluster using PBS batch jobs.

The two principal strong-scaling experiments were executed on separate
compute nodes:

| Configuration | 512 x 512 | 1024 x 1024 |
|---|---|---|
| PBS job ID | 1471134 | 1471153 |
| Compute node | CN-036 | CN-015 |
| PBS queue | auto | auto |
| Requested nodes | 1 | 1 |
| Requested CPU cores | 72 | 72 |
| Requested memory | 8 GB | 16 GB |
| Walltime limit | 30 minutes | 60 minutes |

The PBS scripts loaded the CMake/3.27.6-GCCcore-13.2.0 module.
The Vanda CMakeCache.txt confirmed GCC 13.2.0, CMake Release mode,
and compiler flags -O3 -DNDEBUG. No additional global C++ flags
were recorded.

## 2. Benchmark Matrices

The primary performance experiments used sparse symmetric positive-definite
matrices generated from the two-dimensional Poisson problem.

| Matrix | Unknowns (N) | Nonzeros (NNZ) |
|---|---:|---:|
| Poisson 2D 512 x 512 | 262,144 | 1,308,672 |
| Poisson 2D 1024 x 1024 | 1,048,576 | 5,238,784 |

Additional numerical validation was conducted using a 3D Poisson matrix
and the SuiteSparse thermal2 matrix.

For thermal2, the right-hand side was constructed as b = A * 1,
rather than using the dataset's original right-hand-side file.

## 3. Strong-Scaling Protocol

For each matrix, the serial CG solver was benchmarked first.
The OpenMP solver was then evaluated using:

1, 2, 4, 8, 16, 32, 36, and 72 threads.

All OpenMP strong-scaling configurations used static scheduling
with chunk_size = 0.

Each configuration performed:

- 4 warmup runs;
- 10 measured runs;
- a median runtime calculation using the solve_seconds CSV field.

The serial and OpenMP configurations for each matrix were executed
within the same PBS job. The two matrix sizes were benchmarked in
separate jobs.

The OpenMP environment was configured using:

    OMP_PROC_BIND=close
    OMP_PLACES=cores
    OMP_DYNAMIC=FALSE

## 4. Performance Metrics

Serial-relative speedup:

    Speedup_serial(p) = T_serial / T_OpenMP(p)

OpenMP-relative speedup:

    Speedup_OpenMP(p) = T_OpenMP(1) / T_OpenMP(p)

Parallel efficiency:

    Efficiency(p) = Speedup_OpenMP(p) / p * 100%

Runtime comparisons use median measured solve times.

## 5. Numerical Correctness

Numerical validation recorded the number of CG iterations,
convergence status, true relative residual, and relative solution error.

The target true relative residual tolerance was 1e-8.

Validation covered 2D Poisson, 3D Poisson, and SuiteSparse thermal2
matrices. All selected runs in the consolidated correctness table
converged.

Serial and OpenMP thermal2 runs required slightly different iteration
counts (4333 and 4331 respectively), while both satisfied the specified
residual tolerance.

## 6. Scheduling Investigation

OpenMP scheduling experiments evaluated static, dynamic, and guided
policies, as well as dynamic scheduling with several chunk sizes.

The experiments used the Poisson 2D 512 x 512 matrix and eight threads.

The final scheduling experiment repeated the static baseline at the
beginning and end of the job. Its median execution time increased
from approximately 0.992 s to 2.391 s, a factor of 2.410.

This indicates substantial within-job timing variation. Therefore,
fine-grained comparisons between scheduling policies and chunk sizes
should not be interpreted as controlled performance rankings.

## 7. Experimental Limitations

The two main matrix sizes were benchmarked on different Vanda compute
nodes, so differences in execution conditions may affect comparisons.

Thread configurations were executed sequentially in each PBS job.
The experiments did not randomize configuration order.

Although multiple measurements and warmup runs were used, shared-cluster
execution conditions may introduce runtime variability.

Both serial and OpenMP benchmarks use C++17 std::chrono::steady_clock.
Timing starts immediately before the complete CG solver call and stops
immediately afterward. Matrix construction, warmups, progress reporting,
post-solve solution-error validation, and CSV reporting are excluded.
Internal solver computation and synchronization are included.

## 8. Reproducibility Artifacts

The experiment archive preserves:

- Original per-run benchmark CSV files;
- PBS job scripts;
- PBS execution logs;
- Consolidated scaling, correctness, and scheduling CSV summaries;
- Report figures and summary tables.

The principal scaling scripts are:

- benchmark_openmp_scaling_72.pbs
- benchmark_openmp_scaling_1024.pbs

The consolidated performance statistics can be reproduced from the
archived CSV files without rerunning the solver.
