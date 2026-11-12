> Historical context: source paths and unified-build commands describe the original revisions. For current independent builds and experiments, use the [standalone submission README](../README.md). Removed paths and archived snapshots are recorded in docs/archive/cleanup_manifest.json (repository-relative). Historical results are not verification of the standalone packages.

# DSA5208 — Conjugate Gradient Project Specification

## 1. Project Objective

Implement and benchmark the Conjugate Gradient (CG) method for solving sparse, symmetric positive definite (SPD) linear systems:

\[
Ax=b
\]

Compare the following implementations:

1. Serial C++17
2. Shared-memory CPU OpenMP
3. Distributed-memory MPI
4. Hybrid MPI + OpenMP
5. OpenMP GPU target offloading (optional extension)

The primary objective is to evaluate correctness, convergence, execution time, speedup and parallel scalability.

## 2. Shared Numerical Specification

| Parameter | Specification |
|---|---|
| Floating-point precision | `double` |
| Sparse matrix representation | Compressed Sparse Row (CSR) |
| Matrix requirements | Square, symmetric positive definite |
| Right-hand side | `b = A × 1` |
| Initial solution | `x₀ = 0` |
| Relative residual tolerance | `1e-8` |
| Maximum iterations | `10000` |
| Primary convergence criterion | `||b - Axₖ||₂ / ||b||₂ ≤ 1e-8` |
| Reference solution | Vector of ones |

Every implementation must use equivalent numerical settings.

The solver should detect numerical breakdown, including invalid or nonpositive CG denominators, and report non-convergence when the maximum iteration count is reached.

## 3. Core Conjugate Gradient Algorithm

Given an SPD matrix `A`, right-hand side `b` and initial guess `x₀`:

1. Compute `r₀ = b − Ax₀`.
2. Set `p₀ = r₀`.
3. Compute `ρ₀ = r₀ᵀr₀`.
4. At each iteration:
   - `qₖ = Apₖ`
   - `αₖ = ρₖ / (pₖᵀqₖ)`
   - `xₖ₊₁ = xₖ + αₖpₖ`
   - `rₖ₊₁ = rₖ − αₖqₖ`
   - `ρₖ₊₁ = rₖ₊₁ᵀrₖ₊₁`
   - `βₖ = ρₖ₊₁ / ρₖ`
   - `pₖ₊₁ = rₖ₊₁ + βₖpₖ`
5. Stop when the convergence criterion is satisfied.

The final true residual must be recomputed using `b − Ax`, rather than relying exclusively on the recursively updated residual.

## 4. Benchmark Matrices

### 4.1 Synthetic Poisson Matrices

Use sparse finite-difference discretizations of the Poisson equation:

- 2D Poisson: five-point stencil
- 3D Poisson: seven-point stencil

The shared matrix generator must produce deterministic CSR matrices.

Compare both:

- Increasing problem sizes for scalability analysis.
- 2D and 3D matrices with approximately equal numbers of unknowns to study differences in sparsity and convergence.

Record matrix dimension, nonzero count and average nonzeros per row.

### 4.2 SuiteSparse Matrix

Include the `thermal2` matrix from the SuiteSparse Matrix Collection, subject to verification that the selected matrix satisfies the solver's SPD requirements.

Document its source, dimensions, nonzero count and preprocessing, if any.

## 5. Correctness and Convergence

Every implementation must report:

- Matrix name and dimensions.
- Number of nonzero entries.
- Solver implementation.
- Iteration count.
- Relative residual norm.
- Relative solution error against the all-ones reference solution.
- Convergence status.
- Elapsed solver time.

Correctness must be verified independently of execution time.

Compare iteration counts and numerical accuracy between serial, OpenMP, MPI, hybrid and GPU implementations.

Minor floating-point differences arising from parallel reduction order are acceptable if the required convergence and correctness checks pass.

## 6. Performance Measurement

Measure elapsed wall-clock time for the CG solver.

Exclude matrix generation, file loading and unrelated setup from the primary solver-time measurement.

For MPI and hybrid implementations, report the maximum elapsed time across participating ranks.

For GPU offloading, clearly distinguish any setup or host-device transfer costs from kernel computation when presenting supplementary timings.

Use repeated benchmark runs and report a representative statistic, such as the median, together with the number of repetitions.

### Performance Metrics

Speedup:

`S(p) = T_serial / T_parallel(p)`

Parallel efficiency:

`E(p) = S(p) / p`

For MPI, `p` denotes MPI processes. For CPU OpenMP, it denotes threads. For hybrid configurations, report both the number of MPI ranks and threads per rank, with total CPU execution resources stated explicitly.

GPU speedup should be reported relative to a clearly identified CPU baseline rather than interpreted as CPU parallel efficiency.

## 7. Team Responsibilities

| Member | Responsibility |
|---|---|
| Person A | Serial CG, CPU OpenMP CG, optional OpenMP GPU offloading |
| Person B | Shared Poisson matrix generators, MPI CG |
| Person C | Hybrid MPI + OpenMP CG, independent benchmarking and GPU comparisons |

All implementations must reuse the agreed CSR representation and numerical conventions.

## 8. Shared Implementation Requirements

- C++17 for core solver implementations.
- CMake for building project targets.
- Portable source paths without machine-specific absolute paths.
- Separate executable targets for different solver variants.
- Consistent command-line inputs and result reporting.
- Reproducible build and benchmark instructions.
- No benchmark results committed without configuration metadata.

MPI and hybrid implementations must document matrix partitioning, inter-process communication and collective reductions.

## 9. Vanda Environment

CPU compilation and execution have been verified using GCC, OpenMP and Open MPI.

OpenMP GPU target offloading has been verified on an NVIDIA A40 using NVIDIA HPC SDK 24.5.

GPU support is optional and must not prevent CPU-only builds.

Compile performance-sensitive binaries on the appropriate Vanda compute nodes rather than assuming that binaries built on the login node are optimized for compute nodes.

## 10. References

- Hestenes, M. R., and Stiefel, E. (1952). Methods of Conjugate Gradients for Solving Linear Systems.
- Barrett et al. (1994). Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods.
- Shewchuk, J. R. (1994). An Introduction to the Conjugate Gradient Method Without the Agonizing Pain.
- Saad, Y. (2003). Iterative Methods for Sparse Linear Systems.
- Davis, T. A., and Hu, Y. (2011). The University of Florida Sparse Matrix Collection.

## 11. Items to Finalize Before Implementation

- Exact benchmark grid sizes.
- SuiteSparse `thermal2` validation and input conversion.
- Shared CSR and solver API definitions.
- Executable command-line interface.
- Benchmark repetitions and resource configurations.
- Any additional analysis requested by the instructor, such as conditioning or GMRES comparisons.