# DSA5208 — Parallel Conjugate Gradient Solver

NUS DSA5208 Group Project — Parallel implementation and performance evaluation of the Conjugate Gradient (CG) method for sparse symmetric positive definite (SPD) linear systems.

## 1. Project Overview

This project investigates the performance and scalability of Conjugate Gradient solvers using serial, shared-memory, distributed-memory and hybrid parallel computing.

The implementations solve:

\[
Ax=b
\]

where \(A\) is a sparse SPD matrix.

The project compares numerical correctness, convergence, execution time, speedup and parallel efficiency across different implementations and matrix configurations.

## 2. Implementations

| Implementation | Parallelization | Status |
|---|---|---|
| Serial CG | Single CPU thread | Planned |
| CPU OpenMP CG | Shared-memory multithreading | Planned |
| MPI CG | Distributed-memory multiprocessing | Planned |
| Hybrid CG | MPI + CPU OpenMP | Planned |
| GPU CG | OpenMP target offloading | Optional extension |

The GPU extension uses OpenMP target directives rather than a separate CUDA implementation.

## 3. Team Responsibilities

| Member | Primary responsibility |
|---|---|
| Person A | Serial CG, CPU OpenMP CG, optional GPU OpenMP offloading |
| Person B | Shared Poisson matrix generation, MPI CG |
| Person C | Hybrid MPI + OpenMP CG, independent benchmarking and GPU comparisons |

All implementations must follow the common numerical specification and shared interfaces.

See [Project Specification](docs/PROJECT_SPEC.md).

## 4. Numerical Specification

The initial agreed-upon design proposal uses:

| Parameter | Value |
|---|---|
| Precision | Double precision |
| Matrix representation | CSR |
| Matrix class | Sparse SPD |
| Right-hand side | `b = A × 1` |
| Initial guess | `x₀ = 0` |
| Relative residual tolerance | `1e-8` |
| Maximum iterations | `10000` |

The reference solution is the all-ones vector.

The solver must report convergence, iteration count, final true relative residual and solution error.

These settings are subject to final team confirmation before solver implementation.

## 5. Benchmark Matrices

### Synthetic matrices

- 2D Poisson matrices using the five-point stencil.
- 3D Poisson matrices using the seven-point stencil.
- Increasing matrix sizes for scalability measurements.
- Comparisons of 2D and 3D problems with approximately equal unknown counts.

### Real-world matrix

- SuiteSparse Matrix Collection: `thermal2`.

The real-world matrix must be checked for suitability as an SPD CG benchmark before use.

## 6. Repository Structure

```text
conjugate-gradient/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── docs/
│   ├── PROJECT_SPEC.md
│   └── ENVIRONMENT.md
├── include/
│   └── csr_matrix.hpp
├── src/
│   ├── common/
│   ├── serial/
│   ├── openmp/
│   ├── mpi/
│   ├── hybrid/
│   └── gpu/
├── tests/
│   └── test_csr.cpp
├── scripts/
│   └── vanda/
├── data/
└── results/
```

The solver source directories are reserved for upcoming implementation tasks.

## 7. Development Environment

Primary benchmarking environment: NUS Vanda HPC.

Verified components:

| Component | Version or hardware |
|---|---|
| GCC | 13.2.0 |
| C++ standard | C++17 |
| CMake | 3.27.6 |
| Open MPI | 4.1.8 |
| CPU OpenMP | Supported and tested |
| NVIDIA GPU | A40 |
| NVIDIA HPC SDK | 24.5 |
| OpenMP GPU target | Successfully tested |

Refer to [Vanda Environment](docs/ENVIRONMENT.md) for verified configurations, module commands and PBS allocation instructions.

## 8. Build Instructions

The foundation currently provides a header-only CSR matrix interface and a validation test. Solver executables will be added in subsequent tasks.

On an allocated Vanda CPU compute node:

```bash
module load CMake/3.27.6-GCCcore-13.2.0

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure
```

For a Debug build:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j 2
ctest --test-dir build-debug --output-on-failure
```

The initial CSR validation test has passed in both Release and Debug configurations on Vanda.

## 9. Benchmarking Methodology

Planned evaluation metrics include:

- Matrix dimension and nonzero count.
- CG iteration count and convergence.
- Relative residual and solution error.
- Solver wall-clock execution time.
- Speedup against serial execution.
- CPU parallel efficiency.
- Strong and weak scaling where appropriate.

MPI and hybrid measurements will account for the slowest participating rank.

Benchmark configurations, compiler options and PBS resource allocations must be recorded to support reproducibility.

No CG solver performance results are available at the foundation stage.

## 10. Development Workflow

Development is organized through Git feature branches and pull requests.

Recommended workflow:

1. Branch from the latest `main`.
2. Implement a clearly scoped task.
3. Run applicable tests.
4. Review changes and update documentation.
5. Open a pull request for team review.
6. Merge after review and successful verification.

Initial foundation branch: `chore/project-foundation`.

## 11. References

1. Hestenes, M. R., and Stiefel, E. (1952). *Methods of Conjugate Gradients for Solving Linear Systems.*
2. Barrett, R., et al. (1994). *Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods.*
3. Shewchuk, J. R. (1994). *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain.*
4. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems.*
5. Davis, T. A., and Hu, Y. (2011). *The University of Florida Sparse Matrix Collection.*

## 12. Current Project Status

**Task 0 — Project foundation in progress.**

Completed:

- Vanda CPU, MPI and GPU environment verification.
- Shared numerical specification drafted.
- CSR matrix interface created.
- CMake foundation configured.
- CSR validation test passed in Release and Debug modes.

Pending:

- Finalize repository foundation scripts and placeholders.
- Confirm shared APIs and benchmark sizes with the team.
- Implement serial and parallel CG solvers.
- Conduct correctness testing and performance benchmarks.

### 13. Suggestions for Further Investigation

Three directions suggested for strengthening the project:

1. **2D versus 3D Poisson comparison:** Generate 2D and 3D Poisson systems with approximately equal numbers of unknowns, then compare sparsity, convergence and computational performance.

2. **Matrix conditioning:** Investigate how the condition number affects CG convergence, including differences between the test matrices.

3. **GMRES comparison:** Consider implementing GMRES as an additional iterative solver to compare convergence behaviour and computational cost against CG.

These investigations are supplementary to the core serial, OpenMP, MPI and hybrid CG implementations. Their final scope will depend on team agreement and available project time.
