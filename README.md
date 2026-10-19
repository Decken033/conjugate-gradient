# DSA5208 — Parallel Conjugate Gradient Solver

NUS DSA5208 Group Project — parallel implementation and performance evaluation of the Conjugate Gradient (CG) method for sparse symmetric positive definite (SPD) linear systems.

## 1. Project Overview

This project investigates numerical correctness, convergence, runtime and scalability of Conjugate Gradient solvers using serial, shared-memory (OpenMP), distributed-memory (MPI) and hybrid parallel computing.

The implementations solve

\[
Ax=b,
\]

where \(A\) is a sparse SPD matrix. The serial implementation provides the tested reference solver and shared matrix infrastructure used to develop and evaluate the parallel versions.

**Status as of 19 October 2026:** Task 1 (serial CG implementation, correctness tests, matrix generators, Matrix Market loading and preliminary benchmarks) has been implemented and verified on NUS Vanda. Task 1 documentation and Git integration are being finalized; OpenMP, MPI and hybrid solvers are not claimed complete here.

## 2. Implementations

| Implementation | Parallelization | Status |
|---|---|---|
| Serial CG | Single CPU thread | **Implemented; numerical tests and preliminary benchmarks passed** |
| CPU OpenMP CG | Shared-memory multithreading | Pending (Person A, Task 2) |
| MPI CG | Distributed-memory multiprocessing | Pending / team integration (Person B) |
| Hybrid CG | MPI + CPU OpenMP | Pending / team integration (Person C) |
| GPU CG | OpenMP target offloading | Optional extension; environment feasibility checked, solver not yet implemented |

### Completed Task 1 capabilities

- **CSR sparse representation:** a shared `cg::CSRMatrix` (`rows`, `cols`, `row_ptr`, `col_idx`, `values`) with structural validation; double-precision numerical values.
- **Common numerical operations:** reusable CSR sparse matrix-vector product and operations needed by CG.
- **Serial CG solver:** starts at \(x_0=0\), exposes configurable relative tolerance and iteration limit, reports convergence, iteration count and verified true relative residual; includes numerical-breakdown checks and true-residual verification / restart handling.
- **Synthetic matrices:** reusable CSR generators for 2D five-point and 3D seven-point Poisson operators with Dirichlet-boundary stencils.
- **Real-world matrix import:** Matrix Market coordinate-real general/symmetric loader, 1-based to 0-based conversion, mirrored off-diagonal entries for symmetric input, duplicate-coordinate summation, sorted CSR construction and input validation.
- **Benchmark harness:** one untimed CG warm-up, individually timed solver-only repetitions, immediate warm-up / per-run progress messages, numerical accuracy checks outside the timed region, and a CSV row per measured repetition.
- **Correctness and integration checks:** small CSR and CG cases, Poisson systems, benchmark CSV tests, Matrix Market loader fixtures, and full `thermal2` CSR validation.

**Shared interfaces:** `cg::CSRMatrix`, `cg::csr_matvec`, `cg::make_poisson_2d`, `cg::make_poisson_3d` and `cg::load_matrix_market` can be reused by other solver implementations. For MPI, matrix partitioning and rank-wise distribution remain separate implementation work; every rank need not independently load the full matrix.

## 3. Team Responsibilities

| Member | Primary responsibility |
|---|---|
| Person A | Serial CG, CPU OpenMP CG, optional GPU OpenMP offloading |
| Person B | MPI CG and integration of the shared matrix-generation utilities |
| Person C | Hybrid MPI + OpenMP CG, independent benchmarking and GPU comparisons |

The original work split assigned shared Poisson generation to Person B. During Task 1, Person A implemented the shared Poisson generators and Matrix Market loader so that all solver implementations can use common inputs. Coordination and integration with Person B remain necessary.

All implementations should adhere to the common numerical specification and shared interfaces. See [Project Specification](docs/PROJECT_SPEC.md).

## 4. Numerical Specification and Correctness

The implemented Task 1 serial experiments use:

| Parameter | Value |
|---|---|
| Arithmetic | Double precision (`double`) |
| Storage | Compressed Sparse Row (CSR) |
| Matrix type | Square, intended SPD |
| Known exact solution | \(x_* = \mathbf{1}\) (all ones) |
| Constructed RHS | \(b = A x_*\) |
| Initial guess | \(x_0 = \mathbf{0}\) |
| Relative residual tolerance | `1e-8` |
| Maximum CG iterations | `10000` |
| Numerical outputs | Converged flag, iterations, true relative residual, relative solution error |

The **true relative residual** is evaluated against the original system:

\[
r_{\mathrm{true}}=b-Ax,\qquad
\rho=\frac{\|r_{\mathrm{true}}\|_2}{\|b\|_2}.
\]

The **relative solution error**, with the manufactured reference solution, is

\[
e_x=\frac{\|x-x_*\|_2}{\|x_*\|_2}.
\]

Residual and solution error are distinct: an error larger than the residual is not in itself evidence of an incorrect solver, especially for a matrix with nontrivial conditioning. Structural CSR validation and a successful CG run do **not** mathematically establish positive definiteness of an arbitrary imported matrix; SPD suitability is a dataset/algorithm assumption to verify for any new benchmark.

## 5. Benchmark Matrices and Dataset Reproduction

### Synthetic Poisson matrices

- **2D Poisson:** five-point stencil, diagonal `4`, neighboring off-diagonal `-1`; \(N=n^2\).
- **3D Poisson:** seven-point stencil, diagonal `6`, neighboring off-diagonal `-1`; \(N=n^3\).
- Both are generated directly in CSR and use the same exact-solution/RHS construction as `thermal2`.
- Cases with comparable unknown counts permit 2D-versus-3D comparisons; larger cases will support subsequent strong/weak scaling studies.

### Real-world matrix — SuiteSparse `Schmid/thermal2`

`thermal2` is an unstructured finite-element steady-state thermal problem. The original Matrix Market coefficient matrix has symmetric coordinate storage.

| Property | Verified value |
|---|---:|
| Matrix dimension | 1,228,045 × 1,228,045 |
| Stored entries in `thermal2.mtx` | 4,904,179 |
| Expanded CSR entries | 8,580,313 |
| Matrix Market header | `%%MatrixMarket matrix coordinate real symmetric` |
| Accompanying RHS | `thermal2_b.mtx`, `matrix array real general` |

**Dataset links**

- Official SuiteSparse collection page: https://sparse.tamu.edu/Schmid/thermal2
- **University of Florida alternative archive (successfully downloaded on Vanda on 19 October 2026):** https://www.cise.ufl.edu/research/sparse/MM/Schmid/thermal2.tar.gz

On 19 October 2026, the official SuiteSparse download endpoint returned `403 Forbidden`. The University of Florida archive returned `200 OK` (64,854,356 bytes) and was downloaded and integrity-checked successfully. Accessibility is not guaranteed indefinitely.

From the **repository root on Vanda**:

```bash
mkdir -p data/suitesparse
curl -fL --retry 3 \
  -o data/suitesparse/thermal2.tar.gz \
  https://www.cise.ufl.edu/research/sparse/MM/Schmid/thermal2.tar.gz

gzip -t data/suitesparse/thermal2.tar.gz
tar -xzf data/suitesparse/thermal2.tar.gz -C data/suitesparse

head -n 1 data/suitesparse/thermal2/thermal2.mtx
# %%MatrixMarket matrix coordinate real symmetric
```

Extracted files:

```text
data/suitesparse/thermal2/thermal2.mtx
data/suitesparse/thermal2/thermal2_b.mtx
```

Our CG benchmarks do **not** use the provided `thermal2_b.mtx`; they construct `b = A * ones` to know the exact solution. The loader reconstructs symmetric off-diagonal entries, sums duplicate coordinates and returns validated CSR storage.

Manual full-dataset integration test (on an allocated Vanda compute node):

```bash
./build/test_matrix_market data/suitesparse/thermal2/thermal2.mtx
```

**Observed:** `Rows: 1228045`, `Cols: 1228045`, `NNZ: 8580313`; validation passed. The command took **5.825 s elapsed** in the recorded run, including fixture checks, parsing and validation; this is **not** the serial CG solve time.

Downloaded archives and matrices remain excluded from Git (`data/*` in `.gitignore`).

## 6. Repository Structure

The Task 1 files currently implemented on `feat/serial-cg` are:

```text
conjugate-gradient/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── docs/
│   ├── PROJECT_SPEC.md
│   └── ENVIRONMENT.md
├── include/
│   ├── csr_matrix.hpp
│   ├── csr_ops.hpp
│   ├── cg_solver.hpp
│   ├── poisson.hpp
│   ├── matrix_market.hpp
│   └── serial_benchmark.hpp
├── src/
│   ├── common/
│   │   ├── csr_ops.cpp
│   │   ├── poisson.cpp
│   │   └── matrix_market.cpp
│   ├── serial/
│   │   ├── cg_serial.cpp
│   │   ├── benchmark_serial.cpp
│   │   ├── benchmark_csv.cpp
│   │   └── run_serial_benchmark.cpp
│   └── [openmp/, mpi/, hybrid/, gpu/ planned/reserved]
├── tests/
│   ├── test_csr.cpp
│   ├── test_csr_ops.cpp
│   ├── test_serial_cg.cpp
│   ├── test_serial_benchmark.cpp
│   └── test_matrix_market.cpp
├── scripts/
│   └── vanda/
├── data/                    # downloaded matrices (ignored)
└── results/                 # generated benchmark CSVs (ignored by default)
```

The project foundation and parallel directories may evolve as teammates integrate their work. `results/serial_2d_64.csv`, `results/serial_3d_16.csv` and `results/serial_thermal2_3runs.csv` were explicitly staged as selected reproducibility artifacts despite the general results-directory ignore rule.

### Libraries and executables

| CMake target | Purpose |
|---|---|
| `cg_common` | Shared headers / CSR interface |
| `cg_ops` | Common CSR numerical operations |
| `cg_serial` | Serial CG implementation |
| `cg_poisson` | Poisson matrix generators |
| `cg_matrix_market` | Matrix Market-to-CSR loader |
| `cg_serial_benchmark` | Timing, numerical metrics and CSV output |
| `run_serial_benchmark` | CLI for Poisson and `thermal2` benchmarks |

## 7. Development Environment

Primary testing and benchmarking environment: **NUS Vanda HPC**.

| Component | Verified environment |
|---|---|
| C++ | C++17 |
| Compiler | GCC / GCCcore 13.2.0 |
| CMake module | `CMake/3.27.6-GCCcore-13.2.0` |
| CPU observed during performance investigation | Intel Xeon Platinum 8452Y (CN-016) |
| Shared-memory OpenMP | Environment support checked; solver pending |
| Open MPI | 4.1.8 (environment checked; solver pending) |
| NVIDIA GPU | A40 (GPU work optional) |
| NVIDIA HPC SDK | 24.5 (offload feasibility checked) |

The recorded serial benchmarks were executed on allocated compute nodes, including **CN-016** (performance diagnosis) and **CN-149** (`thermal2`). Avoid treating timings from different compute nodes and allocations as directly interchangeable.

For a CPU interactive job (when permitted by your Vanda allocation):

```bash
qsub -I -P personal-e1536052 -q interactive_cpu \
  -l select=1:ncpus=2:mem=4gb -l walltime=00:30:00
```

Load the module **on the compute node**, including when running an already compiled binary:

```bash
module load CMake/3.27.6-GCCcore-13.2.0
```

This also supplies the matching GCC runtime; running without it may produce a `GLIBCXX_3.4.32 not found` dynamic-library error. See [Vanda Environment](docs/ENVIRONMENT.md) for additional queue and module information.

## 8. Build, Test and Run

Run compilation and performance tests on an **allocated compute node**, not the login node. From the repository root:

```bash
module load CMake/3.27.6-GCCcore-13.2.0

# Release
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure

# Optional Debug build
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j 2
ctest --test-dir build-debug --output-on-failure
```

**Observed verification:** the final Task 1 Release configuration passed **5/5 CTest targets** on Vanda (CSR validation, CSR matvec, serial CG, serial benchmark, Matrix Market). Earlier Debug and Release builds also passed their then-current test suites; the latest five-test Debug configuration has not been separately reported as verified.

### Reproduce synthetic benchmarks

```bash
mkdir -p results

./build/run_serial_benchmark 2d 64 5 results/serial_2d_64.csv
./build/run_serial_benchmark 3d 16 5 results/serial_3d_16.csv

# Larger examples (longer execution)
./build/run_serial_benchmark 2d 256 5 results/serial_2d_256.csv
./build/run_serial_benchmark 3d 40 5 results/serial_3d_40.csv
```

Usage:

```text
run_serial_benchmark <2d|3d> <grid_size> <repeats> <output.csv>
run_serial_benchmark thermal2 <matrix.mtx> <repeats> <output.csv>
```

### Reproduce the `thermal2` benchmark

Download and extract the data as described in Section 5, then run:

```bash
./build/run_serial_benchmark \
  thermal2 \
  data/suitesparse/thermal2/thermal2.mtx \
  3 \
  results/serial_thermal2_3runs.csv
```

The runner prints immediate warm-up and per-run progress messages. A three-repetition `thermal2` experiment executes **four CG solves in total** (one untimed warm-up, three measured), so allow ample PBS walltime. The outer shell `time` result includes all four solves and file loading; it is **not** the solver-only baseline.

## 9. Benchmarking Methodology and Task 1 Results

### 9.1 Timing and CSV protocol

The serial benchmark uses `std::chrono::steady_clock` around **only** `solve_serial_cg(A,b,options)`. Specifically:

1. Construct or load the matrix, validate CSR, and construct \(b=A\mathbf{1}\) **before** timing.
2. Perform one **untimed warm-up CG solve** and verify convergence.
3. For each measured repetition, time the solver call alone and verify convergence.
4. Calculate relative solution error and write per-run records **outside** the timed section.
5. Report each warm-up / run transition immediately; progress printing is outside timed regions.

The CSV header is:

```text
solver,matrix_name,n,nnz,ranks,threads,run_index,iterations,converged,solve_seconds,true_relative_residual,relative_solution_error
```

For serial CG, `ranks=1`, `threads=1`, and `converged=1` denotes success. Retain each recorded repetition rather than reporting only a best run. Unless noted otherwise, times below are **preliminary solver-only measurements** from Release builds on Vanda; a final controlled cross-solver comparison is still required.

### 9.2 Synthetic Poisson results

| Matrix | N | NNZ | CG iterations | True relative residual | Median solve time (s) | Notes |
|---|---:|---:|---:|---:|---:|---|
| 2D Poisson, `n=64` | 4,096 | 20,224 | 122 | 8.71368e-9 | 0.00568559 | Initial 5-run dataset on CN-016; see CSV |
| 3D Poisson, `n=16` | 4,096 | 27,136 | 41 | 6.11714e-9 | 0.00238256 | Initial 5-run dataset on CN-016; see CSV |
| 2D Poisson, `n=256` | 65,536 | 326,656 | 454 | 9.91666e-9 | 0.237475 | Preliminary 5-run measurement |
| 3D Poisson, `n=40` | 64,000 | 438,400 | 101 | 8.83638e-9 | 0.0626236 | Preliminary 5-run measurement |
| 2D Poisson, `n=512` | 262,144 | 1,308,672 | 894 | 9.52607e-9 | **Not finalized** | Fast ≈2.1 s and slow ≈4.6 s regimes; see §9.4 |
| 3D Poisson, `n=64` | 262,144 | 1,810,432 | 158 | 9.03206e-9 | 0.463947 | Preliminary 5-run measurement |

All listed solves reached the configured true relative residual tolerance. Timing comparisons should consider different iteration counts, sparsity, problem geometry and job conditions. The initial CSVs tracked for reproducibility are `results/serial_2d_64.csv` and `results/serial_3d_16.csv`. Larger-case results remain preliminary and may need controlled reruns.

A subsequent `2d 64` compatibility/progress check on **CN-149**, after Matrix Market support and progress output were added, also passed (122 iterations, residual `8.71368e-9`, two solve times `0.0064521 s` and `0.0064379 s`). These separate-node numbers should not replace the original CN-016 dataset.

### 9.3 Real-world `thermal2` results (CN-149)

The first verified real-world solve used \(N=1{,}228{,}045\), `NNZ=8580313`, \(b=A\mathbf{1}\), tolerance `1e-8`, maximum iterations `10000`, and one untimed warm-up.

| Run / statistic | Solve time (s) | Iterations | True relative residual | Relative solution error |
|---|---:|---:|---:|---:|
| Initial single measured run | 153.520213 | 4,333 | 9.8606e-9 | 4.4121e-7 |
| Repeated run 1 | 148.122 | 4,333 | 9.8606e-9 | 4.4121e-7 |
| Repeated run 2 | 150.226 | 4,333 | 9.8606e-9 | 4.4121e-7 |
| Repeated run 3 | 146.410 | 4,333 | 9.8606e-9 | 4.4121e-7 |
| **Three-run median** | **148.122** | — | — | — |

Three-run statistics: mean **148.253 s**, sample standard deviation **1.911 s**, coefficient of variation **1.29%** (rounded). All three runs had identical iteration counts and numerical metrics. The separate initial run is retained as a diagnostic reference and is **not** included in the three-run statistics.

The complete three-run command took **10 min 0.576 s elapsed**, because it included one warm-up and three measured solves, plus loading and other overhead. For future speedup, the preliminary serial reference is the **three-run median solver time, 148.122 s**, subject to repeating baseline measurements under a comparable allocation when parallel benchmarking begins.

Results file: `results/serial_thermal2_3runs.csv` (selected CSV staged for Git). The initial file `results/serial_thermal2_initial.csv` remains local/ignored unless explicitly added later.

### 9.4 Vanda performance variability investigation (CN-016)

A larger 2D Poisson `n=512` serial benchmark (262,144 unknowns, 894 CG iterations) showed a **repeatable fast-to-slow transition** during repeated solves:

- Early repetitions ran at approximately **2.1 s**, while later repetitions were approximately **4.6 s**, despite identical iteration counts, true residuals and solution errors.
- Restricting CPU affinity separately to **CPU 22** and **CPU 28** did **not** remove the transition. Migration between those two allocated CPUs is therefore unlikely to be the principal cause.
- A pinned-run shell measurement (`real=36.765 s`, `user=36.605 s`, `sys=0.032 s`) showed nearly continuous CPU time; sustained waiting to be scheduled was not supported as the main explanation.
- Processor frequency readings were not available through `lscpu` / `cpufreq` sysfs. `perf stat -I 1000 -e cycles:u,instructions:u,task-clock` was available with performance-counter permission level 2.
- In one slow-phase interval, counter-derived cycles per second remained around **3.16–3.18 billion**, while IPC fell from roughly **3.9** to **1.79** and retired instructions per second fell from approximately **12.2 billion** to **5.7 billion**. This identifies a strong decline in observed instruction throughput, not an obvious decline in measured cycles per second.
- A later cache-counter experiment **did not reproduce the slowdown**: all five measured runs remained about **2.05–2.12 s**, IPC stayed roughly **3.9–4.0**, and sampled generic cache-miss percentages rose from about **0.3–0.4%** early to about **1.4–1.8%** later without the earlier collapse in IPC.

**Interpretation and limitations:** the counters confirm two distinct performance regimes and a substantial IPC reduction in the slow regime. They **do not establish the root cause**. Shared-cache / memory interference, differences in hardware stalls, processor execution conditions and other node-level variability remain hypotheses. The counter-derived cycle rate is not a direct, definitive measure of CPU operating frequency; generic cache-miss counts cannot isolate all memory stalls. The later experiment shows that rising generic cache-miss percentages alone are insufficient to explain the slowdown. Do not claim that cache contention, throttling or thermal effects were proven.

**Benchmarking consequence:** do not cherry-pick the 2.1 s regime or report an unqualified median across strongly bimodal measurements as the final `n=512` baseline. Keep run-level data and document anomalous conditions. For serial-versus-parallel speedup, repeat the serial reference using comparable compute-node, affinity, compiler/build and PBS settings; record node and job information and quantify variation.

### 9.5 Planned parallel scaling evaluation

Future OpenMP, MPI and hybrid benchmarks will use the same matrices, tolerance, manufactured exact solution and convergence checks. Planned metrics include:

- Strong scaling at fixed matrix dimensions with increasing threads/ranks.
- Weak scaling with controlled per-worker problem size; account for changes in CG iteration counts as the global problem changes.
- Speedup \(S_p=T_1/T_p\) and parallel efficiency \(E_p=S_p/p\), using a clearly specified baseline and processing-unit count.
- Per-run timing, numerical residuals, error, convergence and allocation details.
- MPI/hybrid timing over the participating ranks, using the slowest-rank elapsed time for collective performance assessment.

These **parallel results have not yet been produced** by Task 1.

## 10. Development Workflow

Development uses Git feature branches and pull requests:

1. Develop a scoped task on a feature branch.
2. Build on Vanda and run applicable correctness/integration tests.
3. Preserve representative benchmark CSVs and document reproducible commands.
4. Review source diff, documentation, numerical outputs and ignored artifacts.
5. Commit and push the branch, open a PR, and merge after team review.

Task 1 development branch: **`feat/serial-cg`**. The branch contains the current serial solver and reusable matrix infrastructure as **local changes awaiting final staging, commit and push** at the time of this README update. Do not treat files on this branch as already merged into `main`.

Git ignores build outputs, downloaded matrices (`data/*`) and generated benchmark results (`results/*`) by default. Select representative small CSVs for version control explicitly, for example:

```bash
git add -f -- results/serial_2d_64.csv results/serial_3d_16.csv results/serial_thermal2_3runs.csv
```

## 11. References

1. Hestenes, M. R., and Stiefel, E. (1952). *Methods of Conjugate Gradients for Solving Linear Systems.*
2. Barrett, R., et al. (1994). *Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods.*
3. Shewchuk, J. R. (1994). *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain.*
4. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems.*
5. Davis, T. A., and Hu, Y. (2011). *The University of Florida Sparse Matrix Collection.*
6. SuiteSparse Matrix Collection, `Schmid/thermal2`: https://sparse.tamu.edu/Schmid/thermal2

## 12. Current Project Status (19 October 2026)

### Task 0 — Project foundation

Completed for the serial workflow: shared C++17/CMake environment, basic CSR validation, Vanda compiler/PBS setup and numerical specification. Parallel environment capabilities have been checked separately; parallel solver correctness/performance is still pending.

### Task 1 — Serial CG and shared numerical infrastructure

| Task 1 milestone | Status / evidence |
|---|---|
| CSR structure and common SpMV | **Done** — shared headers/source, tests pass |
| Serial CG and breakdown/residual handling | **Done** — small SPD tests and Poisson convergence checks pass |
| Debug/Release verification | **Done for prior builds**; current five-test suite verified **5/5 Release** |
| 2D/3D Poisson generators | **Done** — validated on several grid sizes |
| Serial benchmark timing and CSV | **Done** — warm-up, repeated solver-only timing, per-run CSV |
| Task 1.10 synthetic performance investigation | **Documented** — observed node-level timing variability, root cause unresolved |
| Task 1.11 Matrix Market + `thermal2` | **Done** — original archive, loader, full CSR verification, 4,333-iteration CG convergence |
| Task 1.12 benchmark progress and results | **Done** — progress logging verified, selected results staged |
| Task 1.13 README, final source review, Git/PR | **Documentation updated; final stage/commit/push and review pending** |

### Remaining team work

- **Person A:** implement and validate CPU OpenMP CG, benchmark thread scaling, report speedup/efficiency; investigate optional GPU OpenMP offloading if time permits.
- **Person B:** implement/integrate MPI CG and data distribution using compatible CSR/matrix conventions.
- **Person C:** integrate hybrid MPI+OpenMP CG and independently compare benchmark performance, including any GPU study.
- **Team:** agree on final matrix sizes, scaling protocol, comparable compute allocations and consistent baseline methodology; review and merge feature branches.

## 13. Suggestions for Further Investigation

These are **optional** beyond the core project scope:

1. **2D versus 3D Poisson:** compare sparsity, iterations and runtime at approximately equal numbers of unknowns.
2. **Matrix conditioning:** study how conditioning influences CG convergence and residual-versus-solution-error behavior.
3. **GMRES comparison:** compare convergence and computational expense against a non-CG iterative method, if time and project scope allow.

Prioritize required CPU OpenMP, MPI and hybrid implementation and reproducible scaling benchmarks before these optional investigations.
