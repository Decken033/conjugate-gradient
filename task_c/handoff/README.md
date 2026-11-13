# Task C handoff: original unprofiled WSL experiments

Source: commit 56ac360, Hybrid_Team_Handoff/Hybrid_Team_Handoff. Missing common/MPI implementation dependencies and option headers come from the same revision. All copied files are byte-identical; only directory placement and build instructions are new. No V1 workspace or profiling is included.

## Dependencies and independent build

C++17, CMake >=3.20, MPI and CPU OpenMP are required. Recorded environment: Ubuntu 24.04/WSL2, GCC 13.3.0, OpenMPI 4.1.6, CMake 3.28.3 and detected OpenMP 4.5. From task_c/handoff:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ -DMPI_CXX_COMPILER=mpicxx -DBUILD_TESTING=ON
cmake --build build --parallel 4
```

This local CMake requires MPI/OpenMP directly; original CG_ENABLE_MPI/CG_ENABLE_OPENMP_GPU switches are unnecessary. It has no parent/sibling/task_a/task_b source dependency. Original production compiler defaults and linkage are retained. No additional optimization flags are imposed.

## Recorded experiments

The original RUN_REPORT.md and RUN_REPORT.docx describe exploratory WSL timing, not final Vanda results. Two MPI ranks and four threads per rank, five measured solves and one warmup were used:

| Grid | Schedule/chunk | Notes |
|---|---|---|
| 64 | Static/0, two separate series | Chunk zero recorded in hybrid_2d64_scheduling_summary.csv |
| 64 | Dynamic/0 | Five runs |
| 64 | Guided/4 | Five runs |
| 256 | Static/0 | Five runs |
| 256 | Dynamic/0 | Five runs |
| 256 | Guided/0 | Five runs |

The original guide's static-256 command, using this package's build directory, is:

```bash
mkdir -p results/hybrid
OMP_PROC_BIND=true OMP_PLACES=cores \
mpirun -np 2 ./build/run_hybrid_benchmark \
  2d 256 4 static 0 5 results/hybrid/hybrid_2d256_r2_t4_static.csv 1
```

For the other documented schedule/grid combinations, use the table's arguments and a distinct output filename; keep two ranks/four threads, five runs/one warmup. The binding above is documented in RUN_GUIDE, not a per-job hardware-binding manifest. No original handoff campaign shell/PBS script was supplied, so none has been invented. scripts/ contains only the original optional dataset downloader.

Allocation remains per SpMV, including communication metadata/global-vector/output buffers. Timings surround the solver call, report maximum rank elapsed time and exclude post-solve reporting. True-residual checks/restarts and their costs remain original. The reports give 122 iterations for grid64 and 454 for grid256; these are historical observations, not newly verified outcomes or guaranteed bitwise results. The 512 MPI/OpenMP/hybrid comparison was explicitly pending in these reports.

## Correctness and safety scope

From this variant directory, inside a compute allocation with four MPI slots and eight CPU cores (two threads per rank):

```bash
ctest --test-dir build --output-on-failure --no-tests=error -j 1
```

CTest runs original test_hybrid_ops at two ranks and test_hybrid_cg at two/four ranks. Each test has a 90-second timeout; Release test assertions are enabled. Numerical source and test contents are unchanged. Original runner/library failure paths are retained, not certified collectively safe; use identical valid arguments on every rank. Allocation failures, malformed input and unsupported sizes may expose historical limitations. No integrated safety reductions have been imported. Independent verification on NUS Vanda on 13 November 2026 confirmed successful compilation, all three registered CTests passing, and representative Hybrid benchmark execution. A two-rank, four-thread-per-rank Poisson 2D64 run converged in 122 iterations with a true relative residual of approximately 8.71368e-09. See [Vanda verification](../../docs/VANDA_VERIFICATION_2026-11-13.md). These functional tests do not certify every MPI failure path or reproduce the original WSL2 timing results.

## Inputs and outputs

Poisson matrices are generated. CLI: `run_hybrid_benchmark <2d|3d|thermal2> <grid|matrix.mtx> <threads> <static|dynamic|guided> <chunk> <repeats> <output.csv> [warmups]`. Tolerance is 1e-8 and maximum iterations 10000; omitted warmups default to one. Use explicit positive thread counts matching the allocation. The original manufactured RHS is A*ones with zero initial guess. The 14-column benchmark CSV records configuration, iterations, convergence, time, true residual, solution error, schedule and chunk. Use unique output filenames; writers truncate existing files. Inspect convergence and residuals before calculating performance summaries.

No historical CSV is required. Optional thermal2 input can be acquired with the unchanged `scripts/download_thermal2.ps1` from this variant directory:

```powershell
pwsh -File scripts/download_thermal2.ps1 -Destination data/suitesparse
```

This requires PowerShell, tar and network access; transfer the resulting matrix to Vanda if needed. Pass data/suitesparse/thermal2/thermal2.mtx as the CLI matrix path. Download execution and a pinned expected input hash are not verified. No identifiable Task C thermal2 performance campaign is claimed here.

## Provenance gaps

The original handoff references a hybrid CMake file absent from its directory. The minimal target graph is reconstructed from original dependencies; flat baseline includes resolve via local include directories without source edits. WSL host load/binding and exact executable fingerprints are incomplete. A final team report selecting these experiments is unavailable. This unprofiled handoff must not be called the profiled V0 baseline.
