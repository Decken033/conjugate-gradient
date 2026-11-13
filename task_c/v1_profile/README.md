# Task C V1: original workspace reuse and profiling

Source: Member C commit 846239d (origin/marssiiiii-patch-1). All production sources, tests, profiler and scripts are copied byte-for-byte. Per-solve workspace reuse and original profiling overhead are retained. MPI/OpenMP comparator implementations are local copies from this same revision, not Task A/B binaries.

## Dependencies and independent build

C++17, CMake >=3.20, MPI and CPU OpenMP are required. Python 3 is needed only for profile summarization. Original Vanda documentation specifies the following modules. From task_c/v1_profile:

```bash
module purge
module load CMake/3.27.6-GCCcore-13.2.0
module load openmpi/gcc/64/4.1.8
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ -DMPI_CXX_COMPILER=mpicxx -DBUILD_TESTING=ON
cmake --build build --parallel 4
```

MPI/OpenMP are mandatory. GPU and unrelated serial benchmarks are omitted; original production compiler defaults and target linkage are preserved. Build paths are local to this variant.

## Original V1 campaign

Inside a Vanda allocation with eight CPU cores, two MPI processes and four threads per process:

```bash
BUILD_DIR=build bash scripts/run_hybrid_v1_profile_2x4.sh
```

The unchanged script defaults to 2D512, two ranks/four threads, static schedule/chunk zero, ten measured runs, five warmups, OMP_PROC_BIND=true, OMP_PLACES=cores, and `--map-by slot:PE=4 --bind-to core`. BUILD_DIR=build adapts the script's historical build-v1-profile default without editing it. Its default output is results/v1-profile/hybrid_2d512_r2_t4_v1_profile.csv. Supply a new output filename as the first script argument for subsequent runs. REPEATS/WARMUPS overrides exist in the original script; retain defaults for the documented experiment.

Outputs include the benchmark CSV and `.profile.rank0.csv` / `.profile.rank1.csv` sidecars. Summarize only sidecars belonging to one solver/configuration/run set:

```bash
python3 scripts/summarize_profiles.py \
  results/v1-profile/hybrid_2d512_r2_t4_rank_summary.csv \
  'results/v1-profile/hybrid_2d512_r2_t4_v1_profile.csv.profile.rank*.csv'
```

This reads newly generated profiles, not historical CSV inputs. The original summarizer does not enforce homogeneous input sets: do not mix unrelated configurations in the glob.

## Original profiled comparator command patterns

The branch README_PROFILING documents these patterns, with ten measurements/five warmups and an eight-worker budget. Run inside suitable compute allocations from this variant directory:

```bash
mkdir -p results/profiling/mpi results/profiling/openmp results/profiling/hybrid
mpirun -np 8 ./build/run_mpi_benchmark \
  2d 512 10 results/profiling/mpi/mpi_2d512_r8.csv 5
export OMP_NUM_THREADS=8 OMP_PROC_BIND=true OMP_PLACES=cores
./build/run_openmp_benchmark \
  2d 512 8 static 0 10 results/profiling/openmp/openmp_2d512_t8.csv 5
export OMP_NUM_THREADS=2 OMP_PROC_BIND=true OMP_PLACES=cores
mpirun -np 4 --map-by slot:PE=2 --bind-to core \
  ./build/run_hybrid_benchmark \
  2d 512 2 static 0 10 results/profiling/hybrid/hybrid_2d512_r4_t2.csv 5
```

The last command executes the preserved V1 hybrid, not the unresolved historical V0 hybrid. MPI's documented command does not specify explicit binding; do not invent a historical map. MPI profiles are per-rank sidecars; OpenMP writes `<output.csv>.profile.csv`. Comparator code is provided because identifiable profiled comparator results exist, not because a final report confirmed their inclusion. Dedicated comparator correctness tests are not packaged; the three registered tests validate the hybrid implementation against its original operations/reference checks.

## Timing and provenance

recvcounts/displacements, x_global and SpMV output are allocated once per CG solve and reused. Workspace initialization and profiling scopes remain inside measured solves; profile CSV writes occur afterwards. Profile totals accumulate measured runs, not warmups; timer construction still executes in warmups. Per-rank collective duration includes waiting, not just network transfer. No instrumentation has been removed or added.

V1's 512-grid results and V0-labelled 2x4/4x2 hybrid, MPI8 and OpenMP8 outputs are identifiable in the source branch. Exact per-job source hashes/allocations and the final report are missing. A distinct profiled V0 implementation was not recovered; handoff is unprofiled and cannot stand in for it. These commands do not reconstruct V0-to-V1 speedups. The original profiling README's claim that results are empty is stale.

## Correctness and safety scope

From this variant directory, inside a compute allocation with four MPI slots and eight CPU cores (two threads per rank):

```bash
ctest --test-dir build --output-on-failure --no-tests=error -j 1
```

CTest runs original test_hybrid_ops at two ranks and test_hybrid_cg at two/four ranks. Each test has a 90-second timeout; Release test assertions are enabled. Numerical source and test contents are unchanged. Original runner/library failure paths are retained, not certified collectively safe; use identical valid arguments on every rank. Allocation failures, malformed input and unsupported sizes may expose historical limitations. No integrated safety reductions have been imported. Independent verification on NUS Vanda on 13 November 2026 confirmed successful standalone compilation, all three registered CTests passing, and representative Hybrid benchmark execution. A two-rank, four-thread-per-rank Poisson 2D64 benchmark converged in 122 iterations with a true relative residual of approximately 8.71368e-09. Per-rank profiling CSVs were generated successfully, and the original profile aggregation script completed. See [Vanda verification](../../docs/VANDA_VERIFICATION_2026-11-13.md). These functional tests do not reproduce historical V0-to-V1 performance claims or certify every MPI failure path.

## Inputs and outputs

Poisson matrices are generated. CLI: `run_hybrid_benchmark <2d|3d|thermal2> <grid|matrix.mtx> <threads> <static|dynamic|guided> <chunk> <repeats> <output.csv> [warmups]`. Tolerance is 1e-8 and maximum iterations 10000; omitted warmups default to one. Use explicit positive thread counts matching the allocation. The original manufactured RHS is A*ones with zero initial guess. The 14-column benchmark CSV records configuration, iterations, convergence, time, true residual, solution error, schedule and chunk. Use unique output filenames; writers truncate existing files. Inspect convergence and residuals before calculating performance summaries.

No historical CSV is required. Optional thermal2 input can be acquired with the unchanged `scripts/download_thermal2.ps1` from this variant directory:

```powershell
pwsh -File scripts/download_thermal2.ps1 -Destination data/suitesparse
```

This requires PowerShell, tar and network access; transfer the resulting matrix to Vanda if needed. Pass data/suitesparse/thermal2/thermal2.mtx as the CLI matrix path. Download execution and a pinned expected input hash are not verified. No identifiable Task C thermal2 performance campaign is claimed here.
