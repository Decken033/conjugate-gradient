# Task B source provenance and acceptance scope

Production baseline: `56ac3602977385ec39b142f4a96e8cd833c8665a`. Recovered PBS recipe storage revision: `2e1c4913339efd0ea456c9150b524cca4dc2fb2e`. This uses the Phase 1 manifest, not newer integrated numerical sources.

## Exact inventory and comparisons

Nineteen of the 22 original source/header/test files are byte-identical. Exactly three production files differ as described below. Paths are task_b-relative and match baseline source paths. The downloader is also copied byte-for-byte.

| File | Baseline Git blob | Comparison |
|---|---|---|
| `include/cg_solver.hpp` | `34e2f41cbcf8c98465cda518f7ec0c9d13f9e3bd` | exact |
| `include/csr_matrix.hpp` | `da2d608e2a5de7396dddc129b7bb5fbb94e80c85` | exact |
| `include/csr_ops.hpp` | `639ad6a31059a7cffea1366305da214356e9106d` | exact |
| `include/matrix_market.hpp` | `938b3dc879d828f20b1caa07a34998b184544006` | exact |
| `include/mpi_benchmark.hpp` | `d8190c2f613bfe710eb7d63cdc362febc06c8eed` | exact |
| `include/mpi_ops.hpp` | `8914c62642e8dc899f625d38fe78b3a4b5c841e0` | exact |
| `include/mpi_partition.hpp` | `7eb8bc5f42ec164518625abd18504639ed3581b4` | exact |
| `include/mpi_solver.hpp` | `2cd8d65904349d876336e02eb95a32084e1f5d12` | exact |
| `include/poisson.hpp` | `3bbf8e55bf7ea2663861fc67d9dac8d05e5ff58c` | exact |
| `src/common/csr_ops.cpp` | `f8dea6e0f84421e3dfe1290383d1af3df0a3d558` | exact |
| `src/common/matrix_market.cpp` | `e99b29d1dcb4f5de9be5ae8f2def8b100725dab2` | exact |
| `src/common/poisson.cpp` | `4a0904fa3987d78297287c23c737e6e68761906c` | exact |
| `src/mpi/benchmark_mpi.cpp` | `b24b41a3d1d53d0d85f747542137d13284580595` | patched; see below |
| `src/mpi/benchmark_mpi_csv.cpp` | `6cacf3eb508b75e77bc13ed0d5c2876ff146f229` | exact |
| `src/mpi/mpi_ops.cpp` | `5910077e0d561f3d7da85b8b9b5b6092fd6406db` | exact |
| `src/mpi/mpi_partition.cpp` | `443f343983abc5d9632a9264c4ae1570934c88c9` | patched; see below |
| `src/mpi/mpi_solver.cpp` | `d7b32a4f86ec208fad2e853cb78ba5542cd2c783` | exact |
| `src/mpi/run_mpi_benchmark.cpp` | `acbc56d5f155e9188c2ec377e747d79438f49a43` | patched; see below |
| `src/serial/cg_serial.cpp` | `69b05efdf291f66b2d641d2e27111abecf91c74f` | exact |
| `tests/test_mpi_cg.cpp` | `88eff24c7f941660ada95688387b042d5e9e161f` | exact |
| `tests/test_mpi_partition.cpp` | `613d6f3f6e4251c8fe8b3cab9a6eefe32a850584` | exact |
| `tests/test_mpi_spmv.cpp` | `ffee40fbe34ef1df01fa6e79b1147ef979444e07` | exact |

## Three safety deviations

1. src/mpi/run_mpi_benchmark.cpp: invalid argument count and zero repeats throw into the communicator-abort handler instead of rank-local finalization. Strict unsigned parsing rejects malformed controls; optional warmups/defaults and CLI arity remain unchanged. Five effective argument fields are broadcast/compared with one agreement reduction each, before matrix setup. Grid dimension multiplication is checked against int row limits before allocation; name broadcast counts are checked before conversion. Caught standard and unknown exceptions call MPI_Abort after initialization, covering setup, distribution, solver and output failures. Rank zero writes original CSV records, detects any measured nonconvergence and broadcasts the final exit code before normal finalization. The existing unknown-matrix-type exit remains safe because all ranks receive the same exit code before finalizing.
2. src/mpi/benchmark_mpi.cpp: remove the local nonempty guard around solution-error calculation. Every rank now contributes to the same existing Allreduce, with zero sums on empty ranks. This is after t_end. No other benchmark code or measured interval changed.
3. src/mpi/mpi_partition.cpp: require the historical LP64 metadata ABI at compile time. On rank zero before distribution, reject oversized row/rank counts and check each partition's rows+1 and NNZ against MPI int-count limits. The preflight uses the original partition formula, adding no MPI collective. Under the 64-bit ABI and bounded rows/ranks, the original intermediate offset arithmetic does not overflow. Partition extraction/distribution and gather/scatter algorithms are unchanged. A preflight failure relies on the runner's abort handler; this is not a new collective-error API guarantee for arbitrary callers.

The added setup agreement is outside warmups and timing. No changes were made to mpi_solver.cpp or mpi_ops.cpp, including their original reductions and validation. The complete timed text from MPI_Barrier through the max-elapsed MPI_Reduce is unchanged. Partition preflight and final exit-code broadcast are outside the measured solver region. On previously valid nonempty-rank runs, post-solve numerical reduction order is preserved. System/cache effects of setup cannot be promised identical, and source-level preservation does not establish bitwise or performance runtime equivalence.

## Historical provenance

56ac360 is the closest available pre-safety implementation for reported measurements, not an immutable per-job source proof. Member B's later 72c140f validation changes and integration/acceptance reductions are deliberately not imported. Recovered scripts and evidence at 2e1c491 identify configuration history, not the exact historical build hash for every job. Old CSVs/logs remain elsewhere unchanged and are not execution dependencies.

## Packaging and tests

- New Task B CMake extracts original MPI production targets and link dependencies, makes MPI required and omits OpenMP/GPU/hybrid. Serial/CSR reference libraries are test-only. Original three test files are unchanged; CTest registers each at 1/2/4 ranks with Release assertions and 90-second timeouts.
- New test_mpi_benchmark checks empty-rank post-timing reductions, zero-iteration nonconvergence records and recovery at four ranks. It does not change the production CLI's fixed 10000-iteration limit.
- New check_runner.py invokes the production runner with bounded subprocesses; requires diagnostic plus nonzero exit for negatives and checks successful MPMD execution/CSV before asymmetric argument failures. CTest timeout 360 seconds. OpenMPI-specific syntax is checked explicitly. No mocked MPI test is claimed as runtime verification.
- The benchmark regression tests nonconvergence record semantics through the production benchmark API, not a forced end-to-end CLI iteration-limit exhaustion. The runner's nonconvergence exit branch is statically inspected; runtime coverage of that branch remains a limitation.
- Seven recovered PBS files preserve original allocations, grids/ranks, repeats/warmups, modules and placement assumptions. Only source/account paths, recipe comments, obsolete CG_ENABLE_MPI options and the thermal2 dataset path are adapted. Outputs keep original task-relative naming; no replacement experimental design is introduced. The recovered correctness recipe retains MPI 4.1.8 versus benchmark MPI 4.1.6; use fresh caches for module changes.
- scripts/download_thermal2.ps1 is unchanged. Task-local .gitattributes preserves PBS LF endings. README is new standalone documentation. No original root, Task A/C, evidence or branch content is changed.

## Validation and remaining limitations

Compilation, CTest/MPI execution and benchmark runtime are NOT VERIFIED: native and WSL toolchain probes have not provided a usable CMake/MPI compiler stack. Source comparisons, quoted project include closure, target/source-path checks, Python syntax, PBS Bash syntax and whitespace checks are performed during migration. Evidence validators passed: 76 hashes, 60 CSVs, 600 outcomes; report validation passed with optional local DOCX absent. Evidence validation executes no solver.

The library still assumes cross-rank control consistency and valid operation inputs; rank-local exceptions outside the supplied runner can deadlock peers. Fatal errors are aborted, not recovered. Count guards do not provide large-count support or recover from OOM/process/MPI failures. Thermal2 placement and a pinned verified input hash remain unspecified; historical module availability and numerical/performance behaviour must be tested on Vanda.
