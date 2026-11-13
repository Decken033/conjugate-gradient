> **Historical preparation-stage manifest:** This document records the repository preparation state when it was authored, not the current runtime verification status. Statements such as "NOT VERIFIED" and "not yet implemented" describe that earlier stage. Independent packaging and NUS Vanda functional verification were subsequently completed. See the [13 November 2026 Vanda verification results](docs/VANDA_VERIFICATION_2026-11-13.md) for the latest recorded verification and [README.md](README.md) for current standalone package instructions. Original preparation facts, inventories and source-selection reasoning below are retained unchanged.

> Historical context: source paths and unified-build commands describe the original revisions. For current independent builds and experiments, use the [standalone submission README](README.md). Removed paths and archived snapshots are recorded in docs/archive/cleanup_manifest.json (repository-relative). Historical results are not verification of the standalone packages.

# Standalone submission preparation manifest

Preparation only. No source migration, build configuration rewrite, numerical change, timing change, commit or push has been performed. File tables identify existing Git objects; they are not claims of runtime verification. Paths are relative to the source repository at the stated revision.

## Verified references and workspace

- Integration backup: `a9911a0caff88b2d3859b24f73f5b56997d08f56`, branch `integration/task-c-hybrid`; original worktree clean before preparation.
- Original baseline: `56ac3602977385ec39b142f4a96e8cd833c8665a`.
- Locally available `origin/main`: `2e1c4913339efd0ea456c9150b524cca4dc2fb2e`; no fetch performed.
- Member C branch `origin/marssiiiii-patch-1`: `846239dbb7fe04daf56017adc6e8d7c18bda0f15`.
- Submission branch: `submission/standalone-a-b-c`, based on `2e1c4913339efd0ea456c9150b524cca4dc2fb2e` in sibling worktree `conjugate-gradient-standalone-submission`.
- The new worktree initially contains the full main checkout. This is intentional preparation, not the final directory layout. Only this manifest is added.

## Required target inventory

Task A: `cg_common`, `cg_ops`, `cg_poisson`, `cg_matrix_market`, `cg_serial`, `cg_serial_benchmark`, `run_serial_benchmark`, `cg_openmp_ops`, `cg_openmp`, `cg_openmp_benchmark`, `run_openmp_benchmark`; optional `cg_gpu`, `cg_gpu_benchmark`, `run_gpu_benchmark`. Tests: `test_csr`, `test_csr_ops`, `test_matrix_market`, `test_serial_cg`, `test_serial_benchmark`, `test_openmp_ops`, `test_openmp_cg`, optional `test_gpu_cg`.

Task B: local common/Poisson/Matrix Market targets, `cg_mpi_ops`, `cg_mpi`, `cg_mpi_benchmark`, `run_mpi_benchmark`; local `cg_ops` and `cg_serial` for tests. Tests: `test_mpi_partition`, `test_mpi_spmv`, `test_mpi_cg` (historical CTest registrations use four ranks).

Task C: local common/Poisson/Matrix Market/MPI ops dependencies, `cg_hybrid_ops`, `cg_hybrid`, `cg_hybrid_benchmark`, `run_hybrid_benchmark`; `cg_ops` for operations comparison. Tests: `test_hybrid_ops`, `test_hybrid_cg` (V1 CTest uses two ranks; documentation also calls for CG at four ranks). Conditional profiled comparator targets: `cg_mpi`, `cg_mpi_benchmark`, `run_mpi_benchmark`, `cg_openmp_ops`, `cg_openmp`, `cg_openmp_benchmark`, `run_openmp_benchmark`.

Dependencies: C++17 and CMake >=3.20 in source configurations; CPU OpenMP for A/C, MPI for B/C, NVHPC plus NVIDIA GPU for optional A offload. Python is needed by profile summarization. Existing root CMake also requires OpenMP when building MPI; standalone dependency separation has not been implemented or tested.

## Task C source-selection findings ? unresolved

The handoff RUN_REPORT documents exploratory WSL2 Ubuntu 24.04 runs (GCC 13.3, OpenMPI 4.1.6, CMake 3.28.3): 2D grids 64/256, two ranks and four threads, five measured runs and one warmup. It explicitly says the planned 512 comparison had not been run. Its SpMV allocates communication metadata and buffers repeatedly.

Member C's V1 documentation describes per-solve reusable workspace and per-rank operation profiling. The supplied script runs 2D-512, two ranks/four threads, static schedule/chunk zero, ten measured runs and five warmups, core binding and slot:PE=4 mapping. README_V1_PROFILING specifies CMake 3.27.6/GCCcore 13.2 and OpenMPI 4.1.8 modules. Profiling is inside timed solves; sidecar writes are after the benchmark. V1 and handoff timings are not interchangeable.

The branch contains V1 outputs and results/v1-profile/logs_profiling_v0 outputs for hybrid 2x4 and 4x2, MPI 8 ranks and OpenMP 8 threads. README_V1_PROFILING explicitly discusses V0-to-V1 comparison. Output presence is evidence of experiments, not proof the final report selected those experiments. README_PROFILING's claim that results are empty is stale.

No final consolidated Task C/team report is identified in these source trees. Need the final Task C section/tables/figures, identifying whether it reports handoff scheduling, V1 profiling, or V0-to-V1 improvements; the source/build provenance for every reported configuration; and the exact profiled V0 source snapshot and matched allocation/module/binding settings if V0 comparisons are retained. Inspected hybrid history introduces V1 workspace and V0 result files together; a distinct executable profiled V0 source revision has not been established. Do not reconstruct V0 by guessing or substitute current integration.

## Blockers and migration gates

- Resolve Task C final-report selection and recover profiled V0 source if needed.
- Task A baseline preserves original production code, but archived serial campaign hashes do not establish an exact complete baseline build (header/build-file differences were found in the prior audit). Retain this provenance qualification; recover exact recipes for report experiments where absent.
- Task B baseline predates Member B safety changes at 72c140f and subsequent acceptance changes. Historical error/empty-rank handling is unsafe and nonconvergence can report success. Agree on separately documented minimal corrections before public reproduction; do not import additional timed validation collectives without relabelling performance methodology.
- Poisson inputs are generated. thermal2 requires download instructions and a pinned verified input hash; the existing PowerShell downloader is not a complete Vanda/Linux recipe. Historical CSV/logs are not runtime inputs.
- Extract task-local CMake configurations and parameterize personal PBS paths only in a later phase. Preserve scheduling, repetitions, warmups, compiler flags, binding, timing boundaries and profiling settings. Check Release assertions and bounded MPI failure tests then.
- No new build/run commands are asserted valid here. Compilation and MPI/GPU execution are NOT VERIFIED in this preparation phase.

## Task A original production and dependencies

Source revision: `56ac3602977385ec39b142f4a96e8cd833c8665a`. 

| Source path | Git blob |
|---|---|
| `include/cg_solver.hpp` | `34e2f41cbcf8c98465cda518f7ec0c9d13f9e3bd` |
| `include/csr_matrix.hpp` | `da2d608e2a5de7396dddc129b7bb5fbb94e80c85` |
| `include/csr_ops.hpp` | `639ad6a31059a7cffea1366305da214356e9106d` |
| `include/gpu_benchmark.hpp` | `d5b57221ecbb576a5b4ae24adf6c47ae88f13466` |
| `include/gpu_solver.hpp` | `2357be4d2f9de6095b2cefce845f3ef93b271c01` |
| `include/matrix_market.hpp` | `938b3dc879d828f20b1caa07a34998b184544006` |
| `include/openmp_benchmark.hpp` | `db594e01d0a135f4c9697aeeb96895d7e81c9325` |
| `include/openmp_ops.hpp` | `0e581e21bd05ffdddccb6ffdf48453dc10dc90ea` |
| `include/openmp_solver.hpp` | `8a6b6c8dbcb2969c93d2251e84aceb921001f874` |
| `include/poisson.hpp` | `3bbf8e55bf7ea2663861fc67d9dac8d05e5ff58c` |
| `include/serial_benchmark.hpp` | `9b5863f91a7febc8af79217def0b0bbacbee8c12` |
| `src/common/csr_ops.cpp` | `f8dea6e0f84421e3dfe1290383d1af3df0a3d558` |
| `src/common/matrix_market.cpp` | `e99b29d1dcb4f5de9be5ae8f2def8b100725dab2` |
| `src/common/poisson.cpp` | `4a0904fa3987d78297287c23c737e6e68761906c` |
| `src/gpu/benchmark_gpu.cpp` | `124dd0c17c4887075bb194dfc19781c655230c06` |
| `src/gpu/benchmark_gpu_csv.cpp` | `038fcf173399ac34060688d88800e7914c9d2c9a` |
| `src/gpu/cg_gpu.cpp` | `6478e9a7e996dd7b28353191f7e3b541476f6d64` |
| `src/gpu/run_gpu_benchmark.cpp` | `2f12b97f02ae552c6fb624245d9294353f791621` |
| `src/openmp/benchmark_openmp.cpp` | `0a52cccc7883eb6837b940a63b5c7f135326a960` |
| `src/openmp/benchmark_openmp_csv.cpp` | `bf7c75f1e59d97527fbf5790554f30acbce52912` |
| `src/openmp/cg_openmp.cpp` | `4021724280477523bca50b3bdd6f22a6776f0caf` |
| `src/openmp/openmp_ops.cpp` | `e0f78bb1e218a35eb82cb748f7504c60a91748dd` |
| `src/openmp/run_openmp_benchmark.cpp` | `7ad1835895c18381124d92b9d081911b7f6de35f` |
| `src/serial/benchmark_csv.cpp` | `5fd393f68a90ae499e659f0522b7d1f447716274` |
| `src/serial/benchmark_serial.cpp` | `fe4701b1ccb3aa05746fad71c5b0d63a0400a369` |
| `src/serial/cg_serial.cpp` | `69b05efdf291f66b2d641d2e27111abecf91c74f` |
| `src/serial/run_serial_benchmark.cpp` | `71b60ce073d71bb4696e336b6edd1bdb88fec440` |

## Task A build, tests and experiment recipes

Source revision: `56ac3602977385ec39b142f4a96e8cd833c8665a`. The root CMake file is a target-extraction reference, not an already standalone Task A build. Historical metadata are provenance references, not execution inputs.

| Source path | Git blob |
|---|---|
| `CMakeLists.txt` | `5d810aab62c5a4d9daab138cfca516ae689dbf39` |
| `results/gpu_evidence/gpu_scaling_full.pbs` | `135ad534362d18eca271049b9a5746e2823c33f1` |
| `results/serial_scaling/20261104T022346Z_1517629.stdct-mgmt-02_EOF0kD/campaign.pbs` | `939bc14286062e81e41562af0589361f2da5db78` |
| `results/serial_scaling/20261104T022346Z_1517629.stdct-mgmt-02_EOF0kD/commands.sh` | `58d4126a69d48bf4a5e17e080d4b845b577b866f` |
| `results/serial_scaling/20261104T022346Z_1517629.stdct-mgmt-02_EOF0kD/execution_order.csv` | `7b76ff8e49adac3243bb98560b5d976805be4b6c` |
| `results/serial_scaling/20261104T022346Z_1517629.stdct-mgmt-02_EOF0kD/metadata.txt` | `94d95265389004d805b70ee86c55c6ec3dff3ffd` |
| `results/serial_scaling/20261104T022346Z_1517629.stdct-mgmt-02_EOF0kD/source_sha256.txt` | `ed2d53d326926eceb3e6abdad46b7cb92aed3784` |
| `scripts/download_thermal2.ps1` | `c3ac1b63330f811305b2eaf4f6c2ea09d8d28ac9` |
| `scripts/vanda/benchmark_openmp_scaling_1024.pbs` | `8a13a13ab97d8f09eb6caee1627aa2512cd83bca` |
| `scripts/vanda/benchmark_openmp_scaling_72.pbs` | `a467bac180864c28ae74f2b491bd4e2a7869f5f9` |
| `scripts/vanda/benchmark_serial_scaling.pbs` | `939bc14286062e81e41562af0589361f2da5db78` |
| `tests/test_csr.cpp` | `f57f5ec33281b062feef9889a8cf282c561f9da4` |
| `tests/test_csr_ops.cpp` | `3cc6ef4fcbcfe66981c9ce833569550c92585c24` |
| `tests/test_gpu_cg.cpp` | `dc4d245912a8fb75f62a08435264b96eb80ff342` |
| `tests/test_matrix_market.cpp` | `90dc9ef830fb7e13c15c65e010d864f0f7d2d8b4` |
| `tests/test_openmp_cg.cpp` | `1061f415710e809ff7703cf5ea74454165b4cc65` |
| `tests/test_openmp_ops.cpp` | `3594fc03c96572f5b4447d9df248ff452ed95a1e` |
| `tests/test_serial_benchmark.cpp` | `4bcab61320390e94ae21b8de27c34dc10e3dd1e1` |
| `tests/test_serial_cg.cpp` | `cc627d3c07a92eeaa16e29076cc27659f9543236` |

## Task B pre-safety production and dependencies

Source revision: `56ac3602977385ec39b142f4a96e8cd833c8665a`. The serial solver and CSR operations support original cross-solver tests. This is the closest source candidate for historical measurements, not proof of an exact job checkout.

| Source path | Git blob |
|---|---|
| `include/cg_solver.hpp` | `34e2f41cbcf8c98465cda518f7ec0c9d13f9e3bd` |
| `include/csr_matrix.hpp` | `da2d608e2a5de7396dddc129b7bb5fbb94e80c85` |
| `include/csr_ops.hpp` | `639ad6a31059a7cffea1366305da214356e9106d` |
| `include/matrix_market.hpp` | `938b3dc879d828f20b1caa07a34998b184544006` |
| `include/mpi_benchmark.hpp` | `d8190c2f613bfe710eb7d63cdc362febc06c8eed` |
| `include/mpi_ops.hpp` | `8914c62642e8dc899f625d38fe78b3a4b5c841e0` |
| `include/mpi_partition.hpp` | `7eb8bc5f42ec164518625abd18504639ed3581b4` |
| `include/mpi_solver.hpp` | `2cd8d65904349d876336e02eb95a32084e1f5d12` |
| `include/poisson.hpp` | `3bbf8e55bf7ea2663861fc67d9dac8d05e5ff58c` |
| `src/common/csr_ops.cpp` | `f8dea6e0f84421e3dfe1290383d1af3df0a3d558` |
| `src/common/matrix_market.cpp` | `e99b29d1dcb4f5de9be5ae8f2def8b100725dab2` |
| `src/common/poisson.cpp` | `4a0904fa3987d78297287c23c737e6e68761906c` |
| `src/mpi/benchmark_mpi.cpp` | `b24b41a3d1d53d0d85f747542137d13284580595` |
| `src/mpi/benchmark_mpi_csv.cpp` | `6cacf3eb508b75e77bc13ed0d5c2876ff146f229` |
| `src/mpi/mpi_ops.cpp` | `5910077e0d561f3d7da85b8b9b5b6092fd6406db` |
| `src/mpi/mpi_partition.cpp` | `443f343983abc5d9632a9264c4ae1570934c88c9` |
| `src/mpi/mpi_solver.cpp` | `d7b32a4f86ec208fad2e853cb78ba5542cd2c783` |
| `src/mpi/run_mpi_benchmark.cpp` | `acbc56d5f155e9188c2ec377e747d79438f49a43` |
| `src/serial/cg_serial.cpp` | `69b05efdf291f66b2d641d2e27111abecf91c74f` |

## Task B original build, tests and scripts

Source revision: `56ac3602977385ec39b142f4a96e8cd833c8665a`. 

| Source path | Git blob |
|---|---|
| `CMakeLists.txt` | `5d810aab62c5a4d9daab138cfca516ae689dbf39` |
| `scripts/download_thermal2.ps1` | `c3ac1b63330f811305b2eaf4f6c2ea09d8d28ac9` |
| `scripts/vanda/benchmark_mpi_scaling.pbs` | `8114944ad01c10659f58caca4a1753f0e1cfa72c` |
| `scripts/vanda/benchmark_mpi_scaling_1024.pbs` | `bac49d11681f99eb0ec9b43a38fba5c431e8f83c` |
| `scripts/vanda/benchmark_mpi_small.pbs` | `bf4cdcddb7755d50955f334bf5f5a3c1c26c0e7d` |
| `scripts/vanda/benchmark_mpi_thermal2.pbs` | `7533e0733be6ce97c3a8228be7c93597c64c0174` |
| `scripts/vanda/build_mpi.pbs` | `ce7c4e944439aac422375860c6bf88b9a1e510a1` |
| `scripts/vanda/test_mpi_correctness.pbs` | `52a91e54120e67d180e0ca733e96408cb1807b32` |
| `scripts/vanda/validate_mpi.pbs` | `88511177acbc7ed9d81822d68222c3544e6cd387` |
| `tests/test_mpi_cg.cpp` | `88eff24c7f941660ada95688387b042d5e9e161f` |
| `tests/test_mpi_partition.cpp` | `613d6f3f6e4251c8fe8b3cab9a6eefe32a850584` |
| `tests/test_mpi_spmv.cpp` | `ffee40fbe34ef1df01fa6e79b1147ef979444e07` |

## Task B recovered experiment recipes

Source revision: `2e1c4913339efd0ea456c9150b524cca4dc2fb2e`. These recovered recipes supplement the baseline. Their current storage revision does not prove the historical jobs compiled this revision.

| Source path | Git blob |
|---|---|
| `docs/task_b/evidence/scripts/benchmark_mpi_scaling.pbs` | `8114944ad01c10659f58caca4a1753f0e1cfa72c` |
| `docs/task_b/evidence/scripts/benchmark_mpi_scaling_1024.pbs` | `bac49d11681f99eb0ec9b43a38fba5c431e8f83c` |
| `docs/task_b/evidence/scripts/benchmark_mpi_scaling_3d.pbs` | `4545998487b28df16be9e8dc72849b5ca44a327f` |
| `docs/task_b/evidence/scripts/benchmark_mpi_strong_256.pbs` | `9ef3334998f66ff6de1065ac47e2a42d30401e84` |
| `docs/task_b/evidence/scripts/benchmark_mpi_thermal2.pbs` | `3475f5f24d9c49b6f025e708257828bc46fe30ca` |
| `docs/task_b/evidence/scripts/benchmark_mpi_weak_256.pbs` | `03511c1363f3136ea1aa418ebc454e20f49cee20` |
| `docs/task_b/evidence/scripts/test_mpi_correctness.pbs` | `52a91e54120e67d180e0ca733e96408cb1807b32` |

## Task C handoff supplied source and tests

Source revision: `56ac3602977385ec39b142f4a96e8cd833c8665a`. Candidate only; no Task C version selected. All supplied dependency headers are inventoried even where not needed by the final link closure.

| Source path | Git blob |
|---|---|
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/dependencies/common/csr_matrix.hpp` | `da2d608e2a5de7396dddc129b7bb5fbb94e80c85` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/dependencies/common/csr_ops.hpp` | `9bac833084a6bd78ebc3cdd35947b562a6c2854b` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/dependencies/common/matrix_market.hpp` | `9eaba8baf3509100755bcc98699af2c0871cf0a0` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/dependencies/common/poisson.hpp` | `f3670ed5e9edfd815bc4fd53e7eec712842c6a78` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/dependencies/mpi/mpi_benchmark.hpp` | `e73f1585f29e8bf953d56beb684d3d9c04355d24` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/dependencies/mpi/mpi_ops.hpp` | `eacb968ba3b2f71ef214ed5a11d7eb7557edc70b` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/dependencies/mpi/mpi_partition.cpp` | `1d14c5a21645c35b028139a0c78608bf9522e27b` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/dependencies/mpi/mpi_partition.hpp` | `e9f16c0732a12c50cb91467f54099ed701156359` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/dependencies/mpi/mpi_solver.hpp` | `6ced0eb8a83a7fa165a57f4dbc6f9d198fd636cb` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/include/hybrid/hybrid_benchmark.hpp` | `72f380cf2c9e09a149822d750dbd27d94a1473c2` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/include/hybrid/hybrid_ops.hpp` | `e2bff822f39a6a2fd6c7cac81e372620392beae2` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/include/hybrid/hybrid_solver.hpp` | `e9693583606599add381f8188afa8983d049f3f5` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/src/hybrid/benchmark_hybrid.cpp` | `e42de3b1bf5dfffdbb76fcec75995c148298b0bd` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/src/hybrid/benchmark_hybrid_csv.cpp` | `89293a7f2866592e1471dcdce01f4f40fe09826d` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/src/hybrid/hybrid_ops.cpp` | `680f31775f68cb3ac03b04f5f701eb9114a5becc` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/src/hybrid/hybrid_solver.cpp` | `203da0ba87b41c0dd28e150ec56a615c4ce0159e` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/src/hybrid/run_hybrid_benchmark.cpp` | `5e8424ba69d4c931df290d057334fbccff2af66d` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/tests/test_hybrid_cg.cpp` | `ca76f8f6371b79a65ca7f6009601b79d0e451154` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/tests/test_hybrid_ops.cpp` | `444cf03d4980f017db70cb6c601b9d5b1b0cced9` |

## Task C handoff missing dependency candidates

Source revision: `56ac3602977385ec39b142f4a96e8cd833c8665a`. Handoff alone is incomplete. These baseline files are candidate closure inputs; nested include paths require explicit reconciliation during migration. Do not silently substitute later safety kernels.

| Source path | Git blob |
|---|---|
| `include/cg_solver.hpp` | `34e2f41cbcf8c98465cda518f7ec0c9d13f9e3bd` |
| `include/csr_matrix.hpp` | `da2d608e2a5de7396dddc129b7bb5fbb94e80c85` |
| `include/csr_ops.hpp` | `639ad6a31059a7cffea1366305da214356e9106d` |
| `include/matrix_market.hpp` | `938b3dc879d828f20b1caa07a34998b184544006` |
| `include/openmp_ops.hpp` | `0e581e21bd05ffdddccb6ffdf48453dc10dc90ea` |
| `include/poisson.hpp` | `3bbf8e55bf7ea2663861fc67d9dac8d05e5ff58c` |
| `src/common/csr_ops.cpp` | `f8dea6e0f84421e3dfe1290383d1af3df0a3d558` |
| `src/common/matrix_market.cpp` | `e99b29d1dcb4f5de9be5ae8f2def8b100725dab2` |
| `src/common/poisson.cpp` | `4a0904fa3987d78297287c23c737e6e68761906c` |
| `src/mpi/mpi_ops.cpp` | `5910077e0d561f3d7da85b8b9b5b6092fd6406db` |

## Task C handoff documentation and build reference

Source revision: `56ac3602977385ec39b142f4a96e8cd833c8665a`. Baseline root CMake does not build hybrid. A task-specific configuration must be prepared and verified later; the handoff references a hybrid CMake file not supplied in this folder.

| Source path | Git blob |
|---|---|
| `CMakeLists.txt` | `5d810aab62c5a4d9daab138cfca516ae689dbf39` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/README.md` | `aedc529bb80c9c818cfbf52ac02403471df35f08` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/docs/ALGORITHM_AND_PROJECT_INTEGRATION.md` | `1e72977788f736947ae0c8ff2c384f680da39034` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/docs/IMPLEMENTATION_GUIDE.md` | `d328bdc58f84ecc04446ab9be44f5bea69558beb` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/docs/RUN_GUIDE.md` | `97077e13b8c0cbfa5deb0f28252d462c477083b1` |
| `Hybrid_Team_Handoff/Hybrid_Team_Handoff/docs/RUN_REPORT.md` | `9cc8f477cb347cb19c97fc54ef39233b0f147add` |

## Task C V1 production and dependency closure

Source revision: `846239dbb7fe04daf56017adc6e8d7c18bda0f15`. Includes reusable workspace and profiling implementation; not interchangeable with the unprofiled handoff or integrated hybrid.

| Source path | Git blob |
|---|---|
| `include/common/csr_matrix.hpp` | `da2d608e2a5de7396dddc129b7bb5fbb94e80c85` |
| `include/common/csr_ops.hpp` | `9bac833084a6bd78ebc3cdd35947b562a6c2854b` |
| `include/common/matrix_market.hpp` | `9eaba8baf3509100755bcc98699af2c0871cf0a0` |
| `include/common/ops_profiler.hpp` | `c745208253f0d356a22839432adb3b796bf54f57` |
| `include/common/poisson.hpp` | `f3670ed5e9edfd815bc4fd53e7eec712842c6a78` |
| `include/hybrid/hybrid_benchmark.hpp` | `72f380cf2c9e09a149822d750dbd27d94a1473c2` |
| `include/hybrid/hybrid_ops.hpp` | `82e99defef7b2f9037d22b1609af03b713e2c43c` |
| `include/hybrid/hybrid_solver.hpp` | `e9693583606599add381f8188afa8983d049f3f5` |
| `include/hybrid/hybrid_workspace.hpp` | `ba24c3718b8448023acc3f1e5fa6f93980898849` |
| `include/mpi/mpi_ops.hpp` | `eacb968ba3b2f71ef214ed5a11d7eb7557edc70b` |
| `include/mpi/mpi_partition.hpp` | `e9f16c0732a12c50cb91467f54099ed701156359` |
| `include/openmp/openmp_ops.hpp` | `6bd5fb2dc066feb8ead2303b014566fb193886ed` |
| `include/serial/cg_solver.hpp` | `83173e0cd59f5fe310a4d4aad49e5c101c8f1c78` |
| `src/common/csr_ops.cpp` | `346c1757f0bda98078d8399dbd45b82f29d3d890` |
| `src/common/matrix_market.cpp` | `aefde2544146a49b27e0bdde2564a02945dc28cc` |
| `src/common/poisson.cpp` | `19e70770fe46d7d3002c503257526814fb4ab87e` |
| `src/hybrid/benchmark_hybrid.cpp` | `3965d88840425325e4803ca22b83faedb8fbe0d4` |
| `src/hybrid/benchmark_hybrid_csv.cpp` | `89293a7f2866592e1471dcdce01f4f40fe09826d` |
| `src/hybrid/hybrid_ops.cpp` | `efe5617e9150c09ed4cb50df079aa0c95d8a6358` |
| `src/hybrid/hybrid_solver.cpp` | `47c90ab1096c8b628cc8fba1394920bf425ec131` |
| `src/hybrid/hybrid_workspace.cpp` | `7623820e931ac21498ce50caf46b79cff2dd589d` |
| `src/hybrid/run_hybrid_benchmark.cpp` | `5fe65f9b5fab2ad23c2fd998802ffaf9f37b6f62` |
| `src/mpi/mpi_ops.cpp` | `87f5af448383b95eb7c5f47ca0c34b037b5f83c7` |
| `src/mpi/mpi_partition.cpp` | `1d14c5a21645c35b028139a0c78608bf9522e27b` |

## Task C V1 build, tests, recipes and interpretation

Source revision: `846239dbb7fe04daf56017adc6e8d7c18bda0f15`. 

| Source path | Git blob |
|---|---|
| `CMakeLists.txt` | `e19070f4ee56bc00daaec36ef363936c2f99643c` |
| `README_PROFILING.md` | `d6950c6917a6f48efb8946f6af8ee99cceefeac7` |
| `README_V1_PROFILING.md` | `cc32bc48677d459a97bb1c1bd41b23caa4e3d1b5` |
| `docs/hybrid/IMPLEMENTATION_GUIDE.md` | `d328bdc58f84ecc04446ab9be44f5bea69558beb` |
| `docs/profiling/OPS_PROFILING_GUIDE.md` | `9406388a2b7c604edba231b72d86007efe5ddc0c` |
| `scripts/run_hybrid_v1_profile_2x4.sh` | `f9425858c0b99d20d06e9aa30139e9077b29797d` |
| `scripts/summarize_profiles.py` | `725bb33dc205fcbe92a8853f15d2827ce2dc465b` |
| `tests/test_hybrid_cg.cpp` | `ca76f8f6371b79a65ca7f6009601b79d0e451154` |
| `tests/test_hybrid_ops.cpp` | `160010ee9e4024cf1b34e374eb8d5d927c69c3ab` |

## Task C conditional profiled comparison implementations

Source revision: `846239dbb7fe04daf56017adc6e8d7c18bda0f15`. Required only if the final Task C report uses MPI-only/OpenMP-only profiling comparisons. Keep these revision-specific implementations within Task C; Task A/B binaries do not reproduce this instrumentation.

| Source path | Git blob |
|---|---|
| `include/mpi/mpi_benchmark.hpp` | `e73f1585f29e8bf953d56beb684d3d9c04355d24` |
| `include/mpi/mpi_ops.hpp` | `eacb968ba3b2f71ef214ed5a11d7eb7557edc70b` |
| `include/mpi/mpi_partition.hpp` | `e9f16c0732a12c50cb91467f54099ed701156359` |
| `include/mpi/mpi_solver.hpp` | `6ced0eb8a83a7fa165a57f4dbc6f9d198fd636cb` |
| `include/openmp/openmp_benchmark.hpp` | `1ea8778ac10a5bd0c262d6a7fedf664831779c18` |
| `include/openmp/openmp_ops.hpp` | `6bd5fb2dc066feb8ead2303b014566fb193886ed` |
| `include/openmp/openmp_solver.hpp` | `6095fd7e83ac6d9d0c298ea4d9b89cab8cd0cbd1` |
| `src/mpi/benchmark_mpi.cpp` | `82ddb48b7ade580677aff39c39d1cd0dda236fff` |
| `src/mpi/benchmark_mpi_csv.cpp` | `2d264a0e618188dbd8a3da5a4b9ea39823c5da23` |
| `src/mpi/mpi_ops.cpp` | `87f5af448383b95eb7c5f47ca0c34b037b5f83c7` |
| `src/mpi/mpi_partition.cpp` | `1d14c5a21645c35b028139a0c78608bf9522e27b` |
| `src/mpi/mpi_solver.cpp` | `7239b80569b0f5878312496dcd29ccdc13b7797d` |
| `src/mpi/run_mpi_benchmark.cpp` | `893efa549c6605028f50225f843b823ce0a2979b` |
| `src/openmp/benchmark_openmp.cpp` | `7fd5dcf973199ca60aafd1afaffecbc831645a9a` |
| `src/openmp/benchmark_openmp_csv.cpp` | `fe7484b5ad4937751d8185fcda07f56ccabe1260` |
| `src/openmp/cg_openmp.cpp` | `de49e69c7caa1277337c5d3403c5fd8257efe7d1` |
| `src/openmp/openmp_ops.cpp` | `741546a529dc907f57abc120efce3c4f9a9728f6` |
| `src/openmp/run_openmp_benchmark.cpp` | `4c056d09ef9c4a0bfb518779d8fa848bca66580c` |
