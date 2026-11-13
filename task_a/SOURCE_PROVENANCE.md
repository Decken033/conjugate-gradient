# Task A source provenance

Baseline: `56ac3602977385ec39b142f4a96e8cd833c8665a`.
All 27 production/header files and eight tests below were extracted directly from Git blobs and verified byte-for-byte against that revision (including original line endings). No solver recurrence, timing, runner, output schema or test source was edited. All quoted project includes resolve within task_a.

## Packaging changes

- Task A-only CMake derived from baseline root CMake: remove MPI discovery, libraries and tests. Retain original serial/OpenMP/GPU target names, dependencies and production compile/link options, C++17 and GPU default OFF.
- Add assertion-enabling compile options to every test target in Release. No production optimization override is added by CMake. CPU recipe explicitly uses historical -O3; GPU recipe explicitly uses historical -O2.
- Copy serial/OpenMP PBS recipes from baseline scripts/vanda and GPU recipe from results/gpu_evidence/gpu_scaling_full.pbs. Replace personal source/account paths with TASK_A_DIR and qsub account selection; parameterize output paths and prebuilt binary directories, and thermal2 location. Remove obsolete CG_ENABLE_MPI=OFF from the serial configure command because this package has no MPI option. Preserve allocations, modules, binding, workloads, order, schedules, ten repetitions and four warmups.
- The baseline serial scripts/vanda recipe and archived campaign.pbs are the same Git blob; only one copy is packaged. Do not require archived logs, commands or CSVs to execute it.
- Add task-local .gitattributes to retain LF line endings for PBS Bash scripts on future checkouts.
- Copy scripts/download_thermal2.ps1 unchanged. New English README and this provenance document describe the package. No root implementation or other task was modified.

## Experimental provenance qualifications

The baseline is a source preservation reference, not proof every report experiment compiled an identical checkout. The earlier audit of archived serial source_sha256.txt found 21 of 24 comparable files matched after CRLF normalization; include/csr_matrix.hpp, include/openmp_benchmark.hpp and CMakeLists.txt differed. Historical source contents for those differences are not supplied. This package deliberately preserves the requested Git baseline and records rather than guesses those differences.

The original full OpenMP scheduling PBS script is unavailable. Report documentation identifies 2D512/eight threads and static/dynamic/guided/chunk comparisons, with start/end static drift; full job settings and provenance must be recovered before claiming exact scheduling-campaign reproduction. No invented scheduling script is included.

Task A scaling recipes preserve recorded workload parameters. Machine allocation, software modules, GPU availability and dataset download must be checked on Vanda. No pinned verified thermal2 hash is supplied. Historical CSVs/logs are not inputs. No historical evidence was changed.

## Verification

Migration-stage checks passed: byte comparisons (35 files), local quoted-include closure, Task A-only build-source and link dependency inspection, all four PBS scripts via bash -n through WSL, git diff --check and per-file untracked whitespace checks. Native Windows and WSL probes lacked a CMake/C++ compiler toolchain at that stage; no benchmark measurements were generated during migration.

Subsequent independent verification on NUS Vanda on 13 November 2026 confirmed a successful CPU build and all seven CPU CTests passing. The GPU build succeeded with NVHPC 24.7 on an NVIDIA A40, and the GPU correctness CTest passed (1/1). Serial, OpenMP and GPU benchmark smoke tests converged. Actual GPU kernel launches were confirmed using NVCOMPILER_ACC_NOTIFY=1. The tested NVHPC environment required an explicit --gcc-toolchain override to resolve GCC startup-object paths; this is environment-specific. See [Vanda verification](../docs/VANDA_VERIFICATION_2026-11-13.md).

These results establish functional verification of the tested configurations, not full reproduction of the historical benchmark campaigns. Historical source/build qualifications above remain unchanged.

## Exact preserved file inventory

Paths are relative to task_a and also identify their original baseline paths.

| File | Original Git blob | Comparison |
|---|---|---|
| `include/cg_solver.hpp` | `34e2f41cbcf8c98465cda518f7ec0c9d13f9e3bd` | exact |
| `include/csr_matrix.hpp` | `da2d608e2a5de7396dddc129b7bb5fbb94e80c85` | exact |
| `include/csr_ops.hpp` | `639ad6a31059a7cffea1366305da214356e9106d` | exact |
| `include/gpu_benchmark.hpp` | `d5b57221ecbb576a5b4ae24adf6c47ae88f13466` | exact |
| `include/gpu_solver.hpp` | `2357be4d2f9de6095b2cefce845f3ef93b271c01` | exact |
| `include/matrix_market.hpp` | `938b3dc879d828f20b1caa07a34998b184544006` | exact |
| `include/openmp_benchmark.hpp` | `db594e01d0a135f4c9697aeeb96895d7e81c9325` | exact |
| `include/openmp_ops.hpp` | `0e581e21bd05ffdddccb6ffdf48453dc10dc90ea` | exact |
| `include/openmp_solver.hpp` | `8a6b6c8dbcb2969c93d2251e84aceb921001f874` | exact |
| `include/poisson.hpp` | `3bbf8e55bf7ea2663861fc67d9dac8d05e5ff58c` | exact |
| `include/serial_benchmark.hpp` | `9b5863f91a7febc8af79217def0b0bbacbee8c12` | exact |
| `src/common/csr_ops.cpp` | `f8dea6e0f84421e3dfe1290383d1af3df0a3d558` | exact |
| `src/common/matrix_market.cpp` | `e99b29d1dcb4f5de9be5ae8f2def8b100725dab2` | exact |
| `src/common/poisson.cpp` | `4a0904fa3987d78297287c23c737e6e68761906c` | exact |
| `src/gpu/benchmark_gpu.cpp` | `124dd0c17c4887075bb194dfc19781c655230c06` | exact |
| `src/gpu/benchmark_gpu_csv.cpp` | `038fcf173399ac34060688d88800e7914c9d2c9a` | exact |
| `src/gpu/cg_gpu.cpp` | `6478e9a7e996dd7b28353191f7e3b541476f6d64` | exact |
| `src/gpu/run_gpu_benchmark.cpp` | `2f12b97f02ae552c6fb624245d9294353f791621` | exact |
| `src/openmp/benchmark_openmp.cpp` | `0a52cccc7883eb6837b940a63b5c7f135326a960` | exact |
| `src/openmp/benchmark_openmp_csv.cpp` | `bf7c75f1e59d97527fbf5790554f30acbce52912` | exact |
| `src/openmp/cg_openmp.cpp` | `4021724280477523bca50b3bdd6f22a6776f0caf` | exact |
| `src/openmp/openmp_ops.cpp` | `e0f78bb1e218a35eb82cb748f7504c60a91748dd` | exact |
| `src/openmp/run_openmp_benchmark.cpp` | `7ad1835895c18381124d92b9d081911b7f6de35f` | exact |
| `src/serial/benchmark_csv.cpp` | `5fd393f68a90ae499e659f0522b7d1f447716274` | exact |
| `src/serial/benchmark_serial.cpp` | `fe4701b1ccb3aa05746fad71c5b0d63a0400a369` | exact |
| `src/serial/cg_serial.cpp` | `69b05efdf291f66b2d641d2e27111abecf91c74f` | exact |
| `src/serial/run_serial_benchmark.cpp` | `71b60ce073d71bb4696e336b6edd1bdb88fec440` | exact |
| `tests/test_csr.cpp` | `f57f5ec33281b062feef9889a8cf282c561f9da4` | exact |
| `tests/test_csr_ops.cpp` | `3cc6ef4fcbcfe66981c9ce833569550c92585c24` | exact |
| `tests/test_gpu_cg.cpp` | `dc4d245912a8fb75f62a08435264b96eb80ff342` | exact |
| `tests/test_matrix_market.cpp` | `90dc9ef830fb7e13c15c65e010d864f0f7d2d8b4` | exact |
| `tests/test_openmp_cg.cpp` | `1061f415710e809ff7703cf5ea74454165b4cc65` | exact |
| `tests/test_openmp_ops.cpp` | `3594fc03c96572f5b4447d9df248ff452ed95a1e` | exact |
| `tests/test_serial_benchmark.cpp` | `4bcab61320390e94ae21b8de27c34dc10e3dd1e1` | exact |
| `tests/test_serial_cg.cpp` | `cc627d3c07a92eeaa16e29076cc27659f9543236` | exact |
