# Task C source provenance

Sources: handoff `56ac3602977385ec39b142f4a96e8cd833c8665a`; V1 `846239dbb7fe04daf56017adc6e8d7c18bda0f15`.

All files below are copied byte-for-byte. Relocations are packaging-only; no source contents, numerical kernels, profiler or benchmark timings are edited. Handoff flat includes resolve through additional local include directories, not duplicate headers. V1 comparator files are from Member C, not Task A/B.

CMake files and READMEs are new packaging. MPI/OpenMP are mandatory; original production target linkage and compiler defaults are retained. Only test translation units gain -UNDEBUG, and test launchers gain 90-second CTest timeouts. No runtime validation/synchronization is added. The handoff original hybrid CMake is missing, so its local target graph is reconstructed from dependencies. V1 targets are extracted from its original CMakeLists.txt.

The two variants are not substitutes for profiled V0. No exact V0 source-to-result mapping has been recovered. The final report is unavailable. CSVs/profile outputs are provenance evidence, not build or runtime inputs. Caught/uncaught historical MPI error paths are unchanged and have not been certified safe. Use valid matching arguments and bounded allocated-node tests.

Migration-stage static verification passed: all 66 copied files matched original Git blobs (24 handoff, 42 V1); project includes and CMake source paths resolved independently; the original Bash script passed bash -n and the Python summarizer passed syntax parsing; tracked and untracked whitespace checks passed. Native Windows and WSL probes lacked a usable CMake/C++/MPI compiler toolchain at that stage. No historical measurements were fabricated.

Subsequent independent verification on NUS Vanda on 13 November 2026 confirmed that the handoff variant compiled independently and passed 3/3 CTests, and the V1 profiling variant compiled independently and passed 3/3 CTests. Both completed representative Hybrid MPI/OpenMP benchmark smoke tests. V1 generated per-rank profiling CSVs, and profile aggregation completed successfully. See [Vanda verification](../docs/VANDA_VERIFICATION_2026-11-13.md).

The exact profiled V0 source remains unresolved. Historical handoff and V1 timings are not interchangeable, and full historical V0-to-V1 speedups were not reproduced. No recorded evidence establishes that Member C's Hybrid implementation was benchmarked on thermal2. Original MPI failure-path limitations remain; these functional checks do not certify every failure scenario.

| Destination under task_c | Revision | Original path | Git blob |
|---|---|---|---|
| `handoff/include/hybrid/hybrid_benchmark.hpp` | `56ac360` | `Hybrid_Team_Handoff/Hybrid_Team_Handoff/include/hybrid/hybrid_benchmark.hpp` | `72f380cf2c9e09a149822d750dbd27d94a1473c2` |
| `handoff/include/hybrid/hybrid_ops.hpp` | `56ac360` | `Hybrid_Team_Handoff/Hybrid_Team_Handoff/include/hybrid/hybrid_ops.hpp` | `e2bff822f39a6a2fd6c7cac81e372620392beae2` |
| `handoff/include/hybrid/hybrid_solver.hpp` | `56ac360` | `Hybrid_Team_Handoff/Hybrid_Team_Handoff/include/hybrid/hybrid_solver.hpp` | `e9693583606599add381f8188afa8983d049f3f5` |
| `handoff/src/hybrid/benchmark_hybrid.cpp` | `56ac360` | `Hybrid_Team_Handoff/Hybrid_Team_Handoff/src/hybrid/benchmark_hybrid.cpp` | `e42de3b1bf5dfffdbb76fcec75995c148298b0bd` |
| `handoff/src/hybrid/benchmark_hybrid_csv.cpp` | `56ac360` | `Hybrid_Team_Handoff/Hybrid_Team_Handoff/src/hybrid/benchmark_hybrid_csv.cpp` | `89293a7f2866592e1471dcdce01f4f40fe09826d` |
| `handoff/src/hybrid/hybrid_ops.cpp` | `56ac360` | `Hybrid_Team_Handoff/Hybrid_Team_Handoff/src/hybrid/hybrid_ops.cpp` | `680f31775f68cb3ac03b04f5f701eb9114a5becc` |
| `handoff/src/hybrid/hybrid_solver.cpp` | `56ac360` | `Hybrid_Team_Handoff/Hybrid_Team_Handoff/src/hybrid/hybrid_solver.cpp` | `203da0ba87b41c0dd28e150ec56a615c4ce0159e` |
| `handoff/src/hybrid/run_hybrid_benchmark.cpp` | `56ac360` | `Hybrid_Team_Handoff/Hybrid_Team_Handoff/src/hybrid/run_hybrid_benchmark.cpp` | `5e8424ba69d4c931df290d057334fbccff2af66d` |
| `handoff/include/common/csr_matrix.hpp` | `56ac360` | `Hybrid_Team_Handoff/Hybrid_Team_Handoff/dependencies/common/csr_matrix.hpp` | `da2d608e2a5de7396dddc129b7bb5fbb94e80c85` |
| `handoff/include/common/csr_ops.hpp` | `56ac360` | `Hybrid_Team_Handoff/Hybrid_Team_Handoff/dependencies/common/csr_ops.hpp` | `9bac833084a6bd78ebc3cdd35947b562a6c2854b` |
| `handoff/include/common/matrix_market.hpp` | `56ac360` | `Hybrid_Team_Handoff/Hybrid_Team_Handoff/dependencies/common/matrix_market.hpp` | `9eaba8baf3509100755bcc98699af2c0871cf0a0` |
| `handoff/include/common/poisson.hpp` | `56ac360` | `Hybrid_Team_Handoff/Hybrid_Team_Handoff/dependencies/common/poisson.hpp` | `f3670ed5e9edfd815bc4fd53e7eec712842c6a78` |
| `handoff/include/mpi/mpi_partition.hpp` | `56ac360` | `Hybrid_Team_Handoff/Hybrid_Team_Handoff/dependencies/mpi/mpi_partition.hpp` | `e9f16c0732a12c50cb91467f54099ed701156359` |
| `handoff/include/mpi/mpi_ops.hpp` | `56ac360` | `Hybrid_Team_Handoff/Hybrid_Team_Handoff/dependencies/mpi/mpi_ops.hpp` | `eacb968ba3b2f71ef214ed5a11d7eb7557edc70b` |
| `handoff/src/mpi/mpi_partition.cpp` | `56ac360` | `Hybrid_Team_Handoff/Hybrid_Team_Handoff/dependencies/mpi/mpi_partition.cpp` | `1d14c5a21645c35b028139a0c78608bf9522e27b` |
| `handoff/tests/test_hybrid_ops.cpp` | `56ac360` | `Hybrid_Team_Handoff/Hybrid_Team_Handoff/tests/test_hybrid_ops.cpp` | `444cf03d4980f017db70cb6c601b9d5b1b0cced9` |
| `handoff/tests/test_hybrid_cg.cpp` | `56ac360` | `Hybrid_Team_Handoff/Hybrid_Team_Handoff/tests/test_hybrid_cg.cpp` | `ca76f8f6371b79a65ca7f6009601b79d0e451154` |
| `handoff/src/common/csr_ops.cpp` | `56ac360` | `src/common/csr_ops.cpp` | `f8dea6e0f84421e3dfe1290383d1af3df0a3d558` |
| `handoff/src/common/poisson.cpp` | `56ac360` | `src/common/poisson.cpp` | `4a0904fa3987d78297287c23c737e6e68761906c` |
| `handoff/src/common/matrix_market.cpp` | `56ac360` | `src/common/matrix_market.cpp` | `e99b29d1dcb4f5de9be5ae8f2def8b100725dab2` |
| `handoff/src/mpi/mpi_ops.cpp` | `56ac360` | `src/mpi/mpi_ops.cpp` | `5910077e0d561f3d7da85b8b9b5b6092fd6406db` |
| `handoff/include/serial/cg_solver.hpp` | `56ac360` | `include/cg_solver.hpp` | `34e2f41cbcf8c98465cda518f7ec0c9d13f9e3bd` |
| `handoff/include/openmp/openmp_ops.hpp` | `56ac360` | `include/openmp_ops.hpp` | `0e581e21bd05ffdddccb6ffdf48453dc10dc90ea` |
| `v1_profile/src/common/csr_ops.cpp` | `846239d` | `src/common/csr_ops.cpp` | `346c1757f0bda98078d8399dbd45b82f29d3d890` |
| `v1_profile/src/common/matrix_market.cpp` | `846239d` | `src/common/matrix_market.cpp` | `aefde2544146a49b27e0bdde2564a02945dc28cc` |
| `v1_profile/src/common/poisson.cpp` | `846239d` | `src/common/poisson.cpp` | `19e70770fe46d7d3002c503257526814fb4ab87e` |
| `v1_profile/src/hybrid/benchmark_hybrid.cpp` | `846239d` | `src/hybrid/benchmark_hybrid.cpp` | `3965d88840425325e4803ca22b83faedb8fbe0d4` |
| `v1_profile/src/hybrid/benchmark_hybrid_csv.cpp` | `846239d` | `src/hybrid/benchmark_hybrid_csv.cpp` | `89293a7f2866592e1471dcdce01f4f40fe09826d` |
| `v1_profile/src/hybrid/hybrid_ops.cpp` | `846239d` | `src/hybrid/hybrid_ops.cpp` | `efe5617e9150c09ed4cb50df079aa0c95d8a6358` |
| `v1_profile/src/hybrid/hybrid_solver.cpp` | `846239d` | `src/hybrid/hybrid_solver.cpp` | `47c90ab1096c8b628cc8fba1394920bf425ec131` |
| `v1_profile/src/hybrid/hybrid_workspace.cpp` | `846239d` | `src/hybrid/hybrid_workspace.cpp` | `7623820e931ac21498ce50caf46b79cff2dd589d` |
| `v1_profile/src/hybrid/run_hybrid_benchmark.cpp` | `846239d` | `src/hybrid/run_hybrid_benchmark.cpp` | `5fe65f9b5fab2ad23c2fd998802ffaf9f37b6f62` |
| `v1_profile/src/mpi/benchmark_mpi.cpp` | `846239d` | `src/mpi/benchmark_mpi.cpp` | `82ddb48b7ade580677aff39c39d1cd0dda236fff` |
| `v1_profile/src/mpi/benchmark_mpi_csv.cpp` | `846239d` | `src/mpi/benchmark_mpi_csv.cpp` | `2d264a0e618188dbd8a3da5a4b9ea39823c5da23` |
| `v1_profile/src/mpi/mpi_ops.cpp` | `846239d` | `src/mpi/mpi_ops.cpp` | `87f5af448383b95eb7c5f47ca0c34b037b5f83c7` |
| `v1_profile/src/mpi/mpi_partition.cpp` | `846239d` | `src/mpi/mpi_partition.cpp` | `1d14c5a21645c35b028139a0c78608bf9522e27b` |
| `v1_profile/src/mpi/mpi_solver.cpp` | `846239d` | `src/mpi/mpi_solver.cpp` | `7239b80569b0f5878312496dcd29ccdc13b7797d` |
| `v1_profile/src/mpi/run_mpi_benchmark.cpp` | `846239d` | `src/mpi/run_mpi_benchmark.cpp` | `893efa549c6605028f50225f843b823ce0a2979b` |
| `v1_profile/src/openmp/benchmark_openmp.cpp` | `846239d` | `src/openmp/benchmark_openmp.cpp` | `7fd5dcf973199ca60aafd1afaffecbc831645a9a` |
| `v1_profile/src/openmp/benchmark_openmp_csv.cpp` | `846239d` | `src/openmp/benchmark_openmp_csv.cpp` | `fe7484b5ad4937751d8185fcda07f56ccabe1260` |
| `v1_profile/src/openmp/cg_openmp.cpp` | `846239d` | `src/openmp/cg_openmp.cpp` | `de49e69c7caa1277337c5d3403c5fd8257efe7d1` |
| `v1_profile/src/openmp/openmp_ops.cpp` | `846239d` | `src/openmp/openmp_ops.cpp` | `741546a529dc907f57abc120efce3c4f9a9728f6` |
| `v1_profile/src/openmp/run_openmp_benchmark.cpp` | `846239d` | `src/openmp/run_openmp_benchmark.cpp` | `4c056d09ef9c4a0bfb518779d8fa848bca66580c` |
| `v1_profile/include/common/csr_matrix.hpp` | `846239d` | `include/common/csr_matrix.hpp` | `da2d608e2a5de7396dddc129b7bb5fbb94e80c85` |
| `v1_profile/include/common/csr_ops.hpp` | `846239d` | `include/common/csr_ops.hpp` | `9bac833084a6bd78ebc3cdd35947b562a6c2854b` |
| `v1_profile/include/common/matrix_market.hpp` | `846239d` | `include/common/matrix_market.hpp` | `9eaba8baf3509100755bcc98699af2c0871cf0a0` |
| `v1_profile/include/common/ops_profiler.hpp` | `846239d` | `include/common/ops_profiler.hpp` | `c745208253f0d356a22839432adb3b796bf54f57` |
| `v1_profile/include/common/poisson.hpp` | `846239d` | `include/common/poisson.hpp` | `f3670ed5e9edfd815bc4fd53e7eec712842c6a78` |
| `v1_profile/include/hybrid/hybrid_benchmark.hpp` | `846239d` | `include/hybrid/hybrid_benchmark.hpp` | `72f380cf2c9e09a149822d750dbd27d94a1473c2` |
| `v1_profile/include/hybrid/hybrid_ops.hpp` | `846239d` | `include/hybrid/hybrid_ops.hpp` | `82e99defef7b2f9037d22b1609af03b713e2c43c` |
| `v1_profile/include/hybrid/hybrid_solver.hpp` | `846239d` | `include/hybrid/hybrid_solver.hpp` | `e9693583606599add381f8188afa8983d049f3f5` |
| `v1_profile/include/hybrid/hybrid_workspace.hpp` | `846239d` | `include/hybrid/hybrid_workspace.hpp` | `ba24c3718b8448023acc3f1e5fa6f93980898849` |
| `v1_profile/include/mpi/mpi_benchmark.hpp` | `846239d` | `include/mpi/mpi_benchmark.hpp` | `e73f1585f29e8bf953d56beb684d3d9c04355d24` |
| `v1_profile/include/mpi/mpi_ops.hpp` | `846239d` | `include/mpi/mpi_ops.hpp` | `eacb968ba3b2f71ef214ed5a11d7eb7557edc70b` |
| `v1_profile/include/mpi/mpi_partition.hpp` | `846239d` | `include/mpi/mpi_partition.hpp` | `e9f16c0732a12c50cb91467f54099ed701156359` |
| `v1_profile/include/mpi/mpi_solver.hpp` | `846239d` | `include/mpi/mpi_solver.hpp` | `6ced0eb8a83a7fa165a57f4dbc6f9d198fd636cb` |
| `v1_profile/include/openmp/openmp_benchmark.hpp` | `846239d` | `include/openmp/openmp_benchmark.hpp` | `1ea8778ac10a5bd0c262d6a7fedf664831779c18` |
| `v1_profile/include/openmp/openmp_ops.hpp` | `846239d` | `include/openmp/openmp_ops.hpp` | `6bd5fb2dc066feb8ead2303b014566fb193886ed` |
| `v1_profile/include/openmp/openmp_solver.hpp` | `846239d` | `include/openmp/openmp_solver.hpp` | `6095fd7e83ac6d9d0c298ea4d9b89cab8cd0cbd1` |
| `v1_profile/include/serial/cg_solver.hpp` | `846239d` | `include/serial/cg_solver.hpp` | `83173e0cd59f5fe310a4d4aad49e5c101c8f1c78` |
| `v1_profile/tests/test_hybrid_ops.cpp` | `846239d` | `tests/test_hybrid_ops.cpp` | `160010ee9e4024cf1b34e374eb8d5d927c69c3ab` |
| `v1_profile/tests/test_hybrid_cg.cpp` | `846239d` | `tests/test_hybrid_cg.cpp` | `ca76f8f6371b79a65ca7f6009601b79d0e451154` |
| `v1_profile/scripts/run_hybrid_v1_profile_2x4.sh` | `846239d` | `scripts/run_hybrid_v1_profile_2x4.sh` | `f9425858c0b99d20d06e9aa30139e9077b29797d` |
| `v1_profile/scripts/summarize_profiles.py` | `846239d` | `scripts/summarize_profiles.py` | `725bb33dc205fcbe92a8853f15d2827ce2dc465b` |
| `handoff/scripts/download_thermal2.ps1` | `56ac360` | `scripts/download_thermal2.ps1` | `c3ac1b63330f811305b2eaf4f6c2ea09d8d28ac9` |
| `v1_profile/scripts/download_thermal2.ps1` | `846239d` | `scripts/download_thermal2.ps1` | `c3ac1b63330f811305b2eaf4f6c2ea09d8d28ac9` |
