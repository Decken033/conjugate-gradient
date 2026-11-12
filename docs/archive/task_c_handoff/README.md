# Hybrid MPI + OpenMP — Team Handoff

This package snapshots the Hybrid implementation and the current local validation/performance status for team alignment.

## Package contents

- `include/hybrid/`: public Hybrid interfaces.
- `src/hybrid/`: Hybrid operations, CG solver, benchmark implementation, CSV writer, and CLI runner.
- `tests/`: Hybrid operations and CG correctness tests.
- `dependencies/`: key Common/MPI headers and MPI partition implementation referenced by the Hybrid module.
- `docs/`: English algorithm explanation, project integration, run guide, and current run report.
- `results/hybrid/`: CSV snapshots of the measurements supplied in the conversation.

## Important scope note

This is a focused handoff bundle, not a replacement for the complete project repository. Build it inside the full `conjugate-gradient-main` project because it depends on the project's Common, MPI, OpenMP, and CMake targets. Use the project's Hybrid-enabled `CMakeLists.txt` (the local file named `CMakeLists_hybrid.txt` was selected during the reported build).

## Current verified status

The user-provided local Ubuntu/WSL2 terminal log confirms:

- GCC 13.3.0, OpenMPI 4.1.6, CMake 3.28.3.
- CMake found OpenMP 4.5 and MPI 3.1.
- Hybrid targets configured and built successfully.
- `test_hybrid_ops` passed with 2 MPI ranks.
- `test_hybrid_cg` passed with 2 and 4 MPI ranks (2 OpenMP threads per rank in the test output).
- Benchmark measurements are available for 2D Poisson grids 64 and 256, using 2 ranks × 4 threads.

No 2D-512 baseline or MPI-only/OpenMP-only comparison is included yet. Those are planned experiments, not completed results.
