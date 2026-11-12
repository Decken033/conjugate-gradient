# Task C: two independent original hybrid variants

- [handoff](handoff/README.md): unprofiled WSL experiments from 56ac360; per-SpMV allocation, recorded grids 64/256.
- [v1_profile](v1_profile/README.md): workspace reuse and original profiling from Member C revision 846239d; recorded V1 grid512 experiment, plus that revision's profiled MPI/OpenMP comparators.

Build each from its own directory with a distinct build/output directory. Neither uses sibling variants, task_a/task_b sources or the modified integration branch. No root combined build is provided. Historical CSVs are not needed to generate new measurements.

The final Task C/team report is unavailable. Both variants are preserved as identifiable historical implementations, not interchangeable performance baselines. Profiled V0 source remains unresolved; neither variant is labelled an exact V0 reconstruction. Do not combine handoff and V1 timing series or claim reproduced V0-to-V1 speedups without the missing source/build provenance.

See [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md) for every copied path, revision and Git blob. No numerical, operation, benchmark timing or profiling source contents were changed. CMake/README files are packaging-only; test assertions and bounded launch timeouts affect test execution only.

BUILD AND RUNTIME NOT VERIFIED locally. Historical MPI failure-path limitations are retained; use valid identical arguments and bounded tests in appropriate compute allocations. Report source equivalence separately from future runtime verification.
