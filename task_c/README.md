# Task C: two independent original hybrid variants

- [handoff](handoff/README.md): unprofiled WSL experiments from 56ac360; per-SpMV allocation, recorded grids 64/256.
- [v1_profile](v1_profile/README.md): workspace reuse and original profiling from Member C revision 846239d; recorded V1 grid512 experiment, plus that revision's profiled MPI/OpenMP comparators.

Build each from its own directory with a distinct build/output directory. Neither uses sibling variants, task_a/task_b sources or the modified integration branch. No root combined build is provided. Historical CSVs are not needed to generate new measurements.

The final Task C/team report is unavailable. Both variants are preserved as identifiable historical implementations, not interchangeable performance baselines. Profiled V0 source remains unresolved; neither variant is labelled an exact V0 reconstruction. Do not combine handoff and V1 timing series or claim reproduced V0-to-V1 speedups without the missing source/build provenance.

See [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md) for every copied path, revision and Git blob. No numerical, operation, benchmark timing or profiling source contents were changed. CMake/README files are packaging-only; test assertions and bounded launch timeouts affect test execution only.

Independent verification on NUS Vanda on 13 November 2026 confirmed successful standalone builds and all three registered CTests passing for each variant (six Task C tests in total). Both variants also completed representative Hybrid MPI/OpenMP benchmark smoke tests. The V1 profiler generated per-rank CSV files and its aggregation script completed successfully. See [Vanda verification](../docs/VANDA_VERIFICATION_2026-11-13.md).

These checks establish functional execution of the tested configurations, not reproduction of historical performance claims. The exact profiled V0 source remains unavailable, and recorded Task C results do not establish execution on thermal2. Historical MPI failure-path limitations remain; use identical valid arguments and appropriate compute allocations.
