# Section 10.1 — Controlled serial CG scaling

## Validation and provenance

Source: `results/serial_scaling/20261104T022346Z_1517629.stdct-mgmt-02_EOF0kD/`. All eight CSVs have the exact original schema and ten measured rows (80 total). All 80 solves converged with finite true relative residuals ≤ 1e-8 and finite, nonnegative solution errors. N and NNZ match the 2D five-point and 3D seven-point generators. Run indices are 1–10 and iteration counts are constant within every configuration. No runs were filtered or corrected.

The CSVs, execution_order.csv, recorded taskset commands, and archived campaign script agree on all eight configurations, order, output names, four warmups and ten repetitions. Metadata records successful completion (exit 0), GCC 13.2.0 Release (-O3 -DNDEBUG), and two passing serial regression tests. The archived script SHA256 matches its recorded fingerprint. Executable hash and Release flags are retained in metadata; binaries and generated build files are intentionally excluded from the review archive. The downloaded original executable was verified before packaging. Vanda job 1517629.stdct-mgmt-02 ran on GN-A40-076 (Intel Xeon Platinum 8452Y), with one allocated CPU, pinned to CPU 37, 8 GB requested memory and one-hour walltime. Despite the GN hostname, this was a serial CPU run, not GPU execution.

The source was a standalone copy with no Git commit metadata. Source fingerprints are preserved, but original source contents are not bundled with this campaign, so they cannot independently establish byte-for-byte identity with the current checkout. The clean archive retains the original `results/serial_scaling/` layout and excludes generated build artifacts. The original downloaded campaign remains preserved separately. Warmup execution is evidenced by the script and command log, not raw warmup rows.

## Statistics

Every row summarizes ten measured solver times. Sample SD uses denominator n−1 = 9; it is run-to-run spread, not a confidence interval. The initial 512² configuration is used in both main curves. The final check is retained separately and is never pooled with it.

| Configuration | N | NNZ | Iterations | Median (s) | Min (s) | Max (s) | Sample SD (s) | Max true residual | Max solution error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2D 64 | 4,096 | 20,224 | 122 | 0.003590079 | 0.003499342 | 0.003863098 | 0.000168263194 | 8.713678e-09 | 3.098812e-09 |
| 2D 256 | 65,536 | 326,656 | 454 | 0.26192864 | 0.257466524 | 0.263450618 | 0.00192896913 | 9.916657e-09 | 1.185893e-08 |
| 2D 512 | 262,144 | 1,308,672 | 894 | 2.09250699 | 2.07403648 | 2.112053 | 0.0121518693 | 9.526075e-09 | 2.211184e-08 |
| 2D 1024 | 1,048,576 | 5,238,784 | 1755 | 37.6985644 | 36.5162035 | 38.1779368 | 0.531480639 | 9.802119e-09 | 4.818934e-08 |
| 3D 16 | 4,096 | 27,136 | 41 | 0.001445311 | 0.001442985 | 0.00154242 | 4.38650077e-05 | 6.117135e-09 | 1.374876e-09 |
| 3D 40 | 64,000 | 438,400 | 101 | 0.063721961 | 0.063626089 | 0.066435698 | 0.000890479977 | 8.836375e-09 | 3.876745e-09 |
| 3D 64 | 262,144 | 1,810,432 | 158 | 0.546246889 | 0.52050922 | 0.592215297 | 0.0257028863 | 9.032060e-09 | 8.909344e-09 |
| 2D 512 end check | 262,144 | 1,308,672 | 894 | 2.39453799 | 2.32414126 | 2.41612869 | 0.0356701615 | 9.526075e-09 | 2.211184e-08 |

## Numerical observations and interpretation

- Across 2D grids 64–1024, N increases 256× and median time increases 10500.76×. Iterations rise from 122 to 1755.
- Across 3D grids 16–64, N increases 64× and median time increases 377.94×. Iterations rise from 41 to 158.
- At N=4,096, 2D uses 122 iterations versus 41 for 3D; the 2D/3D median runtime ratio is 2.484. At N=262,144, the respective counts are 894 and 158, and the runtime ratio is 3.831. Thus the higher NNZ of 3D does not alone determine whole-solve runtime.
- Each iteration performs sparse matrix–vector and vector operations, suggesting work proportional to iterations × (NNZ + N). These are whole-solver measurements: allocations, internal CSR validation and true-residual checks are included. Cache/memory effects and execution conditions may also contribute; the data do not isolate their causes.
- NNZ = 5g²−4g for 2D and 7g³−6g² for 3D. N and NNZ therefore co-vary within each family; the two figures are complementary descriptions, not independent experiments isolating sparsity. Connecting lines join measured points only and are not fitted complexity laws.

## Timing drift and limitations

Initial 512² median: **2.0925069915 s**. Final median: **2.3945379885 s**. Change: **+0.3020309970 s**, final/initial **1.144339×**, or **+14.4339%**. Both have 894 iterations and identical numerical error metrics. This is evidence of timing instability without a change in convergence behavior, not proof of a particular hardware cause. Four warmups and CPU pinning did not eliminate the observed difference.

The initial and final observed time ranges are disjoint. Treat size trends as observations from this one allocation, with possible temporal confounding. A fixed shuffled order reduces simple monotonic-order confounding but does not replace independent allocations or repeated randomized blocks. No drift correction or confidence interval is inferred from the two endpoint checks, and their percentage difference is not a universal error bound for other sizes.

## Suggested report captions

**Figure 5. Serial CG runtime versus matrix dimension N.** Separate curves show the measured 2D and 3D Poisson workloads. Each marker is the median of ten timed solver calls following four warmups on one pinned Vanda CPU, using true-relative-residual tolerance 1e-8. Both axes use base-10 logarithmic scales. The initial 512² measurement is included; its separate end check is excluded and shows a 14.43% median increase.

**Figure 6. Serial CG runtime versus nonzero count NNZ.** The same seven configurations and timing protocol as Figure 5 are shown with NNZ on the base-10 logarithmic x-axis and median seconds on the base-10 logarithmic y-axis. Curves remain separate by geometry. Runtime reflects iteration-count changes as well as per-iteration work; the 512² end check is excluded.

## Reproduction

Run `python scripts/analyze_serial_scaling.py` with matplotlib and Pillow installed. It reads the archived campaign, validates it, writes `docs/task_a/consolidated/serial_scaling_summary.csv`, regenerates Figures 5–6 in PNG (2700 × 1740, 300 DPI) and SVG, and writes this analysis. It does not execute the archived binary or alter original measurements or Figures 1–4.
