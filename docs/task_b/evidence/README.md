# Task B: recovered Vanda experiment evidence

## Scope and provenance

Recovered from the user-provided Vanda download at `conjugate-gradient-backup/conjugate-gradient` on 12 November 2026. The import did not rerun MPI or modify existing differing files. Existing selected CSVs and hostfiles matched the backup byte-for-byte and were preserved. Seven logs and seven scripts were imported byte-for-byte.

`results/mpi/evidence/manifest.json` lists 76 files: 60 selected raw CSVs, seven stdout logs, seven recovered PBS scripts and two extension hostfiles. Every entry records the backup-relative source path, byte length and SHA-256. These hashes were computed during local import; they are NOT historical submission-time hashes and do not prove an immutable source/script-to-job association.

Paths below are relative to the repository root.

| Campaign | PBS job | Selected CSVs | Timed records | Raw CSV location |
|---|---|---:|---:|---|
| 2D 512 x 512 | 1482451.stdct-mgmt-02 | 8 | 80 | `results/mpi/mpi_2d512_p*_1482451.stdct-mgmt-02.csv` |
| 2D 1024 x 1024 | 1482491.stdct-mgmt-02 | 8 | 80 | `results/mpi/mpi_2d1024_p*_1482491.stdct-mgmt-02.csv` |
| thermal2 | 1495684.stdct-mgmt-02 | 5 | 50 | `results/mpi/thermal2/thermal2_p*_1495684.stdct-mgmt-02.csv` |
| 3D 40/64/102 cubed | 1502451.stdct-mgmt-02 | 24 | 240 | `results/mpi/3d/mpi_3d*_p*_1502451.stdct-mgmt-02.csv` |
| Strong extension | 1509066.stdct-mgmt-02 | 10 | 100 | `results/mpi/extension/strong_1509066.stdct-mgmt-02/` |
| Weak extension | 1509067.stdct-mgmt-02 | 5 | 50 | `results/mpi/extension/weak_1509067.stdct-mgmt-02/` |
| **Total** | | **60** | **600** | |

Earlier job 1482438, validation/smoke CSVs and partial attempts are excluded from these totals and remain ignored. The import does not require downloaded matrices or build binaries to be committed.

## Historical scripts versus reproduction recipes

`docs/task_b/evidence/scripts/` contains recovered snapshots, not rewritten portable scripts:

- `benchmark_mpi_scaling_3d.pbs`: recovered historical 3D script; its labels, configuration order and output paths agree with job 1502451's log. Requests two nodes, 36 MPI slots/node, 64 GB/node, Release, slot mapping and core binding. Absolute scratch/account settings are historical. The script does not record an actual rank-placement map or scheduler exit status.
- `benchmark_mpi_thermal2.pbs`: recovered script uses `data/thermal2/thermal2.mtx`, matching the log. The active `scripts/vanda/benchmark_mpi_thermal2.pbs` instead uses `data/suitesparse/thermal2/thermal2.mtx`; preserve that distinction. The historical missing-file hint still names the other staging path; it is intentionally not edited in the snapshot.
- The two 2D scripts, strong/weak 256-rank scripts and correctness script are also retained for context. Historical MPI module/build-cache inconsistencies remain documented in the report.

`scripts/vanda/benchmark_mpi_3d_reproduce.pbs` is a separately authored reproduction recipe, NOT the recovered script. Adapt account/scratch/input settings before submitting any recipe and use fresh output filenames; the CSV writer truncates existing paths. No job was submitted during this import.

## Logs and limitations

Canonical tracked logs are under `results/mpi/evidence/logs/`:

- Correctness: `dsa5208_mpi_tests.o1482726` records 10/10 CTest passes (seven non-MPI and three MPI targets), then partition/SpMV/CG manual invocations at 1, 2 and 4 ranks. It does not establish high-rank, empty-partition or coordinated-failure regression coverage.
- Six campaign logs use the names matching the CSV job IDs. All 60 selected configurations report `Converged: 10/10`.
- thermal2's log includes all five summaries but lacks a full job/time/allocation header and final completion marker. Warning prefixes are not a rank-placement map.
- Optional-component library warnings are marked ignored; these are not warning-free logs. Historical scheduler accounting, submitted script hashes, per-job source/binary hashes and saved solution vectors are unavailable.

The recovered code matched the audited pre-fix MPI implementation. Collective-safety source fixes and bounded tests have subsequently been prepared. They have not been compiled or executed in this Windows session, and Vanda acceptance remains pending. Archiving evidence does NOT certify final correctness. Saved CSV residuals/convergence flags are recorded metrics, not independently reconstructed solutions.

## Derived results and checks

Retained original figures are in `docs/task_b/`; extension summaries and SVG/PNG figures are in `results/mpi/extension/analysis/`. They are derived artifacts, not included in the 76-file raw-evidence manifest.

Run from the repository root:

```bash
python -B scripts/verify_mpi_evidence.py
python -B scripts/validate_task_b_report.py
```

The first command validates the manifest, exact inventory, hashes and selected CSV numeric fields. The report validator checks tables, figures and logs; its optional local Word-document check is skipped when that untracked document is absent. Neither command executes the MPI solver. `scripts/archive_mpi_evidence.py` records the import recipe and intentionally refuses to replace an existing manifest or differing destination.

Git exceptions expose only curated results, extension artifacts and canonical logs. `.gitattributes` disables newline conversion for imported evidence, preserving manifest hashes across checkout. Files are not staged, committed or pushed automatically.
