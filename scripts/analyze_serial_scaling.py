"""Validate saved Task A serial measurements and reproduce Figures 5–6.

Run: python scripts/analyze_serial_scaling.py
Requires matplotlib and Pillow. Never executes a benchmark or writes raw data.
"""
import argparse
import csv
import hashlib
import math
from pathlib import Path
import re
import shlex
import statistics as stats
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "20261104T022346Z_1517629.stdct-mgmt-02_EOF0kD"
SOURCE = ROOT / "results/serial_scaling" / CAMPAIGN
DOCS = ROOT / "docs/task_a"
FIGURES = DOCS / "figures/revised"
SCHEMA = ("solver,matrix_name,n,nnz,ranks,threads,run_index,iterations,converged,"
          "solve_seconds,true_relative_residual,relative_solution_error").split(",")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="Check data and existing summary without writing outputs")
    args = parser.parse_args()
    protected = list(SOURCE.glob("*")) + [p for p in FIGURES.glob("figure*.*")
                                          if re.match(r"figure[1-4]_", p.name)]
    before = {p: digest(p) for p in protected if p.is_file()}
    metadata = (SOURCE / "metadata.txt").read_text()
    meta = dict(line.split("=", 1) for line in metadata.splitlines()
                if re.match(r"^[a-z_]+=", line))
    for key, value in {"campaign_id": CAMPAIGN, "pbs_job_id": "1517629.stdct-mgmt-02",
                       "warmups": "4", "timed_repetitions": "10", "tolerance": "1e-8",
                       "max_iterations": "10000", "status": "complete", "exit_code": "0",
                       "bound_cpu": "37", "allowed_cpu_list": "37"}.items():
        assert meta[key] == value, (key, meta.get(key))
    assert "CMAKE_BUILD_TYPE:STRING=Release" in metadata
    assert "CMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG" in metadata
    assert "100% tests passed, 0 tests failed out of 2" in metadata
    assert "g++ (GCC) 13.2.0" in metadata
    _, order = read_csv(SOURCE / "execution_order.csv")
    expected = [("2d", 512), ("3d", 16), ("2d", 1024), ("3d", 40),
                ("2d", 64), ("3d", 64), ("2d", 256), ("2d", 512)]
    assert [(r["geometry"], int(r["grid_size"])) for r in order] == expected
    assert [int(r["sequence"]) for r in order] == list(range(1, 9))
    assert [r["role"] for r in order] == ["core"] * 7 + ["drift_check"]
    assert {p.name for p in SOURCE.glob("serial_*.csv")} == {r["output_csv"] for r in order}
    commands = [shlex.split(line) for line in (SOURCE / "commands.sh").read_text().splitlines()
                if line.startswith("taskset ")]
    assert len(commands) == 8
    campaign_script = (SOURCE / "campaign.pbs").read_text()
    recorded_order = re.search(r"^workloads=\((.+)\)$", campaign_script, re.M).group(1)
    assert shlex.split(recorded_order) == [f"{g} {n}" for g, n in expected]
    # Check the archived script against its campaign fingerprint.
    expected_script_hash = next(line.split()[0] for line in
                                (SOURCE / "source_sha256.txt").read_text().splitlines()
                                if line.endswith("/campaign.pbs"))
    assert digest(SOURCE / "campaign.pbs") == expected_script_hash, "Archived script hash mismatch"
    # Binaries/build files are intentionally excluded from the clean archive.
    # Historical executable hash, Release flags and test results remain in metadata.
    assert re.search(r"--- Executable SHA256 ---\s+([0-9a-f]{64})", metadata)

    summary = []
    for item, command in zip(order, commands):
        geometry, grid = item["geometry"], int(item["grid_size"])
        n = grid ** (2 if geometry == "2d" else 3)
        nnz = 5*n - 4*grid if geometry == "2d" else 7*n - 6*grid**2
        assert (int(item["n"]), int(item["nnz"])) == (n, nnz)
        filename = item["output_csv"]
        assert command[:3] == ["taskset", "-c", meta["bound_cpu"]]
        assert Path(command[3]).name == "run_serial_benchmark"
        assert command[4:7] == [geometry, str(grid), "10"] and command[8:] == ["4"]
        assert Path(command[7]).name == filename and Path(command[7]).parent.name == CAMPAIGN
        assert Path(command[3]).parent.parent == Path(command[7]).parent
        columns, rows = read_csv(SOURCE / filename)
        assert columns == SCHEMA and len(rows) == 10
        iterations, times, residuals, errors = [], [], [], []
        for i, row in enumerate(rows, 1):
            assert set(row) == set(SCHEMA) and all(v is not None for v in row.values())
            assert row["solver"] == "serial" and row["matrix_name"] == f"poisson_{geometry}_{grid}"
            for key, value in {"n": n, "nnz": nnz, "ranks": 1, "threads": 1,
                               "run_index": i, "converged": 1}.items():
                assert int(row[key]) == value, (filename, i, key)
            iteration = int(row["iterations"])
            assert 1 <= iteration <= 10000
            time, residual, error = (float(row[k]) for k in
                                    ("solve_seconds", "true_relative_residual", "relative_solution_error"))
            assert all(math.isfinite(v) for v in (time, residual, error))
            assert time > 0 and 0 <= residual <= 1e-8 and error >= 0
            iterations.append(iteration); times.append(time); residuals.append(residual); errors.append(error)
        assert len(set(iterations)) == 1
        summary.append(dict(campaign_id=CAMPAIGN, sequence=int(item["sequence"]),
                            geometry=geometry, grid_size=grid, n=n, nnz=nnz, role=item["role"],
                            runs=10, converged=10, iterations=iterations[0],
                            median_seconds=stats.median(times), min_seconds=min(times),
                            max_seconds=max(times), sample_std_seconds=stats.stdev(times),
                            max_true_relative_residual=max(residuals), max_relative_solution_error=max(errors),
                            source_csv=f"results/serial_scaling/{CAMPAIGN}/{filename}", source_sha256=digest(SOURCE / filename)))
    initial, final = summary[0], summary[-1]
    ratio = final["median_seconds"] / initial["median_seconds"]
    delta = final["median_seconds"] - initial["median_seconds"]
    core = [r for r in summary if r["role"] == "core"]
    output = DOCS / "consolidated/serial_scaling_summary.csv"
    if args.validate_only:
        _, saved = read_csv(output)
        assert len(saved) == len(summary)
        for actual, expected_row in zip(saved, summary):
            for field, value in expected_row.items():
                assert (float(actual[field]) == value if isinstance(value, (int, float))
                        else actual[field] == value), (field, actual[field], value)
        assert all(digest(p) == value for p, value in before.items())
        print(f"Validated 80/80 runs, provenance and summary; drift={(ratio-1)*100:.6f}%; no writes.")
        return
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    # Independently check the saved summary numeric values survive serialization.
    _, saved = read_csv(output)
    for actual, expected_row in zip(saved, summary):
        for field in ("median_seconds", "min_seconds", "max_seconds", "sample_std_seconds",
                      "max_true_relative_residual", "max_relative_solution_error"):
            assert float(actual[field]) == expected_row[field]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 14, "lines.linewidth": 1.8, "svg.fonttype": "none"})
    for number, column, xlabel in [(5, "n", "Matrix dimension N (log₁₀ scale)"),
                                   (6, "nnz", "Number of nonzeros NNZ (log₁₀ scale)")]:
        fig, ax = plt.subplots(figsize=(9, 5.8))
        fig.subplots_adjust(left=0.12, right=0.97, top=0.89, bottom=0.25)
        ax.set_title(f"Serial CG runtime versus {'matrix dimension' if column == 'n' else 'nonzero count'}", pad=14)
        for geometry, color, marker in [("2d", "#0072B2", "o"), ("3d", "#D55E00", "s")]:
            rows = sorted((r for r in core if r["geometry"] == geometry), key=lambda r: r[column])
            x, y = [r[column] for r in rows], [r["median_seconds"] for r in rows]
            line, = ax.plot(x, y, marker=marker, color=color, markersize=6,
                            label=f"{geometry.upper()} Poisson")
            assert list(line.get_xdata()) == x and list(line.get_ydata()) == y
        ax.set_xscale("log", base=10); ax.set_yscale("log", base=10)
        assert ax.xaxis.get_transform().base == ax.yaxis.get_transform().base == 10
        assert sum(len(line.get_xdata()) for line in ax.lines) == 7
        ax.set_xlabel(xlabel); ax.set_ylabel("Median solver execution time (s; log₁₀ scale)")
        ax.grid(which="major", alpha=0.23); ax.grid(which="minor", alpha=0.10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.text(0.12, 0.035, "Medians of 10 timed solves after 4 warmups; one pinned CPU.\n"
                 f"Initial 512² included; end check excluded (median drift: +{(ratio-1)*100:.2f}%).",
                 fontsize=9, va="bottom", linespacing=1.5)
        stem = f"figure{number}_serial_runtime_vs_{column}"
        for extension in ("png", "svg"):
            path = FIGURES / f"{stem}.{extension}"
            fig.savefig(path, dpi=300, facecolor="white")
            if extension == "png":
                with Image.open(path) as image:
                    assert image.size == (2700, 1740) and image.info["dpi"][0] >= 299.9
                    image.verify()
            else:
                assert ET.parse(path).getroot().tag.endswith("}svg")
        plt.close(fig)

    table = ["| Configuration | N | NNZ | Iterations | Median (s) | Min (s) | Max (s) | Sample SD (s) | Max true residual | Max solution error |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(core, key=lambda r: (r["geometry"], r["n"])) + [final]:
        label = f"{r['geometry'].upper()} {r['grid_size']}" + (" end check" if r["role"] == "drift_check" else "")
        table.append(f"| {label} | {r['n']:,} | {r['nnz']:,} | {r['iterations']} | " +
                     " | ".join(f"{r[k]:.9g}" for k in ("median_seconds", "min_seconds", "max_seconds", "sample_std_seconds")) +
                     f" | {r['max_true_relative_residual']:.6e} | {r['max_relative_solution_error']:.6e} |")
    d2 = sorted((r for r in core if r["geometry"] == "2d"), key=lambda r: r["n"])
    d3 = sorted((r for r in core if r["geometry"] == "3d"), key=lambda r: r["n"])
    report = f"""# Section 10.1 — Controlled serial CG scaling

## Validation and provenance

Source: `results/serial_scaling/{CAMPAIGN}/`. All eight CSVs have the exact original schema and ten measured rows (80 total). All 80 solves converged with finite true relative residuals ≤ 1e-8 and finite, nonnegative solution errors. N and NNZ match the 2D five-point and 3D seven-point generators. Run indices are 1–10 and iteration counts are constant within every configuration. No runs were filtered or corrected.

The CSVs, execution_order.csv, recorded taskset commands, and archived campaign script agree on all eight configurations, order, output names, four warmups and ten repetitions. Metadata records successful completion (exit 0), GCC 13.2.0 Release (-O3 -DNDEBUG), and two passing serial regression tests. The archived script SHA256 matches its recorded fingerprint. Executable hash and Release flags are retained in metadata; binaries and generated build files are intentionally excluded from the review archive. The downloaded original executable was verified before packaging. Vanda job {meta['pbs_job_id']} ran on {meta['hostname']} (Intel Xeon Platinum 8452Y), with one allocated CPU, pinned to CPU {meta['bound_cpu']}, 8 GB requested memory and one-hour walltime. Despite the GN hostname, this was a serial CPU run, not GPU execution.

The source was a standalone copy with no Git commit metadata. Source fingerprints are preserved, but original source contents are not bundled with this campaign, so they cannot independently establish byte-for-byte identity with the current checkout. The clean archive retains the original `results/serial_scaling/` layout and excludes generated build artifacts. The original downloaded campaign remains preserved separately. Warmup execution is evidenced by the script and command log, not raw warmup rows.

## Statistics

Every row summarizes ten measured solver times. Sample SD uses denominator n−1 = 9; it is run-to-run spread, not a confidence interval. The initial 512² configuration is used in both main curves. The final check is retained separately and is never pooled with it.

{chr(10).join(table)}

## Numerical observations and interpretation

- Across 2D grids 64–1024, N increases {d2[-1]['n']/d2[0]['n']:.0f}× and median time increases {d2[-1]['median_seconds']/d2[0]['median_seconds']:.2f}×. Iterations rise from {d2[0]['iterations']} to {d2[-1]['iterations']}.
- Across 3D grids 16–64, N increases {d3[-1]['n']/d3[0]['n']:.0f}× and median time increases {d3[-1]['median_seconds']/d3[0]['median_seconds']:.2f}×. Iterations rise from {d3[0]['iterations']} to {d3[-1]['iterations']}.
- At N=4,096, 2D uses 122 iterations versus 41 for 3D; the 2D/3D median runtime ratio is {d2[0]['median_seconds']/d3[0]['median_seconds']:.3f}. At N=262,144, the respective counts are 894 and 158, and the runtime ratio is {initial['median_seconds']/d3[-1]['median_seconds']:.3f}. Thus the higher NNZ of 3D does not alone determine whole-solve runtime.
- Each iteration performs sparse matrix–vector and vector operations, suggesting work proportional to iterations × (NNZ + N). These are whole-solver measurements: allocations, internal CSR validation and true-residual checks are included. Cache/memory effects and execution conditions may also contribute; the data do not isolate their causes.
- NNZ = 5g²−4g for 2D and 7g³−6g² for 3D. N and NNZ therefore co-vary within each family; the two figures are complementary descriptions, not independent experiments isolating sparsity. Connecting lines join measured points only and are not fitted complexity laws.

## Timing drift and limitations

Initial 512² median: **{initial['median_seconds']:.10f} s**. Final median: **{final['median_seconds']:.10f} s**. Change: **+{delta:.10f} s**, final/initial **{ratio:.6f}×**, or **+{(ratio-1)*100:.4f}%**. Both have 894 iterations and identical numerical error metrics. This is evidence of timing instability without a change in convergence behavior, not proof of a particular hardware cause. Four warmups and CPU pinning did not eliminate the observed difference.

The initial and final observed time ranges are {'disjoint' if initial['max_seconds'] < final['min_seconds'] else 'overlapping'}. Treat size trends as observations from this one allocation, with possible temporal confounding. A fixed shuffled order reduces simple monotonic-order confounding but does not replace independent allocations or repeated randomized blocks. No drift correction or confidence interval is inferred from the two endpoint checks, and their percentage difference is not a universal error bound for other sizes.

## Suggested report captions

**Figure 5. Serial CG runtime versus matrix dimension N.** Separate curves show the measured 2D and 3D Poisson workloads. Each marker is the median of ten timed solver calls following four warmups on one pinned Vanda CPU, using true-relative-residual tolerance 1e-8. Both axes use base-10 logarithmic scales. The initial 512² measurement is included; its separate end check is excluded and shows a {(ratio-1)*100:.2f}% median increase.

**Figure 6. Serial CG runtime versus nonzero count NNZ.** The same seven configurations and timing protocol as Figure 5 are shown with NNZ on the base-10 logarithmic x-axis and median seconds on the base-10 logarithmic y-axis. Curves remain separate by geometry. Runtime reflects iteration-count changes as well as per-iteration work; the 512² end check is excluded.

## Reproduction

Run `python scripts/analyze_serial_scaling.py` with matplotlib and Pillow installed. It reads the archived campaign, validates it, writes `docs/task_a/consolidated/serial_scaling_summary.csv`, regenerates Figures 5–6 in PNG (2700 × 1740, 300 DPI) and SVG, and writes this analysis. It does not execute the archived binary or alter original measurements or Figures 1–4.
"""
    (DOCS / "serial_scaling_analysis.md").write_text(report, encoding="utf-8")
    assert all(digest(p) == value for p, value in before.items()), "Protected input changed"
    print("Validated 80/80 rows and provenance; protected input/figure hashes unchanged.")
    print("\n".join(table))
    print(f"Drift: {delta:.10f} s; ratio={ratio:.9f}; percent={(ratio-1)*100:.6f}")


if __name__ == "__main__":
    main()
