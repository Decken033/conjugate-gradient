"""Validate new MPI scaling CSVs and generate summaries plus log-rank SVG plots.

Run on the retained job result directories:
    python scripts/analyze_mpi_scaling_extension.py \
        results/mpi/extension/strong_1509066.stdct-mgmt-02 \
        results/mpi/extension/weak_1509067.stdct-mgmt-02 \
        --output-dir results/mpi/extension/analysis
"""

import argparse
import csv
import math
import statistics
from pathlib import Path
from xml.sax.saxutils import escape


FIELDS = (
    "solver", "matrix_name", "n", "nnz", "ranks", "threads", "run_index",
    "iterations", "converged", "solve_seconds", "true_relative_residual",
    "relative_solution_error",
)
STRONG_RANKS = (1, 2, 4, 8, 16, 32, 36, 72, 144, 256)
WEAK_CASES = ((1, 128), (4, 256), (16, 512), (64, 1024), (256, 2048))


def read_case(directory, prefix, ranks, grid):
    matches = list(directory.glob(f"{prefix}_2d{grid}_p{ranks}_*.csv"))
    if len(matches) != 1:
        raise ValueError(f"Expected one {prefix} CSV for grid={grid}, ranks={ranks} in {directory}; found {len(matches)}")
    path = matches[0]
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected CSV schema: {path}")
        records = list(reader)
    if len(records) != 10 or {int(r["run_index"]) for r in records} != set(range(10)):
        raise ValueError(f"Expected ten uniquely indexed timed runs: {path}")

    times = []
    iterations = set()
    residuals = []
    errors = []
    for record in records:
        if set(record) != set(FIELDS) or any(record[key] in (None, "") for key in FIELDS):
            raise ValueError(f"Incomplete or extra CSV fields: {path}")
        metadata = (
            record["solver"], record["matrix_name"], int(record["n"]),
            int(record["ranks"]), int(record["threads"]),
        )
        if metadata != ("mpi", f"poisson_2d_{grid}", grid * grid, ranks, 1):
            raise ValueError(f"Inconsistent matrix or rank metadata: {path}")
        nnz = 5 * grid * grid - 4 * grid
        if int(record["nnz"]) != nnz or int(record["converged"]) != 1:
            raise ValueError(f"Incorrect NNZ or nonconverged run: {path}")
        elapsed = float(record["solve_seconds"])
        residual = float(record["true_relative_residual"])
        error = float(record["relative_solution_error"])
        if not (math.isfinite(elapsed) and elapsed > 0 and
                math.isfinite(residual) and 0 <= residual <= 1e-8 and
                math.isfinite(error) and 0 <= error < 1e-4):
            raise ValueError(f"Invalid timing or numerical result: {path}")
        times.append(elapsed)
        residuals.append(residual)
        errors.append(error)
        iterations.add(int(record["iterations"]))
    if len(iterations) != 1 or not 0 < next(iter(iterations)) <= 10000:
        raise ValueError(f"Inconsistent iteration count: {path}")

    return {
        "ranks": ranks, "grid": grid, "n": grid * grid, "nnz": nnz,
        "runs": len(times), "mean_seconds": statistics.mean(times),
        "sd_seconds": statistics.stdev(times), "median_seconds": statistics.median(times),
        "min_seconds": min(times), "max_seconds": max(times),
        "iterations": next(iter(iterations)),
        "max_residual": max(residuals), "max_solution_error": max(errors),
        "source": path.name,
    }


def load_groups(strong_dir, weak_dir):
    expected_strong = {f"strong_2d1024_p{p}_1509066.stdct-mgmt-02.csv" for p in STRONG_RANKS}
    expected_weak = {f"weak_2d{grid}_p{p}_1509067.stdct-mgmt-02.csv" for p, grid in WEAK_CASES}
    if {p.name for p in strong_dir.glob("*.csv")} != expected_strong:
        raise ValueError("Strong scaling CSV inventory differs from job 1509066")
    if {p.name for p in weak_dir.glob("*.csv")} != expected_weak:
        raise ValueError("Weak scaling CSV inventory differs from job 1509067")
    strong = [read_case(strong_dir, "strong", p, 1024) for p in STRONG_RANKS]
    weak = [read_case(weak_dir, "weak", p, grid) for p, grid in WEAK_CASES]
    for group in (strong, weak):
        job_ids = {row["source"].rsplit("_", 1)[-1] for row in group}
        if len(job_ids) != 1:
            raise ValueError("A scaling series must come from one PBS job")

    strong_baseline = strong[0]["mean_seconds"]
    for row in strong:
        row["speedup"] = strong_baseline / row["mean_seconds"]
        row["efficiency"] = row["speedup"] / row["ranks"]
        row["seconds_per_iteration"] = row["mean_seconds"] / row["iterations"]

    weak_baseline = weak[0]["mean_seconds"]
    baseline_per_iteration = weak_baseline / weak[0]["iterations"]
    for row in weak:
        row["unknowns_per_rank"] = row["n"] // row["ranks"]
        row["seconds_per_iteration"] = row["mean_seconds"] / row["iterations"]
        row["weak_efficiency"] = weak_baseline / row["mean_seconds"]
        row["iteration_normalized_efficiency"] = (
            baseline_per_iteration / row["seconds_per_iteration"]
        )
    return strong, weak


def write_summary(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def svg_chart(path, title, rows, panels, definition):
    width, height = 980, 900
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f'<title>{escape(title)}</title>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="35" text-anchor="middle" font-family="Arial" '
        f'font-size="22" fill="#26343e">{escape(title)}</text>',
    ]
    x_ticks = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    for panel_index, (label, key) in enumerate(panels):
        left = 65 + (panel_index % 2) * 455
        right = left + 375
        top = 95 + (panel_index // 2) * 390
        bottom = top + 280
        values = [float(row[key]) for row in rows]
        maximum = max(values) * 1.12
        if maximum == 0:
            maximum = 1.0
        x = lambda p: left + math.log2(p) / 8 * (right - left)
        y = lambda value: bottom - value / maximum * (bottom - top)
        parts.append(
            f'<text x="{(left + right) / 2:.1f}" y="{top - 27}" text-anchor="middle" '
            f'font-family="Arial" font-size="16" fill="#26343e">{escape(label)}</text>'
        )
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            yy = bottom - fraction * (bottom - top)
            parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" stroke="#e0e5e8"/>')
            parts.append(
                f'<text x="{left - 7}" y="{yy + 4:.1f}" text-anchor="end" '
                f'font-family="Arial" font-size="11" fill="#52616b">{maximum * fraction:.3g}</text>'
            )
        for tick in x_ticks:
            xx = x(tick)
            parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{bottom}" stroke="#edf0f2"/>')
            parts.append(
                f'<text x="{xx:.1f}" y="{bottom + 20}" text-anchor="middle" '
                f'font-family="Arial" font-size="11" fill="#52616b">{tick}</text>'
            )
        points = " ".join(f"{x(row['ranks']):.1f},{y(row[key]):.1f}" for row in rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="#087e8b" stroke-width="2.5"/>')
        for row in rows:
            parts.append(
                f'<circle cx="{x(row["ranks"]):.1f}" cy="{y(row[key]):.1f}" r="4" fill="#c45624"/>'
            )
        parts.append(
            f'<text x="{(left + right) / 2:.1f}" y="{bottom + 48}" text-anchor="middle" '
            f'font-family="Arial" font-size="12" fill="#52616b">MPI ranks (log2 scale)</text>'
        )
    parts.append(
        f'<text x="{width / 2:.0f}" y="870" text-anchor="middle" '
        f'font-family="Arial" font-size="13" fill="#26343e">{escape(definition)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strong_dir", type=Path, help="One completed strong-scaling PBS job directory")
    parser.add_argument("weak_dir", type=Path, help="One completed weak-scaling PBS job directory")
    parser.add_argument("--output-dir", type=Path, default=Path("results/mpi/extension/analysis"))
    args = parser.parse_args()
    strong, weak = load_groups(args.strong_dir, args.weak_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(args.output_dir / "strong_summary.csv", strong)
    write_summary(args.output_dir / "weak_summary.csv", weak)
    svg_chart(args.output_dir / "strong_scaling.svg", "MPI 2D Poisson strong scaling", strong, (
        ("Mean solver time (s)", "mean_seconds"),
        ("Speedup vs 1 rank", "speedup"),
        ("Parallel efficiency (fraction)", "efficiency"),
        ("Time per iteration (s)", "seconds_per_iteration"),
    ), "Fixed 1024 x 1024 grid; baseline: mean of 10 runs at 1 rank in job 1509066")
    svg_chart(args.output_dir / "weak_scaling.svg", "MPI 2D Poisson weak scaling", weak, (
        ("Mean solver time (s)", "mean_seconds"),
        ("Time per iteration (s)", "seconds_per_iteration"),
        ("Weak efficiency (fraction)", "weak_efficiency"),
        ("CG iterations (count)", "iterations"),
    ), "16,384 unknowns/rank; baseline: 1-rank mean in job 1509067; efficiency = baseline / P-rank mean")
    print(f"Validated {len(strong) + len(weak)} configurations and 150 timed runs.")
    print(f"Summaries and plots: {args.output_dir}")


if __name__ == "__main__":
    main()
