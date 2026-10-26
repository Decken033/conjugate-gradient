"""Audit selected MPI benchmark CSVs and render reproducible SVG figures."""

import csv
import math
import statistics
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "mpi"
FIGURES = ROOT / "docs" / "task_b"
JOB = "1495684.stdct-mgmt-02"
RANKS = (1, 2, 4, 8, 16)
FIELDS = (
    "solver", "matrix_name", "n", "nnz", "ranks", "threads", "run_index",
    "iterations", "converged", "solve_seconds", "true_relative_residual",
    "relative_solution_error",
)


def read_series(path, matrix, ranks, n, nnz):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected CSV columns in {path}")
        records = list(reader)
    if len(records) != 10 or {int(r["run_index"]) for r in records} != set(range(10)):
        raise ValueError(f"Expected exactly ten indexed measurements in {path}")
    for record in records:
        if (record["solver"], record["matrix_name"], int(record["n"]),
                int(record["nnz"]), int(record["ranks"]), int(record["threads"])) != (
                "mpi", matrix, n, nnz, ranks, 1):
            raise ValueError(f"Inconsistent metadata in {path}")
        for field in ("solve_seconds", "true_relative_residual", "relative_solution_error"):
            if not math.isfinite(float(record[field])):
                raise ValueError(f"Nonfinite {field} in {path}")
        if float(record["solve_seconds"]) <= 0:
            raise ValueError(f"Nonpositive solve time in {path}")
    times = [float(r["solve_seconds"]) for r in records]
    return {
        "ranks": ranks, "count": len(records), "mean": statistics.mean(times),
        "sd": statistics.stdev(times), "median": statistics.median(times),
        "min": min(times), "max": max(times),
        "converged": sum(int(r["converged"]) for r in records),
        "iterations": sorted({int(r["iterations"]) for r in records}),
        "residual_max": max(float(r["true_relative_residual"]) for r in records),
        "error_max": max(float(r["relative_solution_error"]) for r in records),
    }


def load_group(pattern, matrix, n, nnz):
    group = [read_series(RESULTS / pattern.format(p=p), matrix, p, n, nnz)
             for p in RANKS]
    baseline = group[0]["mean"]
    for row in group:
        row["speedup"] = baseline / row["mean"]
        row["efficiency"] = row["speedup"] / row["ranks"] * 100
    return group


def line(x1, y1, x2, y2, color, width=1, dash=None):
    style = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
            f'y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{style}/>')


def label(x, y, value, size=14, color="#253341", anchor="middle", weight="normal"):
    return (f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
            f'font-family="Arial,Helvetica,sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}">{escape(str(value))}</text>')


def chart(title, panels, width=1500):
    height = 560
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
              f'viewBox="0 0 {width} {height}" role="img">',
              f'<title>{escape(title)}</title>',
              f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
              label(width / 2, 43, title, 24, weight="bold")]
    panel_width = (width - 100 - 36 * (len(panels) - 1)) / len(panels)
    for index, panel in enumerate(panels):
        left = 72 + index * (panel_width + 36)
        right = left + panel_width - 22
        top, bottom = 119, 438
        ymax = panel["ymax"]
        pieces.append(label((left + right) / 2, 83, panel["title"], 17, weight="bold"))
        for tick in panel["ticks"]:
            y = bottom - tick / ymax * (bottom - top)
            pieces.append(line(left, y, right, y, "#dfe5e8"))
            pieces.append(label(left - 10, y + 5, panel["format"](tick), 12, anchor="end"))
        pieces.extend((line(left, top, left, bottom, "#667581"),
                       line(left, bottom, right, bottom, "#667581")))
        xcoord = lambda i: left + 13 + i * (right - left - 26) / (len(RANKS) - 1)
        for i, rank in enumerate(RANKS):
            pieces.append(label(xcoord(i), bottom + 25, rank, 13))
        pieces.append(label((left + right) / 2, bottom + 55, "MPI ranks", 14))
        for series in panel["series"]:
            values = series["values"]
            points = [(xcoord(i), bottom - value / ymax * (bottom - top))
                      for i, value in enumerate(values)]
            path = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            dash = ' stroke-dasharray="7 5"' if series.get("dash") else ""
            pieces.append(f'<polyline points="{path}" fill="none" stroke="{series["color"]}" '
                          f'stroke-width="2.5"{dash}/>')
            if not series.get("dash"):
                for i, (x, y) in enumerate(points):
                    pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" '
                                  f'fill="{series["color"]}" stroke="#ffffff" stroke-width="1.2"/>')
                    if "sd" in series:
                        delta = series["sd"][i] / ymax * (bottom - top)
                        pieces.extend((line(x, y - delta, x, y + delta, series["color"], 1.5),
                                       line(x - 5, y - delta, x + 5, y - delta, series["color"], 1.5),
                                       line(x - 5, y + delta, x + 5, y + delta, series["color"], 1.5)))
        legend_x = left
        for series in panel["series"]:
            pieces.append(line(legend_x, 520, legend_x + 23, 520, series["color"], 3,
                               "7 5" if series.get("dash") else None))
            pieces.append(label(legend_x + 29, 525, series["name"], 12, anchor="start"))
            legend_x += series.get("legend_width", 135)
    pieces.append("</svg>")
    return "\n".join(pieces) + "\n"


def main():
    thermal = load_group(f"thermal2/thermal2_p{{p}}_{JOB}.csv", "thermal2", 1228045, 8580313)
    for row in thermal:
        if (row["converged"] != row["count"] or row["residual_max"] > 1e-8 or
                any(iteration > 10000 for iteration in row["iterations"])):
            raise ValueError(f"thermal2 numerical check failed at {row['ranks']} ranks")
    poisson_512 = load_group("mpi_2d512_p{p}_1482451.stdct-mgmt-02.csv",
                             "poisson_2d_512", 262144, 1308672)
    poisson_1024 = load_group("mpi_2d1024_p{p}_1482491.stdct-mgmt-02.csv",
                              "poisson_2d_1024", 1048576, 5238784)
    teal, orange, blue, gray = "#087e8b", "#c45624", "#4059ad", "#727b83"
    figure = chart("MPI thermal2 strong scaling: measured solver time", [
        {"title": "Mean solve time", "ymax": 100, "ticks": (0, 25, 50, 75, 100),
         "format": lambda x: f"{x} s", "series": [
             {"name": "Measured +/- SD", "color": teal, "values": [r["mean"] for r in thermal],
              "sd": [r["sd"] for r in thermal], "legend_width": 175},
             {"name": "Ideal T(1)/p", "color": gray, "dash": True,
              "values": [thermal[0]["mean"] / p for p in RANKS]}]},
        {"title": "Speedup vs 1 MPI rank", "ymax": 16, "ticks": (0, 4, 8, 12, 16),
         "format": str, "series": [
             {"name": "Measured", "color": teal, "values": [r["speedup"] for r in thermal]},
             {"name": "Ideal linear", "color": gray, "dash": True,
              "values": list(RANKS)}]},
        {"title": "Parallel efficiency", "ymax": 100, "ticks": (0, 25, 50, 75, 100),
         "format": lambda x: f"{x}%", "series": [
             {"name": "Measured", "color": teal,
              "values": [r["efficiency"] for r in thermal]}]},
    ])
    (FIGURES / "mpi_thermal2_scaling.svg").write_text(figure, encoding="utf-8")
    comparison = chart("MPI strong scaling by matrix: within-job speedup", [
        {"title": "Speedup vs each matrix's 1 MPI rank", "ymax": 4,
         "ticks": (0, 1, 2, 3, 4), "format": str, "series": [
             {"name": "thermal2", "color": teal, "values": [r["speedup"] for r in thermal],
              "legend_width": 115},
             {"name": "Poisson 512 x 512", "color": orange,
              "values": [r["speedup"] for r in poisson_512], "legend_width": 185},
             {"name": "Poisson 1024 x 1024", "color": blue,
              "values": [r["speedup"] for r in poisson_1024], "legend_width": 210}]},
    ], width=950)
    (FIGURES / "mpi_thermal2_comparison.svg").write_text(comparison, encoding="utf-8")
    for row in thermal:
        print(f"{row['ranks']:>2} {row['mean']:.6f} {row['sd']:.6f} "
              f"{row['median']:.6f} {row['min']:.6f} {row['max']:.6f} "
              f"{row['speedup']:.4f} {row['efficiency']:.2f}% "
              f"{row['converged']}/{row['count']} {row['iterations']} "
              f"{row['residual_max']:.6e} {row['error_max']:.6e}")


if __name__ == "__main__":
    main()
