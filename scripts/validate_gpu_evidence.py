"""Read-only validation of the archived Task A GPU campaign (standard library).

Run from any directory: python scripts/validate_gpu_evidence.py
No solver, plotting, or file-writing operations are performed.
"""

import csv
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/gpu_evidence"
SCHEMA = ("solver,matrix_name,n,nnz,ranks,threads,run_index,iterations,"
          "converged,solve_seconds,true_relative_residual,relative_solution_error").split(",")
ORDER = ["gpu_2d_64", "gpu_2d_256", "gpu_2d_512", "gpu_2d_1024",
         "gpu_3d_16", "gpu_3d_40", "gpu_3d_64", "gpu_thermal2",
         "gpu_2d_256_repeat"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def close(actual, expected, label):
    require(math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-15),
            f"{label}: {actual} != {expected}")


def validate():
    _, summaries = read(DATA / "gpu_summary.csv")
    require(len(summaries) == 9, "Expected nine summary rows")
    summary = {r["configuration"]: r for r in summaries}
    require(set(summary) == set(ORDER), "Unexpected/duplicate configurations")
    raw_names = {p.stem for p in DATA.glob("gpu_*.csv")
                 if p.stem not in {"gpu_summary", "gpu_vs_serial", "gpu_vs_openmp"}}
    require(raw_names == set(ORDER), "Unexpected raw CSV inventory")
    medians = {}
    for name in ORDER:
        fields, rows = read(DATA / f"{name}.csv")
        require(fields == SCHEMA and len(rows) == 10, f"{name}: schema/count")
        require([int(r["run_index"]) for r in rows] == list(range(1, 11)),
                f"{name}: run indices")
        base = name.removesuffix("_repeat")
        if base == "gpu_thermal2":
            matrix, n, nnz = "thermal2", 1228045, 8580313
        else:
            _, geometry, grid = base.split("_")
            g = int(grid)
            n = g ** (2 if geometry == "2d" else 3)
            nnz = 5*g*g - 4*g if geometry == "2d" else 7*g**3 - 6*g*g
            matrix = f"poisson_{geometry}_{grid}"
        for r in rows:
            require(r["solver"] == "openmp_gpu" and r["matrix_name"] == matrix,
                    f"{name}: identity")
            require(int(r["n"]) == n and int(r["nnz"]) == nnz, f"{name}: N/NNZ")
            require(r["ranks"] == r["threads"] == r["converged"] == "1",
                    f"{name}: status/ranks/threads")
            require(0 < int(r["iterations"]) <= 10000, f"{name}: iterations")
            for field in ("solve_seconds", "true_relative_residual", "relative_solution_error"):
                value = float(r[field])
                require(math.isfinite(value) and value >= 0, f"{name}: {field}")
            require(float(r["true_relative_residual"]) <= 1e-8, f"{name}: residual")
        times = [float(r["solve_seconds"]) for r in rows]
        iterations = [int(r["iterations"]) for r in rows]
        require(base == "gpu_thermal2" or len(set(iterations)) == 1,
                f"{name}: inconsistent Poisson iterations")
        s = summary[name]
        stats = dict(n=n, nnz=nnz, runs=10, median_seconds=statistics.median(times),
                     mean_seconds=statistics.mean(times), std_seconds=statistics.stdev(times),
                     cv_percent=100*statistics.stdev(times)/statistics.mean(times),
                     min_iterations=min(iterations), max_iterations=max(iterations),
                     max_true_relative_residual=max(float(r["true_relative_residual"]) for r in rows),
                     max_relative_solution_error=max(float(r["relative_solution_error"]) for r in rows))
        for key, value in stats.items():
            close(s[key], value, f"{name}/{key}")
        medians[name] = stats["median_seconds"]
        print(f"{name}: median={medians[name]:.12f} s; iterations={min(iterations)}-{max(iterations)}")

    _, serial = read(ROOT / "docs/task_a/consolidated/serial_scaling_summary.csv")
    serial = {(r["geometry"], r["grid_size"]): r for r in serial if r["role"] == "core"}
    _, comparisons = read(DATA / "gpu_vs_serial.csv")
    require(len(comparisons) == 7, "Expected seven serial comparisons")
    require({(r["geometry"], r["grid_size"]) for r in comparisons} == set(serial),
            "Serial comparison configurations")
    for r in comparisons:
        s = serial[(r["geometry"], r["grid_size"])]
        _, raw = read(ROOT / s["source_csv"])
        cpu = statistics.median(float(x["solve_seconds"]) for x in raw)
        gpu_name = f"gpu_{r['geometry']}_{r['grid_size']}"
        for field in ("n", "nnz"):
            close(r[field], float(s[field]), f"serial {field}")
        close(r["serial_iterations"], float(s["iterations"]), "serial iterations")
        close(r["gpu_iterations"], float(summary[gpu_name]["min_iterations"]), "GPU iterations")
        close(r["serial_median_seconds"], cpu, "serial raw baseline")
        close(r["gpu_median_seconds"], medians[gpu_name], "GPU baseline")
        close(r["gpu_speedup"], cpu / medians[gpu_name], "serial speedup")

    _, scaling = read(ROOT / "docs/task_a/consolidated/scaling_summary.csv")
    _, comparisons = read(DATA / "gpu_vs_openmp.csv")
    require(len(comparisons) == 2 and {r["matrix_name"] for r in comparisons} ==
            {"poisson_2d_512", "poisson_2d_1024"}, "OpenMP configurations")
    for r in comparisons:
        candidates = [s for s in scaling if s["matrix"] == r["matrix_name"] and s["solver"] == "openmp"]
        best = min(candidates, key=lambda s: float(s["median_seconds"]))
        gpu = medians[r["matrix_name"].replace("poisson", "gpu")]
        for field in ("n", "nnz"):
            close(r[field], float(best[field]), f"OpenMP {field}")
        close(r["best_cpu_threads"], float(best["threads"]), "best CPU threads")
        close(r["cpu_openmp_median_seconds"], float(best["median_seconds"]), "OpenMP baseline")
        close(r["gpu_median_seconds"], gpu, "OpenMP comparison GPU median")
        close(r["gpu_speedup_vs_openmp"], float(best["median_seconds"]) / gpu, "OpenMP speedup")

    log = (DATA / "dsa5208_gpu_scaling.o1518726").read_text()
    import re
    require(re.findall(r"=== Starting (gpu_\w+) ===", log) == ORDER, "Log execution order")
    require(re.findall(r"^run_case\s+(gpu_\w+)", (DATA / "gpu_scaling_full.pbs").read_text(), re.M) == ORDER,
            "PBS execution order")
    require(log.count("Warm-up ") == 36 and log.count("| Measured run ") == 90,
            "Log warmup/repetition counts")
    require("=== GPU scaling campaign completed ===" in log, "Campaign completion")
    cache = (DATA / "CMakeCache.txt").read_text()
    for setting in ("CG_ENABLE_MPI:BOOL=OFF", "CG_ENABLE_OPENMP_GPU:BOOL=ON",
                    "CMAKE_BUILD_TYPE:STRING=Release", "CMAKE_CXX_FLAGS_RELEASE:STRING=-O2 -DNDEBUG"):
        require(setting in cache, f"Missing cache setting: {setting}")
    drift = 100 * (medians["gpu_2d_256_repeat"] / medians["gpu_2d_256"] - 1)
    print(f"Initial/final 256^2 drift: {drift:.9f}%")
    print("PASS: 90 raw rows; all summaries, baseline comparisons and campaign order validated.")
    return medians


if __name__ == "__main__":
    validate()
