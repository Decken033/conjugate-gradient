"""Check Task B report tables against selected MPI CSVs and the thermal2 log."""

import csv
import math
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "REPORT_TASK_B.md"
REPORT_ZH = ROOT / "REPORT_TASK_B_ZH.md"
RESULTS = ROOT / "results" / "mpi"
RANKS_POISSON = (1, 2, 4, 8, 16, 32, 36, 72)
RANKS_THERMAL = (1, 2, 4, 8, 16)
FIELDS = (
    "solver", "matrix_name", "n", "nnz", "ranks", "threads", "run_index",
    "iterations", "converged", "solve_seconds", "true_relative_residual",
    "relative_solution_error",
)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def report_rows(text, heading):
    section = text.split(heading, 1)[1].split("\n### ", 1)[0]
    rows = {}
    for line in section.splitlines():
        if re.match(r"^\| \d+ \|", line):
            cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
            rows[int(cells[0])] = cells
    return rows


def records(path, matrix_name, n, nnz, ranks):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        check(tuple(reader.fieldnames or ()) == FIELDS, f"CSV schema: {path}")
        rows = list(reader)
    check(len(rows) == 10, f"CSV row count: {path}")
    check({int(row["run_index"]) for row in rows} == set(range(10)), f"CSV indices: {path}")
    for row in rows:
        actual = (row["solver"], row["matrix_name"], int(row["n"]),
                  int(row["nnz"]), int(row["ranks"]), int(row["threads"]))
        check(actual == ("mpi", matrix_name, n, nnz, ranks, 1), f"CSV metadata: {path}")
        check(int(row["converged"]) == 1, f"Nonconverged record: {path}")
        check(0 < float(row["true_relative_residual"]) <= 1e-8,
              f"Residual tolerance: {path}")
        for key in ("solve_seconds", "true_relative_residual", "relative_solution_error"):
            check(math.isfinite(float(row[key])), f"Nonfinite {key}: {path}")
        check(float(row["solve_seconds"]) > 0, f"Nonpositive solve time: {path}")
    return rows


def check_group(text, heading, ranks_list, filename, matrix_name, n, nnz, thermal=False):
    table = report_rows(text, heading)
    check(set(table) == set(ranks_list), f"Report ranks: {heading}")
    grouped = {p: records(RESULTS / filename.format(p=p), matrix_name, n, nnz, p)
               for p in ranks_list}
    means = {p: statistics.mean(float(row["solve_seconds"]) for row in grouped[p])
             for p in ranks_list}
    for p in ranks_list:
        data = grouped[p]
        cells = table[p]
        times = [float(row["solve_seconds"]) for row in data]
        iterations = {int(row["iterations"]) for row in data}
        check(len(iterations) == 1, f"Nonconstant iteration count: {p}")
        common = [str(p), "10/10", f"{means[p]:.6f}",
                  f"{statistics.stdev(times):.6f}"]
        if thermal:
            common += [f"{statistics.median(times):.6f}"]
        common += [f"{min(times):.6f}–{max(times):.6f}",
                   f"{means[1] / means[p]:.4f}"]
        if thermal:
            common += [f"{100 * means[1] / means[p] / p:.2f}%",
                       f"{next(iter(iterations)):,}",
                       f"{max(float(row['true_relative_residual']) for row in data):.6e}",
                       f"{max(float(row['relative_solution_error']) for row in data):.6e}"]
        else:
            common += [f"{means[1] / means[p] / p:.4f}",
                       f"{next(iter(iterations)):,}"]
        check(cells == common, f"Report table mismatch at {heading}, {p} ranks: {cells} != {common}")
    return grouped


def check_poisson_figure(groups):
    root = ET.parse(ROOT / "docs" / "task_b" / "mpi_scaling.svg").getroot()
    series = [element.attrib["points"] for element in root.iter()
              if element.tag.endswith("polyline")]
    check(len(series) == 4, "Figure 1 series count")
    for index, (group, efficiency) in enumerate(
            ((groups[0], False), (groups[1], False),
             (groups[0], True), (groups[1], True))):
        points = [tuple(map(float, point.split(",")))
                  for point in series[index].split()]
        check(len(points) == len(RANKS_POISSON), f"Figure 1 point count: {index}")
        baseline = statistics.mean(float(row["solve_seconds"]) for row in group[1])
        for (_, y), p in zip(points, RANKS_POISSON):
            mean = statistics.mean(float(row["solve_seconds"]) for row in group[p])
            value = baseline / mean
            expected_y = 525 - value / p * 420 if efficiency else 525 - value * 105
            check(abs(y - expected_y) <= 1.5,
                  f"Figure 1 value mismatch: series {index}, {p} ranks")


def check_thermal_log(grouped):
    log = (RESULTS / "thermal2" / "dsa5208_mpi_thermal2.o1495684").read_text(
        encoding="utf-8")
    markers = re.findall(r"=== thermal2 with (\d+) MPI ranks ===", log)
    check(markers == [str(p) for p in RANKS_THERMAL], "Thermal2 launch sequence")
    blocks = re.split(r"=== thermal2 with \d+ MPI ranks ===", log)[1:]
    for p, block in zip(RANKS_THERMAL, blocks):
        check("Loading Matrix Market: data/thermal2/thermal2.mtx" in block,
              f"Logged matrix path: {p}")
        check("Matrix: N=1228045 NNZ=8580313" in block, f"Logged matrix shape: {p}")
        check(f"Distributing matrix to {p} ranks" in block, f"Logged ranks: {p}")
        check(len(re.findall(r"\[warmup \d/4\]", block)) == 4, f"Warmups: {p}")
        check(len(re.findall(r"\[run \d+/10\]", block)) == 10, f"Timed runs: {p}")
        check("Converged: 10/10" in block, f"Logged convergence: {p}")
        check(f"Results written to: results/mpi/thermal2/thermal2_p{p}_1495684.stdct-mgmt-02.csv" in block,
              f"Logged CSV: {p}")
        check(block.count("libpbs.so.0") == 2 and block.count("(ignored)") == 2,
              f"Launcher warnings: {p}")
        median = statistics.median(float(row["solve_seconds"]) for row in grouped[p])
        logged = float(re.search(r"Median solve time: ([\d.]+) s", block).group(1))
        check(abs(logged - median) <= 0.000051, f"Logged median: {p}")
        check(int(re.search(r"Iterations: (\d+)", block).group(1)) ==
              int(grouped[p][0]["iterations"]), f"Logged iterations: {p}")
        logged_residual = float(re.search(r"True rel. residual: ([\deE+.-]+)", block).group(1))
        first_residual = float(grouped[p][0]["true_relative_residual"])
        check(math.isclose(logged_residual, first_residual, rel_tol=6e-7),
              f"Logged first-run residual: {p}")


def check_bilingual_parity(english, chinese):
    sections = (("### 5.1", RANKS_POISSON),
                ("### 5.2", RANKS_POISSON),
                ("### 5.4", RANKS_THERMAL))
    for heading, ranks_list in sections:
        source = report_rows(english, heading)
        translation = report_rows(chinese, heading)
        check(set(source) == set(translation) == set(ranks_list),
              f"Bilingual report ranks: {heading}")
        for p in ranks_list:
            check(source[p] == translation[p],
                  f"Bilingual table mismatch: {heading}, {p} ranks")

    for pattern, label in ((r"!\[[^]]*\]\(([^)]+)\)", "figures"),
                           (r"(?ms)^\$\$\n(.*?)\n\$\$$", "display formulas"),
                           (r"(?ms)^```bash\n(.*?)\n```$", "Bash commands")):
        check(re.findall(pattern, english) == re.findall(pattern, chinese),
              f"Bilingual {label} mismatch")


def main():
    text = REPORT.read_text(encoding="utf-8")
    translated = REPORT_ZH.read_text(encoding="utf-8")
    poisson_512 = check_group(text, "### 5.1", RANKS_POISSON,
                              "mpi_2d512_p{p}_1482451.stdct-mgmt-02.csv",
                              "poisson_2d_512", 262144, 1308672)
    poisson_1024 = check_group(text, "### 5.2", RANKS_POISSON,
                               "mpi_2d1024_p{p}_1482491.stdct-mgmt-02.csv",
                               "poisson_2d_1024", 1048576, 5238784)
    thermal = check_group(text, "### 5.4", RANKS_THERMAL,
                          "thermal2/thermal2_p{p}_1495684.stdct-mgmt-02.csv",
                          "thermal2", 1228045, 8580313, thermal=True)
    check_thermal_log(thermal)
    check_poisson_figure((poisson_512, poisson_1024))
    check_bilingual_parity(text, translated)
    for figure in re.findall(r"!\[[^]]*\]\(([^)]+)\)", text):
        check((ROOT / figure).is_file(), f"Missing report figure: {figure}")
    print("Verified 21 CSVs, 210 timed records, both reports' 21 table rows, "
          "shared figures/formulas/commands, Figure 1 data, thermal2 log, and figure paths.")


if __name__ == "__main__":
    main()
