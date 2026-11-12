"""Check Task B report tables against selected MPI CSVs and retained logs."""

import csv
import math
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from analyze_mpi_scaling_extension import STRONG_RANKS, WEAK_CASES, load_groups


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "REPORT_TASK_B.md"
RESULTS = ROOT / "results" / "mpi"
LOGS = RESULTS / "evidence" / "logs"
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
    return parse_rows(section)


def parse_rows(section):
    rows = {}
    for line in section.splitlines():
        if re.match(r"^\| \d+ \|", line):
            cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
            ranks = int(cells[0])
            check(ranks not in rows, f"Duplicate report table rank: {ranks}")
            rows[ranks] = cells
    return rows


def records(path, matrix_name, n, nnz, ranks):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        check(tuple(reader.fieldnames or ()) == FIELDS, f"CSV schema: {path}")
        rows = list(reader)
    check(len(rows) == 10, f"CSV row count: {path}")
    check({int(row["run_index"]) for row in rows} == set(range(10)), f"CSV indices: {path}")
    for row in rows:
        check(set(row) == set(FIELDS) and all(row[key] not in (None, "") for key in FIELDS),
              f"Incomplete or extra CSV fields: {path}")
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


def check_3d_group(english):
    job = "1502451.stdct-mgmt-02"
    sizes = {40: (64000, 438400), 64: (262144, 1810432),
             102: (1061208, 7366032)}
    expected = {f"mpi_3d{size}_p{p}_{job}.csv"
                for size in sizes for p in RANKS_POISSON}
    actual = {path.name for path in (RESULTS / "3d").glob("*.csv")}
    check(actual == expected, "3D CSV inventory")
    for size, (n, nnz) in sizes.items():
        after_title = english.split(f"**{size}³", 1)[1]
        section = "| 1 |" + after_title.split("| 1 |", 1)[1].split("\n\n", 1)[0]
        table = parse_rows(section)
        check(set(table) == set(RANKS_POISSON), f"3D table ranks: {size}")
        grouped = {p: records(RESULTS / "3d" / f"mpi_3d{size}_p{p}_{job}.csv",
                              f"poisson_3d_{size}", n, nnz, p)
                   for p in RANKS_POISSON}
        means = {p: statistics.mean(float(row["solve_seconds"]) for row in grouped[p])
                 for p in RANKS_POISSON}
        for p, data in grouped.items():
            times = [float(row["solve_seconds"]) for row in data]
            iterations = {int(row["iterations"]) for row in data}
            check(len(iterations) == 1 and 0 < next(iter(iterations)) <= 10000,
                  f"3D iterations: {size}, {p}")
            errors = [float(row["relative_solution_error"]) for row in data]
            check(all(0 <= error < 1e-4 for error in errors),
                  f"3D solution error: {size}, {p}")
            expected_row = [str(p), "10/10", f"{means[p]:.6f}",
                            f"{statistics.stdev(times):.6f}",
                            f"{min(times):.6f}–{max(times):.6f}",
                            f"{means[1] / means[p]:.4f}",
                            f"{means[1] / means[p] / p:.4f}",
                            str(next(iter(iterations))),
                            f"{max(float(row['true_relative_residual']) for row in data):.6e}",
                            f"{max(errors):.6e}"]
            check(table[p] == expected_row,
                  f"3D report row mismatch: {size}, {p}")


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
    log = (LOGS / "dsa5208_mpi_thermal2.o1495684").read_text(
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


def check_3d_log():
    log = (LOGS / "dsa5208_mpi_scaling_3d.o1502451").read_text(encoding="utf-8")
    check("Job ID: 1502451.stdct-mgmt-02" in log, "3D job ID")
    check("=== 3D MPI scaling completed ===" in log, "3D completion marker")
    for host in ("cn-033.stdct.loc", "cn-040.stdct.loc"):
        check(f"36 {host}" in log, f"3D allocated host: {host}")
    blocks = re.findall(r"--- Grid: (\d+)\^3; ranks: (\d+) ---(.*?)(?=--- Grid:|=== 3D MPI scaling completed ===)",
                        log, re.S)
    check([(int(size), int(p)) for size, p, _ in blocks] ==
          [(size, p) for size in (40, 64, 102) for p in RANKS_POISSON],
          "3D configuration order")
    for size, p, block in blocks:
        check(len(re.findall(r"\[warmup \d/4\]", block)) == 4,
              f"3D warmups: {size}, {p}")
        check(len(re.findall(r"\[run \d+/10\]", block)) == 10,
              f"3D runs: {size}, {p}")
        check("Converged: 10/10" in block, f"3D convergence: {size}, {p}")
        check(f"mpi_3d{size}_p{p}_1502451.stdct-mgmt-02.csv" in block,
              f"3D output path: {size}, {p}")
        path = RESULTS / "3d" / f"mpi_3d{size}_p{p}_1502451.stdct-mgmt-02.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        logged_median = float(re.search(r"Median solve time: ([\d.]+) s", block).group(1))
        csv_median = statistics.median(float(row["solve_seconds"]) for row in rows)
        check(math.isclose(logged_median, csv_median, rel_tol=6e-6, abs_tol=6e-8),
              f"3D median: {size}, {p}")
        logged_iterations = int(re.search(r"Iterations: (\d+)", block).group(1))
        check(logged_iterations == int(rows[0]["iterations"]),
              f"3D first-run iterations: {size}, {p}")
        logged_residual = float(re.search(r"True rel. residual: ([\deE+.-]+)", block).group(1))
        check(math.isclose(logged_residual, float(rows[0]["true_relative_residual"]),
                           rel_tol=6e-6), f"3D first-run residual: {size}, {p}")


def check_extension_log(kind, job, rows):
    log = (LOGS / f"dsa5208_mpi_{kind}_256.o{job}").read_text(encoding="utf-8")
    check(f"Job ID: {job}.stdct-mgmt-02" in log, f"{kind} job ID")
    check("Loading OpenMPI/4.1.6-GCC-13.2.0" in log, f"{kind} MPI module")
    check("mpirun (Open MPI) 4.1.6" in log, f"{kind} MPI runtime")
    check("Built target run_mpi_benchmark" in log, f"{kind} build")
    check(f"{kind.capitalize()} scaling completed:" in log, f"{kind} completion")
    hostfile = ROOT / "results" / "mpi" / "extension" / f"{kind}_{job}.stdct-mgmt-02" / "hosts.txt"
    hosts = hostfile.read_text(encoding="utf-8").splitlines()
    check(len(hosts) == 8 and all(line.endswith(" slots=36") for line in hosts),
          f"{kind} hostfile allocation")
    check(all(line in log for line in hosts), f"{kind} logged hostfile")
    check(len(re.findall(r"^\s*36 (?:cn|gn)-", log, re.M)) == 8,
          f"{kind} logged PBS slots")
    if kind == "strong":
        pattern = r"=== Strong scaling: 1024 x 1024, ranks=(\d+),[^\n]* ==="
        expected = [str(p) for p in STRONG_RANKS]
    else:
        pattern = r"=== Weak scaling: (\d+) x \1, ranks=(\d+),[^\n]* ==="
        expected = [(str(grid), str(p)) for p, grid in WEAK_CASES]
    markers = re.findall(pattern, log)
    check(markers == expected, f"{kind} launch sequence")
    blocks = re.split(pattern, log)[1:]
    stride = 2 if kind == "strong" else 3
    for row, block in zip(rows, blocks[stride - 1::stride]):
        p = row["ranks"]
        check(len(re.findall(r"\[warmup \d/4\]", block)) == 4,
              f"{kind} warmups at {p}")
        check(len(re.findall(r"\[run \d+/10\]", block)) == 10,
              f"{kind} timed runs at {p}")
        check("Converged: 10/10" in block, f"{kind} convergence at {p}")
        check(row["source"] in block, f"{kind} logged CSV at {p}")
        logged_median = float(re.search(r"Median solve time: ([\d.]+) s", block).group(1))
        check(math.isclose(logged_median, row["median_seconds"], abs_tol=0.00051),
              f"{kind} logged median at {p}")
        check(int(re.search(r"Iterations: (\d+)", block).group(1)) == row["iterations"],
              f"{kind} logged iterations at {p}")


def check_extension_table(text, heading, rows, kind):
    table = report_rows(text, heading)
    check(set(table) == {row["ranks"] for row in rows}, f"{kind} report inventory")
    for row in rows:
        common = [str(row["ranks"])]
        if kind == "weak":
            common += [f'{row["grid"]}x{row["grid"]}', str(row["unknowns_per_rank"])]
        common += ["10/10", f'{row["mean_seconds"]:.6f}',
                   f'{row["sd_seconds"]:.6f}',
                   f'{row["min_seconds"]:.6f}-{row["max_seconds"]:.6f}']
        if kind == "strong":
            common += [f'{row["speedup"]:.4f}', f'{row["efficiency"]:.4f}']
        else:
            common += [f'{row["weak_efficiency"]:.6f}']
        common += [str(row["iterations"]), f'{row["seconds_per_iteration"]:.6f}',
                   f'{row["max_residual"]:.3e}', f'{row["max_solution_error"]:.3e}']
        check(table[row["ranks"]] == common, f"{kind} report numbers at {row['ranks']}")
    return table


def check_extension_artifacts(strong, weak, english):
    output = ROOT / "results" / "mpi" / "extension" / "analysis"
    for kind, rows, headings, keys in (
        ("strong", strong, "### 7.6",
         ("mean_seconds", "speedup", "efficiency", "seconds_per_iteration")),
        ("weak", weak, "### 7.7",
         ("mean_seconds", "seconds_per_iteration", "weak_efficiency", "iterations")),
    ):
        csv_path = output / f"{kind}_summary.csv"
        with csv_path.open(newline="", encoding="utf-8") as stream:
            saved = list(csv.DictReader(stream))
        check(len(saved) == len(rows), f"{kind} summary row count")
        for actual, expected in zip(saved, rows):
            check(set(actual) == set(expected), f"{kind} summary columns")
            for key, value in expected.items():
                if isinstance(value, (int, float)):
                    check(math.isclose(float(actual[key]), value, rel_tol=1e-12, abs_tol=1e-12),
                          f"{kind} summary {key}")
                else:
                    check(actual[key] == value, f"{kind} summary {key}")
        check_extension_table(english, headings, rows, kind)
        chart = ET.parse(output / f"{kind}_scaling.svg").getroot()
        polylines = [element for element in chart.iter() if element.tag.endswith("polyline")]
        check(len(polylines) == 4, f"{kind} chart panels")
        chart_text = " ".join(element.text or "" for element in chart.iter())
        check("log2 scale" in chart_text and "baseline" in chart_text.lower(),
              f"{kind} chart scale and baseline")
        for panel, key in enumerate(keys):
            points = [tuple(map(float, item.split(",")))
                      for item in polylines[panel].attrib["points"].split()]
            check(len(points) == len(rows), f"{kind} chart point count")
            maximum = max(float(row[key]) for row in rows) * 1.12
            for (x, y), row in zip(points, rows):
                expected_x = 65 + (panel % 2) * 455 + math.log2(row["ranks"]) / 8 * 375
                expected_y = 375 + (panel // 2) * 390 - float(row[key]) / maximum * 280
                check(abs(x - expected_x) < 0.11 and abs(y - expected_y) < 0.11,
                      f"{kind} chart coordinate at {row['ranks']}")
        check_extension_log(kind, "1509066" if kind == "strong" else "1509067", rows)
    for text in (english,):
        for marker in ("28.855629833", "7.488011", "24.540093", "0.0369419313",
                       "291.132526", "3,439", "collective"):
            check(marker in text, f"Missing extension discussion number: {marker}")
    docx_path = ROOT / "report" / "DSA5208_Project3_Person_B_MPI_Report.docx"
    if not docx_path.is_file():
        print("Skipped optional local DOCX check (document not present).")
        return
    with ZipFile(docx_path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        doc_text = " ".join(element.text or "" for element in root.iter()
                            if element.tag.endswith("}t"))
        images = [name for name in archive.namelist() if name.startswith("word/media/")]
    tables = [element for element in root.iter() if element.tag.endswith("}tbl")]
    check(len(tables) >= 12, "DOCX extension table count")
    for table, rows in zip(tables[-2:], (strong, weak)):
        actual = []
        for tr in table:
            if not tr.tag.endswith("}tr"):
                continue
            cells = []
            for tc in tr:
                if tc.tag.endswith("}tc"):
                    cells.append("".join(t.text or "" for t in tc.iter()
                                         if t.tag.endswith("}t")))
            actual.append(cells)
        check(len(actual) == len(rows) + 1, "DOCX extension table row count")
        kind = "strong" if len(rows) == 10 else "weak"
        expected = check_extension_table(english, "### 7.6" if kind == "strong" else "### 7.7",
                                         rows, kind)
        for record, cells in zip(rows, actual[1:]):
            check(cells == expected[record["ranks"]],
                  f"DOCX {kind} table at {record['ranks']} ranks")
    for marker in ("1509066", "1509067", "7.488011", "24.540093",
                   "291.132526", "3439", "0.084656", "communication",
                   "60 benchmark CSV", "600 timed runs", "7.6 Extended", "7.7 Weak"):
        check(marker in doc_text, f"DOCX extension key number: {marker}")
    check(len(images) >= 6, "DOCX extension figures")


def main():
    text = REPORT.read_text(encoding="utf-8")
    poisson_512 = check_group(text, "### 7.1", RANKS_POISSON,
                              "mpi_2d512_p{p}_1482451.stdct-mgmt-02.csv",
                              "poisson_2d_512", 262144, 1308672)
    poisson_1024 = check_group(text, "### 7.2", RANKS_POISSON,
                               "mpi_2d1024_p{p}_1482491.stdct-mgmt-02.csv",
                               "poisson_2d_1024", 1048576, 5238784)
    thermal = check_group(text, "### 7.4", RANKS_THERMAL,
                          "thermal2/thermal2_p{p}_1495684.stdct-mgmt-02.csv",
                          "thermal2", 1228045, 8580313, thermal=True)
    check_thermal_log(thermal)
    check_3d_log()
    check_3d_group(text)
    check_poisson_figure((poisson_512, poisson_1024))
    strong, weak = load_groups(
        RESULTS / "extension" / "strong_1509066.stdct-mgmt-02",
        RESULTS / "extension" / "weak_1509067.stdct-mgmt-02")
    check_extension_artifacts(strong, weak, text)
    for figure in re.findall(r"!\[[^]]*\]\(([^)]+)\)", text):
        check((ROOT / figure).is_file(), f"Missing report figure: {figure}")
    print("Verified 60 CSVs, 600 timed records, English scaling tables, "
          "extension summaries and log-rank figures (optional DOCX check when present), "
          "thermal2, 3D, strong and weak logs, and figure paths.")


if __name__ == "__main__":
    main()
