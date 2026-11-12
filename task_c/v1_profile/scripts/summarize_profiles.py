#!/usr/bin/env python3
"""Aggregate per-rank profiling CSV files produced by MPI/Hybrid runners."""
import csv
import glob
import os
import statistics
import sys

if len(sys.argv) < 3:
    print("Usage: python3 scripts/summarize_profiles.py <output.csv> <rank-profile-glob> [glob ...]")
    print('Example: python3 scripts/summarize_profiles.py results/profile_summary.csv "results/hybrid/*.profile.rank*.csv"')
    raise SystemExit(2)

out_path = sys.argv[1]
files = []
for pattern in sys.argv[2:]:
    files.extend(glob.glob(pattern))
files = sorted(set(files))
if not files:
    raise SystemExit("No profile files matched the supplied pattern(s)")

by_operation = {}
for path in files:
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            op = row["operation"]
            item = by_operation.setdefault(op, {"times": [], "calls": [], "solver": row["solver"]})
            item["times"].append(float(row["total_seconds"]))
            item["calls"].append(int(row["calls"]))

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
with open(out_path, "w", newline="") as f:
    fields = ["solver", "operation", "rank_count", "mean_rank_total_s", "max_rank_total_s", "min_rank_total_s", "max_rank_avg_s", "calls_per_rank"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for op, v in by_operation.items():
        times = v["times"]
        calls = v["calls"]
        max_avg = max((t / c if c else 0.0) for t, c in zip(times, calls))
        w.writerow({
            "solver": v["solver"], "operation": op, "rank_count": len(times),
            "mean_rank_total_s": statistics.mean(times),
            "max_rank_total_s": max(times), "min_rank_total_s": min(times),
            "max_rank_avg_s": max_avg,
            "calls_per_rank": max(calls) if calls else 0,
        })
print(f"Wrote {out_path} from {len(files)} rank profile files")
