"""Recreate Task A report figures from saved summaries; never run benchmarks.

Run from any directory: python scripts/plot_task_a_report_figures.py
Requires pandas, numpy, matplotlib, and Pillow. Outputs PNG (300 dpi) and SVG.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/task_a/consolidated"
OUT = ROOT / "docs/task_a/figures/revised"
MATRICES = {"poisson_2d_512": "Poisson 2D 512 × 512",
            "poisson_2d_1024": "Poisson 2D 1024 × 1024"}
THREADS = [1, 2, 4, 8, 16, 32, 36, 72]


def read_and_validate():
    scaling = pd.read_csv(DATA / "scaling_summary.csv", float_precision="round_trip")
    scheduling = pd.read_csv(DATA / "scheduling_summary.csv", float_precision="round_trip")
    correctness = pd.read_csv(DATA / "correctness_summary.csv", float_precision="round_trip")
    for frame in (scaling, scheduling):
        assert frame.runs.eq(10).all(), "Expected ten timed runs"
        assert frame.converged.eq(frame.runs).all()
        assert frame.median_seconds.gt(0).all()
        assert frame.median_seconds.between(frame.min_seconds, frame.max_seconds).all()
    for matrix in MATRICES:
        rows = scaling[scaling.matrix.eq(matrix)]
        serial = rows[rows.solver.eq("serial")]
        omp = rows[rows.solver.eq("openmp")].sort_values("threads")
        assert len(serial) == 1 and omp.threads.tolist() == THREADS
        t1 = omp.loc[omp.threads.eq(1), "median_seconds"].item()
        np.testing.assert_allclose(omp.speedup_vs_serial,
                                   serial.median_seconds.item() / omp.median_seconds, rtol=1e-12)
        np.testing.assert_allclose(omp.speedup_vs_omp1, t1 / omp.median_seconds, rtol=1e-12)
        np.testing.assert_allclose(omp.efficiency_percent,
                                   t1 / omp.median_seconds / omp.threads * 100, rtol=1e-12)
    common = correctness.merge(scaling, on="source_csv", suffixes=("_c", "_s"))
    assert len(common) == 4
    for column in ("median_seconds", "runs", "converged", "max_true_relative_residual",
                   "max_relative_solution_error"):
        np.testing.assert_array_equal(common[column + "_c"], common[column + "_s"])
    assert correctness.converged.eq(correctness.runs).all()
    final = scheduling[scheduling.experiment_group.eq("final_policies_and_chunks")].copy()
    assert len(final) == 8
    assert final.matrix.eq("poisson_2d_512").all() and final.threads.eq(8).all()
    expected = ["static_start", "dynamic_default", "dynamic_64", "dynamic_256",
                "dynamic_1024", "dynamic_4096", "guided_default", "static_end"]
    names = final.source_csv.str.removeprefix("openmp_sched_final_").str.removesuffix(".csv")
    assert names.tolist() == expected, "Final campaign order changed"
    final["label"] = names.str.replace("_", "\n").str.capitalize()
    return scaling, final


def canvas(title, caption):
    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.89, bottom=0.25)
    ax.set_title(title, pad=14)
    ax.grid(axis="y", which="both", alpha=0.23)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.12, 0.035, caption, fontsize=9, va="bottom", linespacing=1.5)
    return fig, ax


def save(fig, stem):
    fig.canvas.draw()
    for extension in ("png", "svg"):
        path = OUT / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        if extension == "png":
            with Image.open(path) as im:
                assert im.size == (2700, 1740)
                im.verify()
        else:
            assert ET.parse(path).getroot().tag.endswith("}svg")
        print(path.relative_to(ROOT))
    plt.close(fig)


def main():
    scaling, final = read_and_validate()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 14, "lines.linewidth": 1.8,
                         "svg.fonttype": "none"})
    specs = [
        ("figure1_openmp_execution_time", "OpenMP CG execution time", "median_seconds",
         "Median solver execution time (s; log scale)",
         "Median of 10 timed runs per configuration. Both axes are logarithmic.\n"
         "Matrix sizes were measured in separate Vanda jobs."),
        ("figure2_openmp_speedup", "OpenMP CG speedup", "speedup_vs_serial",
         "Speedup relative to serial (×)",
         "Speedup = T_serial / T_OpenMP(p), using median solver times from 10 timed runs.\n"
         "The baseline is the independent serial implementation for each matrix."),
        ("figure3_openmp_efficiency", "OpenMP CG parallel efficiency", "efficiency_percent",
         "Parallel efficiency (%)",
         "Efficiency = [T_OpenMP(1) / T_OpenMP(p)] / p × 100%, using median solver times.\n"
         "The baseline is one-thread OpenMP; each configuration has 10 timed runs."),
    ]
    for stem, title, column, ylabel, caption in specs:
        fig, ax = canvas(title, caption)
        ax.set_xscale("log", base=2)
        for (matrix, label), color, marker in zip(MATRICES.items(), ["#0072B2", "#D55E00"], ["o", "s"]):
            rows = scaling[scaling.solver.eq("openmp") & scaling.matrix.eq(matrix)].sort_values("threads")
            line, = ax.plot(rows.threads.to_numpy(), rows[column].to_numpy(),
                            label=label, color=color, marker=marker, markersize=5)
            np.testing.assert_array_equal(line.get_xdata(), rows.threads)
            np.testing.assert_array_equal(line.get_ydata(), rows[column])
        ax.set_xticks(THREADS, labels=[str(t) for t in THREADS])
        ax.xaxis.set_minor_locator(NullLocator())
        # Keep close numerical coordinates (32 and 36) while avoiding label overlap.
        for tick in ax.get_xticklabels():
            if tick.get_text() == "32":
                tick.set_horizontalalignment("right")
            elif tick.get_text() == "36":
                tick.set_horizontalalignment("left")
        ax.set_xlim(0.85, 85)
        ax.set_xlabel("OpenMP threads (log₂ scale)")
        ax.set_ylabel(ylabel)
        if column == "median_seconds":
            ax.set_yscale("log", base=10)
            ax.set_yticks([1, 2, 5, 10, 20, 40])
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.set_ylim(0.7, 45)
        else:
            ax.set_ylim(bottom=0, top=105 if column == "efficiency_percent" else None)
        assert ax.xaxis.get_transform().base == 2
        np.testing.assert_allclose(ax.xaxis.get_transform().transform(THREADS), np.log2(THREADS))
        assert ax.get_yscale() == ("log" if column == "median_seconds" else "linear")
        ax.legend(frameon=False)
        save(fig, stem)

    drift = final.median_seconds.iloc[-1] / final.median_seconds.iloc[0]
    fig, ax = canvas("OpenMP scheduling · Poisson 2D 512 × 512 · 8 threads",
                      "Final campaign only; configurations retain their recorded execution order.\n"
                      f"Medians of 10 timed runs. Static end/start = {drift:.3f}×; timing drift confounds policy comparisons.")
    # Points avoid suggesting a zero baseline on a logarithmic time axis.
    line, = ax.plot(final.label.tolist(), final.median_seconds.to_numpy(),
                    linestyle="none", marker="o", color="#0072B2", markersize=7)
    np.testing.assert_array_equal(line.get_ydata(), final.median_seconds)
    assert list(line.get_xdata()) == final.label.tolist()
    ax.set_yscale("log", base=10)
    assert ax.yaxis.get_transform().base == 10
    ax.set_ylim(0.7, 30)
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel("Median solver execution time (s; log₁₀ scale)")
    ax.set_xlabel("Scheduling configuration")
    ax.tick_params(axis="x", labelsize=9)
    for i, value in enumerate(final.median_seconds):
        ax.annotate(f"{value:.3f}", (i, value), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9)
    save(fig, "figure4_openmp_scheduling")
    print("Validated: 48 scaling plot values, 8 scheduling values, metric formulas, "
          "4 shared correctness rows, numeric log coordinates, and 8 output files.")


if __name__ == "__main__":
    main()
