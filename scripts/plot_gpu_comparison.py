"""Plot Task A GPU extension figures from saved benchmark comparisons.

Run:
    python scripts/plot_gpu_comparison.py

No benchmarks are executed.
Outputs three PNG/SVG figure pairs at 300 DPI.
"""

from pathlib import Path
import argparse
import xml.etree.ElementTree as ET

from validate_gpu_evidence import validate

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/gpu_evidence"
OUT = ROOT / "docs/task_a/figures/gpu"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"


def load_data():
    serial = pd.read_csv(DATA / "gpu_vs_serial.csv")
    openmp = pd.read_csv(DATA / "gpu_vs_openmp.csv")

    assert len(serial) == 7
    assert len(openmp) == 2

    assert serial.serial_median_seconds.gt(0).all()
    assert serial.gpu_median_seconds.gt(0).all()

    np.testing.assert_allclose(
        serial.gpu_speedup,
        serial.serial_median_seconds / serial.gpu_median_seconds,
        rtol=1e-12,
    )

    np.testing.assert_allclose(
        openmp.gpu_speedup_vs_openmp,
        openmp.cpu_openmp_median_seconds /
        openmp.gpu_median_seconds,
        rtol=1e-12,
    )

    assert set(serial.geometry) == {"2d", "3d"}
    assert set(openmp.matrix_name) == {
        "poisson_2d_512", "poisson_2d_1024"
    }

    return serial, openmp


def canvas(title, caption):
    fig, ax = plt.subplots(figsize=(9, 5.8))

    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        top=0.89,
        bottom=0.25,
    )

    ax.set_title(title, pad=14)
    ax.grid(axis="y", which="both", alpha=0.23)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.12, 0.035,
        caption,
        fontsize=9,
        va="bottom",
        linespacing=1.5,
    )

    return fig, ax


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)

    fig.canvas.draw()

    for ext in ("png", "svg"):
        path = OUT / f"{stem}.{ext}"

        fig.savefig(
            path,
            dpi=300,
            facecolor="white",
        )

        if ext == "png":
            with Image.open(path) as image:
                assert image.size == (2700, 1740)
                image.verify()
        else:
            # Matplotlib emits trailing spaces in SVG markup. Remove only
            # line-end whitespace so regenerated figures pass Git checks.
            svg = path.read_text(encoding="utf-8")
            path.write_text("\n".join(line.rstrip(" \t") for line in svg.splitlines()) + "\n",
                            encoding="utf-8", newline="\n")
            assert ET.parse(path).getroot().tag.endswith("}svg")

        print(path.relative_to(ROOT))

    plt.close(fig)


def plot_runtime(serial):
    fig, ax = canvas(
        "Serial CPU versus GPU CG execution time",
        "Median of 10 measured solves per configuration; both axes are logarithmic.\n"
        "GPU time includes device allocation and transfers. CPU and GPU used different Vanda nodes.",
    )

    for geometry, color, marker in [
        ("2d", BLUE, "o"),
        ("3d", ORANGE, "s"),
    ]:
        rows = serial.loc[
            serial.geometry.eq(geometry)
        ].sort_values("n")

        label = geometry.upper()

        ax.plot(
            rows.n,
            rows.serial_median_seconds,
            color=color,
            marker=marker,
            linestyle="--",
            label=f"{label} serial CPU",
        )

        ax.plot(
            rows.n,
            rows.gpu_median_seconds,
            color=color,
            marker=marker,
            linestyle="-",
            label=f"{label} NVIDIA A40",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Number of unknowns (N)")
    ax.set_ylabel("Median solver time (s; log scale)")
    ax.legend(frameon=False, fontsize=9, ncol=2)

    save(fig, "figure7_gpu_runtime_scaling")


def plot_speedup(serial):
    fig, ax = canvas(
        "GPU CG speedup relative to serial CPU",
        "Speedup = serial CPU median / GPU median; 1x indicates equal runtime.\n"
        "Measured end-to-end GPU solver time; cross-node comparison.",
    )

    for geometry, color, marker in [
        ("2d", BLUE, "o"),
        ("3d", ORANGE, "s"),
    ]:
        rows = serial.loc[
            serial.geometry.eq(geometry)
        ].sort_values("n")

        ax.plot(
            rows.n,
            rows.gpu_speedup,
            color=color,
            marker=marker,
            label=f"Poisson {geometry.upper()}",
        )

        # Separate 2D and 3D annotations at similar N values.
        # On the logarithmic y-axis, place 2D labels above
        # their markers and 3D labels below their markers.
        for _, row in rows.iterrows():
            if row.gpu_speedup < 1.0:
                if geometry == "2d":
                    # 0.54x: below the blue point.
                    offset = (0, -12)
                    horizontal_align = "center"
                    vertical_align = "top"
                else:
                    # 0.61x: left of the orange point.
                    offset = (-8, 12)
                    horizontal_align = "right"
                    vertical_align = "bottom"
            else:
                offset = (0, 13) if geometry == "2d" else (0, -17)
                horizontal_align = "center"
                vertical_align = "bottom" if geometry == "2d" else "top"

            ax.annotate(
                f"{row.gpu_speedup:.2f}x",
                (row.n, row.gpu_speedup),
                xytext=offset,
                textcoords="offset points",
                ha=horizontal_align,
                va=vertical_align,
                fontsize=9,
            )

    ax.axhline(
        1.0,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="CPU/GPU parity",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Leave sufficient room for the smallest-case annotations.
    ax.set_xlim(left=2500)
    ax.set_ylim(bottom=0.35)

    ax.set_xlabel("Number of unknowns (N)")
    ax.set_ylabel("GPU speedup over serial CPU")
    ax.legend(frameon=False, fontsize=9)

    save(fig, "figure8_gpu_speedup_vs_serial")


def plot_openmp(openmp):
    rows = openmp.sort_values("n")

    fig, ax = canvas(
        "GPU versus best measured CPU OpenMP",
        "CPU baseline is the fastest median among tested thread counts (1-72).\n"
        "GPU timing includes device allocation and transfers; experiments used different nodes.",
    )

    x = np.arange(len(rows))
    width = 0.32

    cpu = ax.bar(
        x - width / 2,
        rows.cpu_openmp_median_seconds,
        width,
        color=BLUE,
        label="Best CPU OpenMP",
    )

    gpu = ax.bar(
        x + width / 2,
        rows.gpu_median_seconds,
        width,
        color=GREEN,
        label="NVIDIA A40",
    )

    labels = [
        f"Poisson 2D {n} x {n}"
        for n in (512, 1024)
    ]

    ax.set_xticks(x, labels)
    ax.set_ylabel("Median solver time (s)")
    ax.legend(frameon=False)

    for index, (_, row) in enumerate(rows.iterrows()):
        ax.annotate(
            f"{row.gpu_speedup_vs_openmp:.2f}x GPU speedup\n"
            f"CPU: {int(row.best_cpu_threads)} threads",
            xy=(x[index], row.cpu_openmp_median_seconds),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_ylim(
        0,
        rows.cpu_openmp_median_seconds.max() * 1.22,
    )

    save(fig, "figure9_gpu_vs_cpu_openmp")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true",
                        help="Check evidence and plot inputs without rewriting figures")
    args = parser.parse_args()
    validate()
    serial, openmp = load_data()
    if args.validate_only:
        print("PASS: plot inputs validated; figures unchanged.")
        return

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "lines.linewidth": 1.8,
        "svg.fonttype": "none",
    })

    plot_runtime(serial)
    plot_speedup(serial)
    plot_openmp(openmp)

    print("GPU figure generation completed.")


if __name__ == "__main__":
    main()
