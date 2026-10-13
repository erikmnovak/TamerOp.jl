"""Render the recorded interaction samples; no benchmark work is performed."""
from pathlib import Path
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter


def main():
    here = Path(__file__).resolve().parent
    docs = here.parents[1]
    samples = json.loads((here / "measurements.json").read_text())["samples"]
    labels = {"continuous": "Ring\n49 cells",
              "continuous-large": "Image\n2,601 cells",
              "continuous-slices": "Square modules\nnew line restrictions"}
    with plt.style.context(docs / "assets/benchmarks/report.mplstyle"):
        fig, ax = plt.subplots(figsize=(7.4, 3.9), layout="constrained")
        for row, sample in enumerate(samples):
            warm = [x / 1000 for x in sample.get("warmMs", sample.get("warmDistinctLinesMs"))]
            first = sample.get("firstQueryMs", sample.get("firstSliceQueryMs")) / 1000
            ax.scatter(warm, [row + ((i % 5) - 2) * 0.055 for i in range(len(warm))],
                       s=22, color="#156b91", alpha=0.75,
                       label="Subsequent input" if row == 0 else None)
            ax.scatter([first], [row], marker="D", s=36, facecolors="white",
                       edgecolors="#44505b", label="First browser input" if row == 0 else None)
            ax.scatter([statistics.median(warm)], [row], marker="|", s=220,
                       color="#20272c", linewidths=1.5,
                       label="Subsequent median" if row == 0 else None)
        ax.set_xscale("log")
        ax.set_xlim(0.15, 18)
        ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.set_yticks(range(len(samples)), [labels[s["route"]] for s in samples])
        ax.set_ylim(len(samples)-0.5, -0.5)
        ax.set_xlabel("Input to committed picture (seconds, logarithmic scale)")
        ax.grid(axis="y", visible=False)
        ax.set_title("Browser interaction on the declared fixtures", pad=28)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                  frameon=False, fontsize=9, columnspacing=1)
        fig.savefig(here / "latency.svg", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
