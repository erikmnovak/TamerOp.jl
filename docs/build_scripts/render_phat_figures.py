#!/usr/bin/env python3
"""Render PHAT's public summaries with the shared benchmark style; never time code.

Run from any directory with Python and Matplotlib installed. Existing sealed
PHAT v2 files are inputs only; revised figures have their own presentation seal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

DOCS = Path(__file__).resolve().parents[1]
DATA = DOCS / "benchmarks/phat_v2"
STYLE = DOCS / "assets/benchmarks/report.mplstyle"
FAMILIES = {
    "graph": ("Graphs", ["Path forest", "Single cycle", "Disconnected cycles", "Graph with extra edges"]),
    "simplicial": ("Simplicial inputs", ["Complete 2-skeleton", "Triangulated disk", "Joined sphere boundaries", "Complete 3-skeleton"]),
    "cubical": ("Cubical grids", ["Open 2D grid", "Periodic 2D grid", "Open 3D grid", "Periodic 3D grid"]),
    "algebraic": ("Algebraic controls", ["2 transvections/cell", "4 transvections/cell", "8 transvections/cell", "16 transvections/cell"]),
}
TOOLS = {"tamerop": ("TamerOp", "#156b91", "o", "-"),
         "phat": ("PHAT", "#d36935", "s", "--")}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(output: Path, preview: Path | None = None) -> None:
    output = output.resolve()
    if output == DATA.resolve():
        raise ValueError("Use a presentation subdirectory; keep the sealed results intact.")
    output.mkdir(parents=True, exist_ok=True)
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
    # Check every sealed input, including the original figures and data dictionary.
    for line in (DATA / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if digest(DATA / name) != expected:
            raise ValueError(f"Sealed PHAT v2 input changed: {name}")
    results = json.loads((DATA / "results.json").read_text())
    rows = [r for r in json.loads((DATA / "observations.json").read_text())
            if r["phase"] == "combined"]
    keys = {(r["family"], r["variant"], r["size_level"]) for r in rows}
    expected = {(family, variant, level) for family in FAMILIES
                for variant in range(1, 5) for level in range(1, 4)}
    if len(rows) != 48 or keys != expected:
        raise ValueError("The figures require exactly the frozen 48 combined-phase cases.")
    coverage = results["coverage"]
    if coverage["eligible_cases"] != len(rows) or coverage["fixed_cases"] != len(rows):
        raise ValueError("Incomplete coverage requires an explicitly revised profile policy.")
    if any(r[k] <= 0 for r in rows
           for k in ("cells", "tamerop_seconds", "phat_seconds", "phat_over_tamerop")):
        raise ValueError("Log plots require verified positive times, sizes and ratios.")

    outputs = []

    def save(fig, name):
        # Freeze the solved layout before switching between SVG and PNG renderers.
        # Otherwise a second constrained-layout pass can clip preview labels.
        fig.canvas.draw()
        fig.set_layout_engine("none")
        path = output / f"{name}.svg"
        fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
        if preview is not None:
            fig.savefig(preview / f"{name}.png", bbox_inches="tight")
        outputs.append(path)
        plt.close(fig)

    with plt.style.context(STYLE), matplotlib.rc_context({"svg.hashsalt": "tamerop-phat-presentation-v1"}):
        fig, ax = plt.subplots(figsize=(7.8, 4.5), layout="constrained")
        ax.axvspan(1 / 1.10, 1.10, color="#eeeeee", zorder=0)
        ax.axvline(1, color="#44505b", linewidth=1, zorder=1)
        for i, family in enumerate(FAMILIES):
            selected = sorted((r for r in rows if r["family"] == family), key=lambda r: r["case"])
            ax.scatter([r["phat_over_tamerop"] for r in selected],
                       [i + (j - 5.5) * .035 for j in range(12)],
                       s=22, color="#788590", alpha=.65, zorder=2)
            timing = coverage["families"][family]["timing"]
            ratio, (lo, hi) = timing["ratio"], timing["ci95"]
            ax.errorbar(ratio, i, xerr=[[ratio - lo], [hi - ratio]],
                        fmt="D", color=TOOLS["tamerop"][1], capsize=4,
                        markersize=6, zorder=3)
        ax.set(xscale="log", xlabel="PHAT time / TamerOp time (above 1 favors TamerOp)",
               title="Construction plus complete barcode")
        ax.set_yticks(range(4), [family.title() for family in FAMILIES])
        ax.invert_yaxis()
        ax.set_ylim(3.65, -.45)
        ax.grid(False, axis="y")
        ax.xaxis.set_major_locator(FixedLocator([.5, 1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        handles = [Line2D([], [], marker="D", color=TOOLS["tamerop"][1], linestyle="-",
                          label="Family ratio and 95% interval"),
                   Line2D([], [], marker="o", color="#788590", linestyle="",
                          label="Individual case ratio")]
        fig.legend(handles=handles, loc="outside lower center", ncol=2)
        save(fig, "family_ratios")

        for family, (_, variants) in FAMILIES.items():
            fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), layout="constrained")
            for variant, ax in enumerate(axes.flat, 1):
                selected = sorted((r for r in rows if r["family"] == family and r["variant"] == variant),
                                  key=lambda r: r["size_level"])
                for tool, (label, color, marker, linestyle) in TOOLS.items():
                    ax.plot([r["cells"] for r in selected], [r[f"{tool}_seconds"] * 1000 for r in selected],
                            label=label, color=color, marker=marker, linestyle=linestyle)
                ax.set(xscale="log", yscale="log", xlabel="Input cells",
                       ylabel="Construction + query (ms)", title=variants[variant - 1])
            # One tool key for all four panels; the report heading names the family.
            fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
            save(fig, f"{family}_scaling")

        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        end = max(2, max(max(r["phat_over_tamerop"], 1 / r["phat_over_tamerop"]) for r in rows)) * 1.06
        for tool, (label, color, _, linestyle) in TOOLS.items():
            # Preserve the original profile's paired geometric case ratios.
            costs = sorted(max(1, 1 / r["phat_over_tamerop"]) if tool == "tamerop"
                           else max(1, r["phat_over_tamerop"]) for r in rows)
            points = sorted({1.0, *costs})
            x = [.99, *points, end]
            y = [0, *[sum(c <= value for c in costs) / len(rows) for value in points], 1]
            ax.step(x, y, where="post", label=label, color=color, linestyle=linestyle)
        ax.set(xscale="log", ylim=(0, 1.02), xlabel="Factor of the fastest paired-case time",
               ylabel="Fraction of all 48 requests", title="Complete-request performance profile")
        ax.legend(loc="lower right")
        save(fig, "performance_profile")

    record = {
        "revision": "qpa-style-1",
        "scope": "Presentation only; no new timings, cases, weights or statistical estimates.",
        "measurement_candidate": results["candidate"],
        "inputs_sha256": {name: digest(DATA / name) for name in
                          ("results.json", "observations.json", "timings.csv", "provenance.json", "SHA256SUMS")},
        "renderer": "docs/build_scripts/render_phat_figures.py",
        "renderer_sha256": digest(Path(__file__)),
        "style": "docs/assets/benchmarks/report.mplstyle",
        "style_sha256": digest(STYLE),
        "python": platform.python_version(), "matplotlib": matplotlib.__version__,
        "figure_semantics": {
            "scaling": "All 48 combined-phase cases, milliseconds; each series joins one variant's three size levels. Panel scales vary.",
            "ratios": "Unchanged paired geometric case ratios and recorded family 95% intervals; practical band [1/1.10, 1.10].",
            "profile": "Equal weight over all 48 cases, using paired geometric case ratios as in the original profile; actual step jumps.",
        },
        "figures_sha256": {p.name: digest(p) for p in outputs},
    }
    manifest = output / "presentation.json"
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    (output / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in sorted([*outputs, manifest])))
    print(f"Rendered {len(outputs)} figures from the unchanged 48-case PHAT v2 summaries into {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA / "presentation")
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    render(args.output, args.preview_dir)
