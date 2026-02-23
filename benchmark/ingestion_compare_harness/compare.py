#!/usr/bin/env python3
"""
Compare tamer-op and multipers ingestion results and compute slowdown summaries.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(x: Any) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "":
        return float("nan")
    return float(s)


def _geom_mean(vals: list[float]) -> float:
    good = [v for v in vals if math.isfinite(v) and v > 0.0]
    if not good:
        return float("nan")
    return math.exp(sum(math.log(v) for v in good) / len(good))


def _write_rows(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare tamer-op vs multipers ingestion CSV outputs.")
    p.add_argument("--tamer", type=Path, default=Path(__file__).with_name("results_tamer.csv"))
    p.add_argument("--multipers", type=Path, default=Path(__file__).with_name("results_multipers.csv"))
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("comparison.csv"))
    p.add_argument("--summary_out", type=Path, default=Path(__file__).with_name("comparison_summary.csv"))
    args = p.parse_args()

    t_rows = _read_csv(args.tamer)
    m_rows = _read_csv(args.multipers)
    t_map = {(r["case_id"], r["regime"]): r for r in t_rows}
    m_map = {(r["case_id"], r["regime"]): r for r in m_rows}
    keys = sorted(set(t_map) & set(m_map))
    if not keys:
        raise RuntimeError("No common cases between tamer and multipers result files.")

    joined: list[dict[str, Any]] = []
    by_regime: dict[str, list[float]] = defaultdict(list)
    by_regime_cold: dict[str, list[float]] = defaultdict(list)

    for key in keys:
        tr = t_map[key]
        mr = m_map[key]
        case_id, regime = key
        t_warm = _f(tr.get("warm_median_ms"))
        m_warm = _f(mr.get("warm_median_ms"))
        t_cold = _f(tr.get("cold_ms"))
        m_cold = _f(mr.get("cold_ms"))
        warm_slowdown = t_warm / m_warm if math.isfinite(t_warm) and math.isfinite(m_warm) and m_warm > 0 else float("nan")
        cold_slowdown = t_cold / m_cold if math.isfinite(t_cold) and math.isfinite(m_cold) and m_cold > 0 else float("nan")
        by_regime[regime].append(warm_slowdown)
        by_regime_cold[regime].append(cold_slowdown)

        joined.append(
            {
                "case_id": case_id,
                "regime": regime,
                "n_points": tr.get("n_points", ""),
                "ambient_dim": tr.get("ambient_dim", ""),
                "max_dim": tr.get("max_dim", ""),
                "tamer_cold_ms": tr.get("cold_ms", ""),
                "multipers_cold_ms": mr.get("cold_ms", ""),
                "cold_slowdown_tamer_over_multipers": f"{cold_slowdown:.6f}" if math.isfinite(cold_slowdown) else "",
                "tamer_warm_median_ms": tr.get("warm_median_ms", ""),
                "multipers_warm_median_ms": mr.get("warm_median_ms", ""),
                "warm_slowdown_tamer_over_multipers": f"{warm_slowdown:.6f}" if math.isfinite(warm_slowdown) else "",
                "tamer_simplex_count": tr.get("simplex_count", ""),
                "multipers_simplex_count": mr.get("simplex_count", ""),
            }
        )

    summary: list[dict[str, Any]] = []
    for regime in sorted(by_regime):
        warm_g = _geom_mean(by_regime[regime])
        cold_g = _geom_mean(by_regime_cold[regime])
        summary.append(
            {
                "regime": regime,
                "n_cases": sum(math.isfinite(v) for v in by_regime[regime]),
                "warm_geomean_slowdown_tamer_over_multipers": f"{warm_g:.6f}" if math.isfinite(warm_g) else "",
                "cold_geomean_slowdown_tamer_over_multipers": f"{cold_g:.6f}" if math.isfinite(cold_g) else "",
            }
        )

    _write_rows(
        args.out,
        joined,
        [
            "case_id",
            "regime",
            "n_points",
            "ambient_dim",
            "max_dim",
            "tamer_cold_ms",
            "multipers_cold_ms",
            "cold_slowdown_tamer_over_multipers",
            "tamer_warm_median_ms",
            "multipers_warm_median_ms",
            "warm_slowdown_tamer_over_multipers",
            "tamer_simplex_count",
            "multipers_simplex_count",
        ],
    )
    _write_rows(
        args.summary_out,
        summary,
        [
            "regime",
            "n_cases",
            "warm_geomean_slowdown_tamer_over_multipers",
            "cold_geomean_slowdown_tamer_over_multipers",
        ],
    )

    print(f"Wrote case-level comparison: {args.out}")
    print(f"Wrote summary comparison: {args.summary_out}")
    print("")
    for row in summary:
        print(
            f"{row['regime']:18s} warm_geomean={row['warm_geomean_slowdown_tamer_over_multipers']} "
            f"cold_geomean={row['cold_geomean_slowdown_tamer_over_multipers']} n={row['n_cases']}"
        )


if __name__ == "__main__":
    main()
