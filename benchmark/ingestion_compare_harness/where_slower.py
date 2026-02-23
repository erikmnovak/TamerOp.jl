#!/usr/bin/env python3
"""
Summarize where tamer-op is slower/faster than multipers from comparison CSV output.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import geometric_mean
import tomllib


def _load_manifest_cases(path: Path) -> dict[str, dict]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    out = {}
    for c in raw.get("cases", []):
        out[str(c["id"])] = c
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Report cases where tamer is slower/faster.")
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("where_slower.csv"))
    args = p.parse_args()

    meta = _load_manifest_cases(args.manifest)
    rows = list(csv.DictReader(args.comparison.open(newline="", encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"No rows in {args.comparison}")

    enriched = []
    warm_ratios = []
    cold_ratios = []
    for r in rows:
        cid = r["case_id"]
        m = meta.get(cid, {})
        warm = float(r["warm_slowdown_tamer_over_multipers"])
        cold = float(r["cold_slowdown_tamer_over_multipers"])
        warm_ratios.append(warm)
        cold_ratios.append(cold)
        enriched.append(
            {
                "case_id": cid,
                "regime": r["regime"],
                "n_points": m.get("n_points", r.get("n_points", "")),
                "ambient_dim": m.get("ambient_dim", r.get("ambient_dim", "")),
                "max_dim": m.get("max_dim", r.get("max_dim", "")),
                "warm_tamer_over_multipers": warm,
                "cold_tamer_over_multipers": cold,
                "warm_status": "slower" if warm > 1.0 else "faster",
                "cold_status": "slower" if cold > 1.0 else "faster",
            }
        )

    enriched.sort(key=lambda x: x["warm_tamer_over_multipers"], reverse=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        cols = [
            "case_id",
            "regime",
            "n_points",
            "ambient_dim",
            "max_dim",
            "warm_tamer_over_multipers",
            "cold_tamer_over_multipers",
            "warm_status",
            "cold_status",
        ]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for e in enriched:
            w.writerow(e)

    warm_gm = geometric_mean(warm_ratios)
    cold_gm = geometric_mean(cold_ratios)
    print(f"cases={len(enriched)}")
    print(f"warm_geomean_tamer_over_multipers={warm_gm:.6f} (speedup={1.0 / warm_gm:.6f}x)")
    print(f"cold_geomean_tamer_over_multipers={cold_gm:.6f} (speedup={1.0 / cold_gm:.6f}x)")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
