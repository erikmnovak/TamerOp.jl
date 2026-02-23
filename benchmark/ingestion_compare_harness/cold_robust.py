#!/usr/bin/env python3
"""
Robust cold-start benchmark driver for tamer-op vs multipers.

Method:
- Run each case in a fresh process (both tools), with reps=1.
- Repeat this process-level cold run several times.
- Aggregate with median cold_ms per tool per case.
- Report a single robust cold slowdown as geometric mean of per-case medians:
    geomean_case( median_cold_tamer / median_cold_multipers )
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any
import tomllib


def _load_cases(manifest_path: Path, regime: str, case_filter: str) -> list[dict[str, Any]]:
    raw = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    cases = raw.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise RuntimeError(f"No cases found in {manifest_path}")
    out = []
    for c in cases:
        cid = str(c["id"])
        creg = str(c["regime"])
        if regime != "all" and creg != regime:
            continue
        if case_filter and cid != case_filter:
            continue
        out.append(c)
    if not out:
        raise RuntimeError(f"No matching cases in {manifest_path} for regime={regime} case={case_filter!r}")
    return out


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = [
            f"Command failed ({proc.returncode}): {' '.join(cmd)}",
            "--- stdout ---",
            proc.stdout[-4000:],
            "--- stderr ---",
            proc.stderr[-4000:],
        ]
        raise RuntimeError("\n".join(msg))


def _read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _geomean(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main() -> None:
    p = argparse.ArgumentParser(description="Robust cold-start benchmark for ingestion harness.")
    p.add_argument("--manifest", type=Path, default=Path(__file__).with_name("fixtures_50k") / "manifest.toml")
    p.add_argument("--regime", type=str, default="claim_matching")
    p.add_argument("--case", type=str, default="")
    p.add_argument("--restarts", type=int, default=3, help="Number of fresh-process cold repeats per case/tool.")
    p.add_argument("--profile", type=str, default="desktop")
    p.add_argument("--raw_out", type=Path, default=Path(__file__).with_name("cold_robust_raw.csv"))
    p.add_argument("--summary_out", type=Path, default=Path(__file__).with_name("cold_robust_summary.csv"))
    args = p.parse_args()

    if args.restarts < 1:
        raise ValueError("--restarts must be >= 1")

    root = Path(__file__).resolve().parents[2]
    harness = Path(__file__).resolve().parent
    run_tamer = harness / "run_tamer.jl"
    run_multipers = harness / "run_multipers.py"
    cases = _load_cases(args.manifest, args.regime, args.case)

    raw_rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, list[float]]] = {}

    for c in cases:
        case_id = str(c["id"])
        grouped[case_id] = {"tamer": [], "multipers": []}
        for rep in range(1, args.restarts + 1):
            with tempfile.TemporaryDirectory(prefix="cold_robust_") as td:
                td_path = Path(td)
                tamer_out = td_path / "tamer.csv"
                multipers_out = td_path / "multipers.csv"

                tcmd = [
                    "julia",
                    "--project=.",
                    str(run_tamer),
                    f"--manifest={args.manifest}",
                    f"--out={tamer_out}",
                    f"--profile={args.profile}",
                    "--reps=1",
                    f"--regime={c['regime']}",
                    f"--case={case_id}",
                ]
                _run_cmd(tcmd, cwd=root)
                trow = _read_single_row(tamer_out)
                tcold = float(trow["cold_ms"])
                grouped[case_id]["tamer"].append(tcold)
                raw_rows.append(
                    {
                        "tool": "tamer_op",
                        "case_id": case_id,
                        "regime": str(c["regime"]),
                        "repeat": rep,
                        "cold_ms": tcold,
                        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    }
                )

                mcmd = [
                    "python",
                    str(run_multipers),
                    "--manifest",
                    str(args.manifest),
                    "--out",
                    str(multipers_out),
                    "--profile",
                    args.profile,
                    "--reps",
                    "1",
                    "--regime",
                    str(c["regime"]),
                    "--case",
                    case_id,
                ]
                _run_cmd(mcmd, cwd=root)
                mrow = _read_single_row(multipers_out)
                mcold = float(mrow["cold_ms"])
                grouped[case_id]["multipers"].append(mcold)
                raw_rows.append(
                    {
                        "tool": "multipers",
                        "case_id": case_id,
                        "regime": str(c["regime"]),
                        "repeat": rep,
                        "cold_ms": mcold,
                        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    }
                )

            print(
                f"{case_id:28s} rep={rep}/{args.restarts} "
                f"tamer={tcold:.3f}ms multipers={mcold:.3f}ms",
                flush=True,
            )

    case_rows: list[dict[str, Any]] = []
    ratios: list[float] = []
    for c in cases:
        case_id = str(c["id"])
        tmed = statistics.median(grouped[case_id]["tamer"])
        mmed = statistics.median(grouped[case_id]["multipers"])
        ratio = tmed / mmed
        ratios.append(ratio)
        case_rows.append(
            {
                "case_id": case_id,
                "regime": str(c["regime"]),
                "n_points": int(c["n_points"]),
                "ambient_dim": int(c["ambient_dim"]),
                "max_dim": int(c["max_dim"]),
                "tamer_cold_median_ms": tmed,
                "multipers_cold_median_ms": mmed,
                "cold_slowdown_tamer_over_multipers": ratio,
            }
        )

    robust = _geomean(ratios)
    summary_rows = [
        {
            "regime": args.regime,
            "n_cases": len(cases),
            "restarts": args.restarts,
            "robust_cold_geomean_slowdown_tamer_over_multipers": robust,
            "robust_cold_geomean_speedup_tamer_over_multipers": 1.0 / robust if robust > 0 else float("nan"),
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    ]

    _write_csv(
        args.raw_out,
        raw_rows,
        ["tool", "case_id", "regime", "repeat", "cold_ms", "timestamp_utc"],
    )
    _write_csv(
        args.summary_out,
        case_rows + [{}] + summary_rows,
        [
            "case_id",
            "regime",
            "n_points",
            "ambient_dim",
            "max_dim",
            "tamer_cold_median_ms",
            "multipers_cold_median_ms",
            "cold_slowdown_tamer_over_multipers",
            "n_cases",
            "restarts",
            "robust_cold_geomean_slowdown_tamer_over_multipers",
            "robust_cold_geomean_speedup_tamer_over_multipers",
            "timestamp_utc",
        ],
    )

    print("", flush=True)
    print(f"Wrote raw cold repeats: {args.raw_out}", flush=True)
    print(f"Wrote robust cold summary: {args.summary_out}", flush=True)
    print(f"Robust cold slowdown (tamer/multipers): {robust:.6f}", flush=True)
    print(f"Robust cold speedup (tamer over multipers): {1.0 / robust:.6f}x", flush=True)


if __name__ == "__main__":
    main()
