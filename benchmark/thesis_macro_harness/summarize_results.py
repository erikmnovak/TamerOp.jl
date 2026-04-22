#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _parse_float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return None
    return float(raw)


def _parse_int(row: dict[str, str], key: str) -> int | None:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return None
    return int(raw)


def _median(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return statistics.median(vals)


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return statistics.mean(vals)


def _fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as fh:
        return list(csv.DictReader(fh))


def _group_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("source_kind", ""),
            row.get("family", ""),
            row.get("family_case", ""),
            row.get("invariant_kind", ""),
            row.get("degree_label", ""),
            row.get("size_tier", ""),
            row.get("backend_label", ""),
        )
        groups[key].append(row)

    summary_rows: list[dict[str, object]] = []
    for key in sorted(groups):
        bucket = groups[key]
        ok_rows = [row for row in bucket if row.get("status") == "ok"]
        err_rows = [row for row in bucket if row.get("status") != "ok"]
        summary_rows.append(
            {
                "source_kind": key[0],
                "family": key[1],
                "family_case": key[2],
                "invariant_kind": key[3],
                "degree_label": key[4],
                "size_tier": key[5],
                "backend_label": key[6],
                "job_count": len(bucket),
                "ok_count": len(ok_rows),
                "error_count": len(err_rows),
                "cold_median_ms": _median(_parse_float(row, "cold_ms") for row in ok_rows if _parse_float(row, "cold_ms") is not None),
                "warm_median_ms": _median(_parse_float(row, "warm_median_ms") for row in ok_rows if _parse_float(row, "warm_median_ms") is not None),
                "warm_mean_ms": _mean(_parse_float(row, "warm_median_ms") for row in ok_rows if _parse_float(row, "warm_median_ms") is not None),
                "source_size_median": _median(_parse_int(row, "source_size") for row in ok_rows if _parse_int(row, "source_size") is not None),
                "encoding_vertices_median": _median(_parse_int(row, "encoding_vertices") for row in ok_rows if _parse_int(row, "encoding_vertices") is not None),
                "output_terms_median": _median(_parse_int(row, "output_term_count") for row in ok_rows if _parse_int(row, "output_term_count") is not None),
                "exact_rows": sum(1 for row in ok_rows if row.get("exact_supported") == "true"),
                "status": "ok" if ok_rows and not err_rows else ("mixed" if ok_rows else "error"),
            }
        )
    return summary_rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key, value in list(out.items()):
                if isinstance(value, float):
                    out[key] = _fmt_float(value)
            writer.writerow(out)


def _markdown_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "source_kind",
        "family",
        "invariant_kind",
        "degree_label",
        "size_tier",
        "backend_label",
        "job_count",
        "status",
        "cold_median_ms",
        "warm_median_ms",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        vals = []
        for key in headers:
            value = row.get(key, "")
            if isinstance(value, float):
                vals.append(_fmt_float(value))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _latex_table(rows: list[dict[str, object]]) -> str:
    headers = [
        ("source_kind", "Source"),
        ("family", "Family"),
        ("invariant_kind", "Invariant"),
        ("degree_label", "Degree"),
        ("size_tier", "Size"),
        ("backend_label", "Backend"),
        ("job_count", "Jobs"),
        ("status", "Status"),
        ("cold_median_ms", "Cold ms"),
        ("warm_median_ms", "Warm ms"),
    ]
    body = ["\\begin{tabular}{llllllllll}", "\\hline", " & ".join(label for _, label in headers) + r" \\", "\\hline"]
    for row in rows:
        vals = []
        for key, _ in headers:
            value = row.get(key, "")
            if isinstance(value, float):
                vals.append(_fmt_float(value))
            else:
                vals.append(str(value).replace("_", r"\_"))
        body.append(" & ".join(vals) + r" \\")
    body.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(body) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--summary_out", required=True)
    parser.add_argument("--markdown_out", required=True)
    parser.add_argument("--latex_out", required=True)
    parser.add_argument("--errors_out", required=False)
    args = parser.parse_args()

    rows = _load_rows(Path(args.results))
    summary_rows = _group_rows(rows)
    fieldnames = [
        "source_kind",
        "family",
        "family_case",
        "invariant_kind",
        "degree_label",
        "size_tier",
        "backend_label",
        "job_count",
        "ok_count",
        "error_count",
        "exact_rows",
        "status",
        "cold_median_ms",
        "warm_median_ms",
        "warm_mean_ms",
        "source_size_median",
        "encoding_vertices_median",
        "output_terms_median",
    ]
    _write_csv(Path(args.summary_out), summary_rows, fieldnames)

    ok_or_mixed = [row for row in summary_rows if row["status"] != "error"]
    Path(args.markdown_out).write_text(_markdown_table(ok_or_mixed))
    Path(args.latex_out).write_text(_latex_table(ok_or_mixed))

    if args.errors_out:
        error_rows = [row for row in rows if row.get("status") != "ok"]
        if error_rows:
            error_fields = list(error_rows[0].keys())
        else:
            error_fields = ["job_id", "status", "error_stage", "error_type", "error_message"]
        _write_csv(Path(args.errors_out), error_rows, error_fields)


if __name__ == "__main__":
    main()
