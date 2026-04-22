#!/usr/bin/env python3
"""
Run end-to-end invariant benchmarks for multipers on deterministic fixtures.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import itertools
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
import tomllib

import numpy as np


def _profile_defaults(profile: str) -> dict[str, int | bool]:
    profile = profile.lower()
    if profile == "desktop":
        return {"reps": 4, "trim_between_reps": True, "trim_between_cases": True}
    if profile == "balanced":
        return {"reps": 5, "trim_between_reps": False, "trim_between_cases": True}
    if profile == "stress":
        return {"reps": 9, "trim_between_reps": False, "trim_between_cases": False}
    if profile == "probe":
        return {"reps": 3, "trim_between_reps": False, "trim_between_cases": True}
    raise ValueError("--profile must be one of: desktop, balanced, stress, probe")


def _memory_relief() -> None:
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _try_import_multipers():
    try:
        import gudhi as gd  # type: ignore
        import multipers as mp  # type: ignore
        import multipers.filtrations as mf  # type: ignore
        import multipers.ml.point_clouds as mmp  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "multipers is not available. Install it in the active Python environment."
        ) from exc
    return mmp, mp, gd, mf


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    cases = raw.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No cases found in {path}")
    return raw


def _load_case_data(case: dict[str, Any], fixture: Path) -> np.ndarray:
    dataset = str(case["dataset"])
    data = np.loadtxt(fixture, delimiter=",", dtype=np.float64)
    if dataset == "gaussian_shell":
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data
    if dataset == "image_sine":
        if data.ndim != 2:
            raise RuntimeError(f"image_sine fixture must be 2D matrix, got shape={data.shape}.")
        return data
    raise RuntimeError(f"Unsupported dataset: {dataset}")


def _build_multipers_filtered_complex(mmp, mp, gd, mf, data: np.ndarray, case: dict[str, Any]):
    regime = str(case["regime"])
    max_dim = int(case["max_dim"])
    points = data
    if regime == "degree_rips_parity":
        degree_radius = float(case["degree_radius"])
        try:
            st = mf.DegreeRips(points=points, threshold_radius=degree_radius)
            return st, {"builder": "filtrations.DegreeRips", "threshold_radius": degree_radius, "max_dim": max_dim}
        except NotImplementedError as exc:
            raise RuntimeError(
                "filtrations.DegreeRips is unavailable in the local multipers build; "
                "lower-star emulation is disabled for this parity benchmark."
            ) from exc
    if regime == "rips_parity":
        if "parity_radius" in case:
            parity_radius = float(case["parity_radius"])
            st0 = gd.RipsComplex(points=points, max_edge_length=parity_radius).create_simplex_tree(
                max_dimension=max_dim
            )
            st = mp.simplex_tree_multi.SimplexTreeMulti(st0, num_parameters=1, safe_conversion=False)
            return st, {
                "builder": "gudhi_rips_radius",
                "radius": parity_radius,
                "max_dim": max_dim,
            }
        common = {
            "complex": "rips",
            "output_type": "simplextree",
            "expand_dim": max_dim,
        }
        tries = [
            {
                **common,
                "num_collapses": 0,
            },
            {
                **common,
            },
        ]
        last_err: Exception | None = None
        for kw in tries:
            try:
                est = mmp.PointCloud2FilteredComplex(**kw)
                out = est.fit_transform([points])
                if len(out) == 0 or len(out[0]) == 0:
                    raise RuntimeError("PointCloud2FilteredComplex returned empty output.")
                st = out[0][0]
                return st, kw
            except Exception as exc:
                last_err = exc
        assert last_err is not None
        raise RuntimeError("All Rips constructor attempts failed.") from last_err
    if regime == "rips_codensity_parity":
        codensity_radius = float(case["codensity_radius"])
        dtm_mass_val = float(case["codensity_dtm_mass"])
        st = mf.RipsCodensity(points, dtm_mass=dtm_mass_val, threshold_radius=codensity_radius)
        return st, {
            "builder": "filtrations.RipsCodensity",
            "threshold_radius": codensity_radius,
            "dtm_mass": dtm_mass_val,
            "max_dim": max_dim,
        }
    if regime == "rips_lowerstar_parity":
        lowerstar_radius = float(case["lowerstar_radius"])
        fvals = np.asarray(points[:, 0], dtype=np.float64)
        st = mf.RipsLowerstar(points=points, function=fvals, threshold_radius=lowerstar_radius)
        return st, {
            "builder": "filtrations.RipsLowerstar",
            "threshold_radius": lowerstar_radius,
            "function": "coord1",
            "max_dim": max_dim,
        }
    if regime == "landmark_parity":
        if "landmarks" not in case:
            raise RuntimeError("landmark_parity case requires landmarks in manifest.")
        if "landmark_radius" not in case:
            raise RuntimeError("landmark_parity case requires landmark_radius in manifest.")
        landmarks_1 = np.asarray(case["landmarks"], dtype=np.int64)
        if landmarks_1.size == 0:
            raise RuntimeError("landmark_parity landmarks cannot be empty.")
        idx0 = landmarks_1 - 1
        if idx0.min() < 0 or idx0.max() >= points.shape[0]:
            raise RuntimeError("landmark_parity landmarks are out of bounds for fixture point cloud.")
        sub = points[idx0, :]
        radius = float(case["landmark_radius"])
        st0 = gd.RipsComplex(points=sub, max_edge_length=radius).create_simplex_tree(max_dimension=max_dim)
        st = mp.simplex_tree_multi.SimplexTreeMulti(st0, num_parameters=1, safe_conversion=False)
        return st, {
            "builder": "gudhi_landmark_rips_radius",
            "radius": radius,
            "max_dim": max_dim,
            "n_landmarks": int(sub.shape[0]),
        }
    if regime == "delaunay_lowerstar_parity":
        fvals = np.asarray(points[:, 0], dtype=np.float64)
        try:
            st = mf.DelaunayLowerstar(points, fvals, verbose=False)
        except (AssertionError, NotImplementedError) as exc:
            msg = str(exc)
            if "function_delaunay" in msg or "DelaunayLowerstar" in msg:
                raise RuntimeError(
                    "filtrations.DelaunayLowerstar is unavailable in the local multipers build; "
                    "Delaunay lower-star invariant parity is disabled."
                ) from exc
            raise
        return st, {
            "builder": "filtrations.DelaunayLowerstar",
            "function": "coord1",
            "max_dim": max_dim,
        }
    if regime == "alpha_parity":
        tries = [
            {
                "complex": "alpha",
                "output_type": "slicer_novine",
                "expand_dim": max_dim,
                "num_collapses": 0,
                "safe_conversion": False,
            },
            {
                "complex": "alpha",
                "output_type": "slicer",
                "expand_dim": max_dim,
                "num_collapses": 0,
                "safe_conversion": False,
            },
        ]
        last_err: Exception | None = None
        for kw in tries:
            try:
                est = mmp.PointCloud2FilteredComplex(**kw)
                out = est.fit_transform([points])
                if len(out) == 0 or len(out[0]) == 0:
                    raise RuntimeError("PointCloud2FilteredComplex returned empty output.")
                st = out[0][0]
                return st, kw
            except Exception as exc:
                last_err = exc
        assert last_err is not None
        raise RuntimeError("All alpha constructor attempts failed.") from last_err
    if regime == "core_delaunay_parity":
        ks = case.get("core_ks")
        ks_vals = None if ks is None else [int(v) for v in ks]
        st = mf.CoreDelaunay(points, ks=ks_vals, precision="safe", verbose=False)
        st.prune_above_dimension(max_dim)
        return st, {"builder": "filtrations.CoreDelaunay", "max_dim": max_dim, "ks": ks_vals}
    if regime == "cubical_parity":
        image = data
        if image.ndim != 2:
            raise RuntimeError(f"cubical_parity expects 2D image fixture, got shape={image.shape}")
        slicer = mf.Cubical(image[:, :, None])
        return slicer, {"builder": "filtrations.Cubical", "image_shape": tuple(int(v) for v in image.shape)}
    raise RuntimeError(f"Unsupported regime for invariant benchmark: {regime}")


def _parse_invariants(raw: str) -> list[str]:
    s = raw.strip().lower()
    if s in ("", "all"):
        return []
    out: list[str] = []
    for part in s.split(","):
        token = part.strip().lower().replace("-", "_")
        if token in ("rank", "rank_signed_measure"):
            out.append("rank_signed_measure")
        elif token == "rank_invariant":
            out.append("rank_invariant")
        elif token in ("restricted_hilbert", "hilbert"):
            out.append("restricted_hilbert")
        elif token in ("slice", "slice_barcodes"):
            out.append("slice_barcodes")
        elif token in ("landscape", "mp_landscape"):
            out.append("mp_landscape")
        elif token in ("euler", "euler_signed_measure"):
            out.append("euler_signed_measure")
        else:
            raise ValueError(f"--invariants contains unsupported token {part}")
    if not out:
        raise ValueError("--invariants selected no invariants.")
    return list(dict.fromkeys(out))


def _approved_invariants_for_regime(manifest: dict[str, Any], regime: str) -> list[str] | None:
    tbl = manifest.get("invariant_eligibility")
    if tbl is None:
        return None
    if not isinstance(tbl, dict):
        raise ValueError("manifest invariant_eligibility must be a table.")
    vals = tbl.get(regime)
    if vals is None:
        return None
    if not isinstance(vals, list):
        raise ValueError(f"manifest invariant_eligibility[{regime!r}] must be an array.")
    out: list[str] = []
    for v in vals:
        token = str(v).strip().lower().replace("-", "_")
        if token in ("rank", "rank_signed_measure"):
            out.append("rank_signed_measure")
        elif token == "rank_invariant":
            out.append("rank_invariant")
        elif token in ("restricted_hilbert", "hilbert"):
            out.append("restricted_hilbert")
        elif token in ("slice", "slice_barcodes"):
            out.append("slice_barcodes")
        elif token in ("landscape", "mp_landscape"):
            out.append("mp_landscape")
        elif token in ("euler", "euler_signed_measure"):
            out.append("euler_signed_measure")
        else:
            raise ValueError(f"Unsupported manifest-approved invariant token {v!r} for regime {regime!r}")
    return list(dict.fromkeys(out))


def _selected_invariants_for_case(requested: list[str], approved: list[str] | None) -> list[str]:
    requested_all = len(requested) == 0
    if approved is None:
        return ["euler_signed_measure"] if requested_all else requested
    if requested_all:
        return approved
    approved_set = set(approved)
    return [inv for inv in requested if inv in approved_set]


def _weight_token(weight: Any) -> str:
    if isinstance(weight, (int, np.integer)):
        return str(int(weight))
    if isinstance(weight, (float, np.floating)):
        f = float(weight)
        if math.isfinite(f) and f.is_integer():
            return str(int(round(f)))
        return format(f, ".17g")
    return str(weight)


def _coord_token(x: float) -> str:
    return format(float(x), ".17g")


def _coord_key_tuple(coords: tuple[float, ...] | list[float] | np.ndarray) -> str:
    return "|".join(_coord_token(float(x)) for x in coords)


def _needs_hard_exit_rank_workaround(regime: str, invariant_kind: str) -> bool:
    return regime == "cubical_parity" and invariant_kind in ("rank_signed_measure", "rank_invariant")


def _normalize_hilbert_measure_contract(
    axes: tuple[np.ndarray, ...] | list[np.ndarray],
    res,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, np.ndarray]]:
    raw_axes = tuple(np.asarray(axis, dtype=np.float64) for axis in axes)
    locs, weights = _extract_measure_payload(res)
    locs_arr = np.asarray(locs, dtype=np.float64)
    weights_arr = np.asarray(weights)
    ndim = len(raw_axes)
    if weights_arr.size == 0:
        if locs_arr.ndim == 1:
            locs_arr = locs_arr.reshape(0, ndim)
        elif locs_arr.ndim != 2:
            locs_arr = np.reshape(locs_arr, (0, ndim))
    elif locs_arr.ndim == 1:
        locs_arr = locs_arr.reshape(1, -1) if weights_arr.size == 1 else locs_arr.reshape(-1, 1)
    elif locs_arr.ndim != 2:
        locs_arr = np.reshape(locs_arr, (weights_arr.size, -1))
    if ndim == 0:
        return raw_axes, (locs_arr, weights_arr)
    if locs_arr.ndim != 2 or (locs_arr.shape[1] not in (0, ndim)):
        raise RuntimeError(
            f"Hilbert signed measure must encode {ndim} coordinates, got shape={locs_arr.shape}."
        )
    if locs_arr.shape[0] != weights_arr.size:
        raise RuntimeError(
            f"Signed-measure location/weight shape mismatch: locs={locs_arr.shape}, weights={weights_arr.shape}"
        )
    if locs_arr.shape[1] == 0:
        return raw_axes, (locs_arr.reshape(0, ndim), weights_arr)

    keep_dims: list[int] = []
    for dim, axis in enumerate(raw_axes):
        drop_dummy = axis.size == 1 and np.isneginf(axis[0]) and (
            weights_arr.size == 0 or np.all(locs_arr[:, dim] == axis[0])
        )
        if not drop_dummy:
            keep_dims.append(dim)
    if len(keep_dims) == ndim or not keep_dims:
        return raw_axes, (locs_arr, weights_arr)

    norm_axes = tuple(raw_axes[dim] for dim in keep_dims)
    norm_locs = locs_arr[:, keep_dims]
    return norm_axes, (norm_locs, weights_arr)


def _integrate_measure_on_axes(
    mp,
    axes: tuple[np.ndarray, ...],
    res,
) -> np.ndarray:
    locs, weights = _extract_measure_payload(res)
    return np.asarray(
        mp.point_measure.integrate_measure(
            locs,
            weights,
            filtration_grid=axes,
            return_grid=False,
        ),
        dtype=np.float64,
    )


def _sparse_hilbert_terms_from_measure(
    axes: tuple[np.ndarray, ...] | list[np.ndarray],
    res,
) -> list[tuple[tuple[float, ...], str]] | None:
    hilbert_axes, hilbert_measure = _normalize_hilbert_measure_contract(axes, res)
    if len(hilbert_axes) != 1:
        return None
    locs, weights = hilbert_measure
    locs_arr = np.asarray(locs, dtype=np.float64)
    weights_arr = np.asarray(weights)
    if weights_arr.size == 0:
        return []
    if locs_arr.ndim != 2 or locs_arr.shape[1] != 1:
        raise RuntimeError(f"Hilbert signed measure must be 1D after normalization, got shape={locs_arr.shape}.")
    xs = locs_arr[:, 0]
    out: list[tuple[tuple[float, ...], str]] = []
    for q in np.asarray(hilbert_axes[0], dtype=np.float64).tolist():
        idx = xs <= float(q)
        if not np.any(idx):
            continue
        val = weights_arr[idx].sum()
        if isinstance(val, np.generic):
            val = val.item()
        if val == 0:
            continue
        out.append(((float(q),), _weight_token(val)))
    return out


def _sparse_function_terms_from_values(
    axes: tuple[np.ndarray, ...] | list[np.ndarray],
    values: np.ndarray,
) -> list[tuple[tuple[float, ...], str]]:
    axes0 = tuple(np.asarray(axis, dtype=np.float64) for axis in axes)
    arr = np.asarray(values)
    if len(axes0) == 0:
        return []
    expected_shape = tuple(axis.size for axis in axes0)
    if arr.shape != expected_shape:
        raise RuntimeError(f"Function table shape mismatch: values={arr.shape}, expected={expected_shape}")
    nz = np.argwhere(arr != 0)
    out: list[tuple[tuple[float, ...], str]] = []
    for idx in nz.tolist():
        coords = tuple(float(axes0[dim][i]) for dim, i in enumerate(idx))
        out.append((coords, _weight_token(arr[tuple(idx)].item())))
    out.sort(key=lambda item: item[0])
    return out


def _normalize_rank_measure_contract(
    axes: tuple[np.ndarray, ...] | list[np.ndarray],
    res,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, np.ndarray]]:
    raw_axes = tuple(np.asarray(axis, dtype=np.float64) for axis in axes)
    locs, weights = _extract_measure_payload(res)
    locs_arr = np.asarray(locs, dtype=np.float64)
    weights_arr = np.asarray(weights)
    if weights_arr.size == 0:
        if locs_arr.ndim == 1:
            locs_arr = locs_arr.reshape(0, len(raw_axes) * 2)
        elif locs_arr.ndim != 2:
            locs_arr = np.reshape(locs_arr, (0, len(raw_axes) * 2))
    elif locs_arr.ndim == 1:
        locs_arr = locs_arr.reshape(1, -1) if weights_arr.size == 1 else locs_arr.reshape(-1, 1)
    elif locs_arr.ndim != 2:
        locs_arr = np.reshape(locs_arr, (weights_arr.size, -1))
    ndim = len(raw_axes)
    if ndim == 0:
        return raw_axes, (locs_arr, weights_arr)
    if locs_arr.ndim != 2 or (locs_arr.shape[1] not in (0, 2 * ndim)):
        raise RuntimeError(
            f"Rank signed measure must encode {ndim} birth/death coordinates, got shape={locs_arr.shape}."
        )
    if locs_arr.shape[0] != weights_arr.size:
        raise RuntimeError(
            f"Signed-measure location/weight shape mismatch: locs={locs_arr.shape}, weights={weights_arr.shape}"
        )
    if locs_arr.shape[1] == 0:
        return raw_axes, (locs_arr.reshape(0, 2 * ndim), weights_arr)

    births = locs_arr[:, :ndim]
    deaths = locs_arr[:, ndim:]
    keep_dims: list[int] = []
    for dim, axis in enumerate(raw_axes):
        drop_dummy = (
            axis.size == 1
            and np.isneginf(axis[0])
            and (
                weights_arr.size == 0
                or (
                    np.all(births[:, dim] == axis[0])
                    and np.all(np.isposinf(deaths[:, dim]))
                )
            )
        )
        if not drop_dummy:
            keep_dims.append(dim)
    if len(keep_dims) == ndim or not keep_dims:
        return raw_axes, (locs_arr, weights_arr)

    norm_axes = tuple(raw_axes[dim] for dim in keep_dims)
    norm_locs = np.concatenate([births[:, keep_dims], deaths[:, keep_dims]], axis=1)
    return norm_axes, (norm_locs, weights_arr)


def _sparse_rank_terms_from_measure(
    axes: tuple[np.ndarray, ...] | list[np.ndarray],
    res,
) -> list[tuple[tuple[float, ...], str]]:
    rank_axes, rank_measure = _normalize_rank_measure_contract(axes, res)
    ndim = len(rank_axes)
    if ndim == 0:
        return []
    locs, weights = rank_measure
    if weights.size == 0:
        return []
    locs_arr = np.asarray(locs, dtype=np.float64)
    weights_arr = np.asarray(weights)
    births = locs_arr[:, :ndim]
    deaths = locs_arr[:, ndim:]
    grid_points = [
        np.asarray(point, dtype=np.float64)
        for point in itertools.product(*(axis.tolist() for axis in rank_axes))
    ]
    out: list[tuple[tuple[float, ...], str]] = []
    for birth_point in grid_points:
        birth_ok = np.all(births <= birth_point[None, :], axis=1)
        if not np.any(birth_ok):
            continue
        for death_point in grid_points:
            if np.any(birth_point > death_point):
                continue
            idx = birth_ok & np.all(deaths > death_point[None, :], axis=1)
            if not np.any(idx):
                continue
            val = weights_arr[idx].sum()
            if isinstance(val, np.generic):
                val = val.item()
            if val == 0:
                continue
            coords = tuple(float(x) for x in birth_point.tolist()) + tuple(
                float(x) for x in death_point.tolist()
            )
            out.append((coords, _weight_token(val)))
    out.sort(key=lambda item: item[0])
    return out


def _slice_query_from_case(case: dict[str, Any]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if "slice_directions" not in case or "slice_offsets" not in case:
        raise RuntimeError("slice_barcodes requires slice_directions and slice_offsets in the manifest.")
    directions = [np.asarray(row, dtype=np.float64) for row in case["slice_directions"]]
    offsets = [np.asarray(row, dtype=np.float64) for row in case["slice_offsets"]]
    if not directions or not offsets:
        raise RuntimeError("slice_barcodes requires nonempty slice_directions and slice_offsets.")
    return directions, offsets


def _mp_landscape_query_from_case(case: dict[str, Any]) -> tuple[int, np.ndarray]:
    if "mp_kmax" not in case or "mp_tgrid" not in case:
        raise RuntimeError("mp_landscape requires mp_kmax and mp_tgrid in the manifest.")
    kmax = int(case["mp_kmax"])
    tgrid = np.asarray(case["mp_tgrid"], dtype=np.float64)
    if kmax <= 0:
        raise RuntimeError("mp_landscape requires mp_kmax > 0.")
    if tgrid.ndim != 1 or tgrid.size < 2:
        raise RuntimeError("mp_landscape requires a 1D mp_tgrid with at least two entries.")
    return kmax, tgrid


def _canonical_barcode_terms(intervals: np.ndarray) -> list[tuple[tuple[float, float], int]]:
    arr = np.asarray(intervals, dtype=np.float64)
    if arr.size == 0:
        return []
    arr = arr.reshape(-1, 2)
    agg: dict[tuple[float, float], int] = {}
    for birth, death in arr.tolist():
        key = (float(birth), float(death))
        agg[key] = agg.get(key, 0) + 1
    out: list[tuple[tuple[float, float], int]] = []
    for key in sorted(agg, key=lambda t: (_coord_token(t[0]), _coord_token(t[1]))):
        mult = agg[key]
        if mult == 0:
            continue
        out.append((key, mult))
    return out


def _serialize_barcode_terms(terms: list[tuple[tuple[float, float], int]]) -> str:
    return ";".join(
        _coord_token(interval[0]) + "|" + _coord_token(interval[1]) + "=>" + _weight_token(mult)
        for interval, mult in terms
    )


def _slice_output_contract(res: dict[str, Any]) -> tuple[int, float, str]:
    records: list[str] = []
    term_count = 0
    abs_mass = 0.0
    directions = res["directions"]
    offsets = res["offsets"]
    barcodes = res["barcodes"]
    for i, direction in enumerate(directions):
        for j, offset in enumerate(offsets):
            terms = _canonical_barcode_terms(barcodes[i][j])
            term_count += len(terms)
            abs_mass += sum(abs(float(mult)) for _, mult in terms)
            records.append(
                "dir="
                + _coord_key_tuple(tuple(direction.tolist()))
                + "@@off="
                + _coord_key_tuple(tuple(offset.tolist()))
                + "@@bars="
                + _serialize_barcode_terms(terms)
            )
    return term_count, abs_mass, "###".join(records)


def _landscape_values_from_intervals(intervals: np.ndarray, tgrid: np.ndarray, kmax: int) -> np.ndarray:
    arr = np.asarray(intervals, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((kmax, tgrid.size), dtype=np.float64)
    arr = arr.reshape(-1, 2)
    arr = arr[arr[:, 0] < arr[:, 1]]
    if arr.size == 0:
        return np.zeros((kmax, tgrid.size), dtype=np.float64)

    births = arr[:, 0][:, None]
    deaths = arr[:, 1][:, None]
    tg = np.asarray(tgrid, dtype=np.float64)[None, :]
    tents = np.minimum(tg - births, deaths - tg)
    tents[tents < 0.0] = 0.0

    if tents.shape[0] <= kmax:
        top = np.sort(tents, axis=0)[::-1]
        out = np.zeros((kmax, tgrid.size), dtype=np.float64)
        out[: top.shape[0], :] = top
        return out

    idx = np.argpartition(tents, -kmax, axis=0)[-kmax:, :]
    top = np.take_along_axis(tents, idx, axis=0)
    top.sort(axis=0)
    return top[::-1, :]


def _serialize_landscape_values(vals: np.ndarray) -> str:
    rows: list[str] = []
    arr = np.asarray(vals, dtype=np.float64)
    for k in range(arr.shape[0]):
        rows.append("|".join(_coord_token(float(x)) for x in arr[k, :].tolist()))
    return ";;".join(rows)


def _mp_landscape_output_contract(res: dict[str, Any]) -> tuple[int, float, str]:
    kmax = int(res["kmax"])
    tgrid = np.asarray(res["tgrid"], dtype=np.float64)
    values = np.asarray(res["values"], dtype=np.float64)
    weights = np.asarray(res["weights"], dtype=np.float64)
    directions = res["directions"]
    offsets = res["offsets"]
    records: list[str] = []
    abs_mass = float(np.sum(np.abs(values)))
    for i, direction in enumerate(directions):
        for j, offset in enumerate(offsets):
            vals = values[i, j, :, :]
            records.append(
                "dir="
                + _coord_key_tuple(tuple(direction.tolist()))
                + "@@off="
                + _coord_key_tuple(tuple(offset.tolist()))
                + "@@w="
                + _coord_token(float(weights[i, j]))
                + "@@vals="
                + _serialize_landscape_values(vals)
            )
    payload = "kmax=" + str(kmax) + "@@tgrid=" + _coord_key_tuple(tuple(tgrid.tolist())) + "###" + "###".join(records)
    return int(values.size), abs_mass, payload


def _extract_measure_payload(res):
    if isinstance(res, tuple):
        if len(res) == 0:
            return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.float64)
        if len(res) == 2 and not isinstance(res[0], tuple):
            return res
        return res[0]
    if isinstance(res, list):
        if len(res) == 0:
            return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.float64)
        return res[0]
    raise TypeError(f"Unsupported signed-measure payload type: {type(res)!r}")


def _canonical_measure_terms(res) -> list[tuple[tuple[float, ...], str]]:
    locs, weights = _extract_measure_payload(res)
    weights_arr = np.asarray(weights)
    if weights_arr.size == 0:
        return []
    locs_arr = np.asarray(locs, dtype=np.float64)
    if locs_arr.ndim == 1:
        if weights_arr.size == 1:
            locs_arr = locs_arr.reshape(1, -1)
        else:
            locs_arr = locs_arr.reshape(-1, 1)
    elif locs_arr.ndim != 2:
        locs_arr = np.reshape(locs_arr, (weights_arr.size, -1))
    if locs_arr.shape[0] != weights_arr.size:
        raise RuntimeError(
            f"Signed-measure location/weight shape mismatch: locs={locs_arr.shape}, weights={weights_arr.shape}"
        )
    agg: dict[tuple[float, ...], Any] = {}
    for row, weight in zip(locs_arr.tolist(), weights_arr.tolist(), strict=True):
        key = tuple(float(x) for x in row)
        agg[key] = agg.get(key, 0) + weight
    out: list[tuple[tuple[float, ...], str]] = []
    for key in sorted(agg):
        weight = agg[key]
        if weight == 0:
            continue
        out.append((key, _weight_token(weight)))
    return out


def _serialize_measure_terms(terms: list[tuple[tuple[float, ...], str]]) -> str:
    return ";".join("|".join(_coord_token(x) for x in coords) + "=>" + weight for coords, weight in terms)


def _timed_call(f):
    t0 = time.perf_counter()
    out = f()
    t1 = time.perf_counter()
    return out, 1000.0 * (t1 - t0)


def _p90(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = max(0, int(np.ceil(0.9 * len(s))) - 1)
    return s[idx]


def _native_rank_query_axes(filtered_complex, mp) -> tuple[np.ndarray, ...]:
    if getattr(filtered_complex, "is_squeezed", False):
        grid = getattr(filtered_complex, "filtration_grid", None)
        if grid is None or len(grid) == 0:
            raise RuntimeError("Squeezed multipers complex is missing filtration_grid for rank query contract.")
        return tuple(np.asarray(axis, dtype=np.float64) for axis in grid)
    grid = mp.grids.compute_grid(filtered_complex, strategy="exact")
    return tuple(np.asarray(axis, dtype=np.float64) for axis in grid)


def _signed_measure_target_complex(filtered_complex, mp, invariant_kind: str):
    if invariant_kind not in ("rank_invariant", "restricted_hilbert"):
        return filtered_complex
    if hasattr(filtered_complex, "persistence_on_line"):
        return filtered_complex
    return mp.Slicer(filtered_complex)


def _hilbert_query_axes_from_case(case: dict[str, Any]) -> tuple[np.ndarray, ...] | None:
    raw_axes = case.get("hilbert_query_axes")
    if raw_axes is None:
        return None
    if not isinstance(raw_axes, list) or not raw_axes:
        raise RuntimeError("hilbert_query_axes must be a nonempty array of coordinate arrays.")
    return tuple(np.asarray(axis, dtype=np.float64) for axis in raw_axes)


def _expand_hilbert_query_axes_for_target(
    canonical_axes: tuple[np.ndarray, ...],
    target,
    mp,
) -> tuple[np.ndarray, ...]:
    native_axes = _native_rank_query_axes(target, mp)
    if len(canonical_axes) == len(native_axes):
        return canonical_axes
    expanded: list[np.ndarray] = []
    canon_idx = 0
    for native_axis in native_axes:
        is_dummy = native_axis.size == 1 and np.isneginf(native_axis[0])
        if is_dummy:
            expanded.append(np.asarray(native_axis, dtype=np.float64))
            continue
        if canon_idx >= len(canonical_axes):
            raise RuntimeError(
                f"hilbert_query_axes dimension mismatch: canonical={len(canonical_axes)} native={len(native_axes)}"
            )
        expanded.append(np.asarray(canonical_axes[canon_idx], dtype=np.float64))
        canon_idx += 1
    if canon_idx != len(canonical_axes):
        raise RuntimeError(
            f"Unused hilbert_query_axes dimensions: canonical={len(canonical_axes)} native={len(native_axes)}"
        )
    return tuple(expanded)


def _run_invariant_uncached(mmp, mp, gd, mf, data: np.ndarray, case: dict[str, Any], invariant_kind: str, degree: int):
    filtered_complex, meta = _build_multipers_filtered_complex(mmp, mp, gd, mf, data, case)
    if (
        invariant_kind == "euler_signed_measure"
        and str(case["regime"]) == "core_delaunay_parity"
        and getattr(filtered_complex, "is_kcritical", False)
    ):
        raise RuntimeError(
            "multipers signed_measure does not support Euler on CoreDelaunay k-critical outputs "
            "in the local build."
        )
    if invariant_kind == "rank_signed_measure":
        rank_query_axes = _native_rank_query_axes(filtered_complex, mp)
        res = mp.signed_measure(
            filtered_complex,
            degree=degree,
            invariant="rank",
            plot=False,
            verbose=False,
            grid=rank_query_axes,
        )
        meta = {**meta, "rank_query_axes": rank_query_axes}
    elif invariant_kind == "rank_invariant":
        target = _signed_measure_target_complex(filtered_complex, mp, invariant_kind)
        rank_query_axes = _native_rank_query_axes(target, mp)
        res = mp.signed_measure(
            target,
            degree=degree,
            invariant="rank",
            plot=False,
            verbose=False,
            grid=rank_query_axes,
        )
        meta = {**meta, "rank_query_axes": rank_query_axes}
    elif invariant_kind == "restricted_hilbert":
        target = _signed_measure_target_complex(filtered_complex, mp, invariant_kind)
        canonical_hilbert_axes = _hilbert_query_axes_from_case(case)
        hilbert_measure_axes = _native_rank_query_axes(target, mp)
        res = mp.signed_measure(
            target,
            degree=degree,
            invariant="hilbert",
            plot=False,
            verbose=False,
            grid=hilbert_measure_axes,
        )
        meta = {**meta, "hilbert_measure_axes": hilbert_measure_axes}
        if canonical_hilbert_axes is not None:
            meta["hilbert_query_axes"] = canonical_hilbert_axes
    elif invariant_kind == "slice_barcodes":
        directions, offsets = _slice_query_from_case(case)
        slicer = mp.Slicer(filtered_complex)
        barcodes: list[list[np.ndarray]] = []
        for direction in directions:
            row: list[np.ndarray] = []
            for offset in offsets:
                raw_bars = slicer.persistence_on_line(offset, direction=direction, full=False)
                if degree < len(raw_bars):
                    arr = np.asarray(raw_bars[degree], dtype=np.float64)
                else:
                    arr = np.empty((0, 2), dtype=np.float64)
                if arr.size == 0:
                    arr = np.empty((0, 2), dtype=np.float64)
                else:
                    arr = arr.reshape(-1, 2)
                row.append(arr)
            barcodes.append(row)
        res = {"directions": directions, "offsets": offsets, "barcodes": barcodes}
    elif invariant_kind == "mp_landscape":
        directions, offsets = _slice_query_from_case(case)
        kmax, tgrid = _mp_landscape_query_from_case(case)
        slicer = mp.Slicer(filtered_complex)
        values = np.zeros((len(directions), len(offsets), kmax, tgrid.size), dtype=np.float64)
        for i, direction in enumerate(directions):
            for j, offset in enumerate(offsets):
                raw_bars = slicer.persistence_on_line(offset, direction=direction, full=False)
                if degree < len(raw_bars):
                    arr = np.asarray(raw_bars[degree], dtype=np.float64)
                else:
                    arr = np.empty((0, 2), dtype=np.float64)
                if arr.size == 0:
                    arr = np.empty((0, 2), dtype=np.float64)
                else:
                    arr = arr.reshape(-1, 2)
                values[i, j, :, :] = _landscape_values_from_intervals(arr, tgrid, kmax)
        res = {
            "directions": directions,
            "offsets": offsets,
            "weights": np.ones((len(directions), len(offsets)), dtype=np.float64),
            "kmax": kmax,
            "tgrid": tgrid,
            "values": values,
        }
    elif invariant_kind == "euler_signed_measure":
        res = mp.signed_measure(
            filtered_complex,
            degree=None,
            invariant="euler",
            plot=False,
            verbose=False,
        )
    else:
        raise RuntimeError(f"Unsupported invariant_kind={invariant_kind}")
    return res, meta


def _extract_measure_stats(res) -> tuple[int, float]:
    terms = _canonical_measure_terms(res)
    abs_mass = 0.0
    for _, weight in terms:
        if "//" in weight:
            num, den = weight.split("//", 1)
            abs_mass += abs(int(num)) / abs(int(den))
        else:
            abs_mass += abs(float(weight))
    return len(terms), abs_mass


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "tool",
        "case_id",
        "regime",
        "invariant_kind",
        "degree",
        "n_points",
        "ambient_dim",
        "max_dim",
        "cold_ms",
        "cold_alloc_kib",
        "warm_median_ms",
        "warm_p90_ms",
        "warm_alloc_median_kib",
        "output_term_count",
        "output_abs_mass",
        "output_measure_canonical",
        "output_rank_query_axes_canonical",
        "output_rank_table_canonical",
        "notes",
        "timestamp_utc",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _run_rank_workaround_once(args, case_id: str, regime: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="multipers_rank_workaround_") as tmpdir:
        out_path = Path(tmpdir) / "rows.csv"
        cmd = [
            args.python if getattr(args, "python", None) else sys.executable,
            str(Path(__file__).resolve()),
            "--manifest",
            str(args.manifest),
            "--out",
            str(out_path),
            "--profile",
            str(args.profile),
            "--regime",
            regime,
            "--case",
            case_id,
            "--invariants",
            "rank_signed_measure",
            "--degree",
            str(args.degree),
            "--_hard_exit_after_write",
            "--reps",
            "1",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"cubical rank workaround subprocess failed with exit code {proc.returncode}")
        if not out_path.exists():
            raise RuntimeError("cubical rank workaround subprocess produced no CSV output")
        with out_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 1:
            raise RuntimeError(f"cubical rank workaround subprocess produced {len(rows)} rows, expected 1")
        return rows[0]


def _run_rank_workaround_subprocess(args, case_id: str, regime: str) -> list[dict[str, Any]]:
    defaults = _profile_defaults(args.profile)
    reps = defaults["reps"] if args.reps is None else args.reps
    cold_row = _run_rank_workaround_once(args, case_id, regime)
    warm_rows = [_run_rank_workaround_once(args, case_id, regime) for _ in range(reps)]

    stable_fields = [
        "tool",
        "case_id",
        "regime",
        "invariant_kind",
        "degree",
        "n_points",
        "ambient_dim",
        "max_dim",
        "output_term_count",
        "output_abs_mass",
        "output_measure_canonical",
        "output_rank_query_axes_canonical",
        "output_rank_table_canonical",
    ]
    for row in warm_rows:
        for key in stable_fields:
            if row[key] != cold_row[key]:
                raise RuntimeError(f"cubical rank workaround subprocess mismatch on {key}: {row[key]!r} != {cold_row[key]!r}")

    warm_times = [float(row["cold_ms"]) for row in warm_rows]
    out_row = dict(cold_row)
    out_row["warm_median_ms"] = format(statistics.median(warm_times), ".17g")
    out_row["warm_p90_ms"] = format(_p90(warm_times), ".17g")
    out_row["timestamp_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return [out_row]


def main() -> None:
    p = argparse.ArgumentParser(description="Run end-to-end invariant benchmark for multipers.")
    p.add_argument("--manifest", type=Path, default=Path(__file__).with_name("fixtures_invariants") / "manifest.toml")
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("results_multipers_invariants.csv"))
    p.add_argument("--profile", default="desktop")
    p.add_argument("--reps", type=int, default=None)
    p.add_argument("--regime", default="all")
    p.add_argument("--case", default="")
    p.add_argument("--invariants", default="all")
    p.add_argument("--degree", type=int, default=0)
    p.add_argument("--_hard_exit_after_write", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    defaults = _profile_defaults(args.profile)
    reps = defaults["reps"] if args.reps is None else args.reps
    if reps < 1:
        raise ValueError("--reps must be >= 1.")

    invariants = _parse_invariants(args.invariants)
    print(f"[profile] {args.profile} (reps={reps}, trim_between_reps={defaults['trim_between_reps']}, trim_between_cases={defaults['trim_between_cases']})")
    print(f"Invariants: {'all' if not invariants else invariants}")

    mmp, mp, gd, mf = _try_import_multipers()
    manifest = _load_manifest(args.manifest)
    cases = manifest["cases"]
    rows: list[dict[str, Any]] = []
    allow_memory_relief = not args._hard_exit_after_write

    for case in cases:
        case_id = str(case["id"])
        regime = str(case["regime"])
        if args.regime != "all" and regime != args.regime:
            continue
        if args.case and case_id != args.case:
            continue

        approved_invariants = _approved_invariants_for_regime(manifest, regime)
        selected_invariants = _selected_invariants_for_case(invariants, approved_invariants)
        if not selected_invariants:
            print(f"[skip] {case_id}: no benchmark-approved invariants for regime={regime}")
            continue

        fixture = args.manifest.parent / str(case["path"])
        n_points = int(case["n_points"])
        ambient_dim = int(case["ambient_dim"])
        max_dim = int(case["max_dim"])
        data = _load_case_data(case, fixture)

        for invariant_kind in selected_invariants:
            if _needs_hard_exit_rank_workaround(regime, invariant_kind) and not args._hard_exit_after_write:
                rows.extend(_run_rank_workaround_subprocess(args, case_id, regime))
                continue
            if invariant_kind == "slice_barcodes":
                notes = "cold_mode=warm_uncached;cache=none;stage=filtered_complex_to_slice_barcodes"
            elif invariant_kind == "mp_landscape":
                notes = "cold_mode=warm_uncached;cache=none;stage=filtered_complex_to_slice_barcodes_to_mp_landscape"
            elif invariant_kind == "rank_invariant":
                notes = "cold_mode=warm_uncached;cache=none;stage=filtered_complex_to_rank_invariant"
            elif invariant_kind == "restricted_hilbert":
                notes = "cold_mode=warm_uncached;cache=none;stage=filtered_complex_to_restricted_hilbert_measure_contract"
            else:
                notes = "cold_mode=warm_uncached;cache=none;stage=filtered_complex_to_signed_measure"
            if args._hard_exit_after_write:
                notes = notes + ";rank_workaround=hard_exit_subprocess"
            if args._hard_exit_after_write:
                try:
                    (cold_res, meta), cold_ms = _timed_call(
                        lambda: _run_invariant_uncached(mmp, mp, gd, mf, data, case, invariant_kind, args.degree)
                    )
                except Exception as exc:
                    print(f"[skip] {case_id} {invariant_kind}: {exc}")
                    continue
            else:
                try:
                    _run_invariant_uncached(mmp, mp, gd, mf, data, case, invariant_kind, args.degree)
                except Exception as exc:
                    print(f"[skip] {case_id} {invariant_kind}: {exc}")
                    continue

                if allow_memory_relief:
                    _memory_relief()
                (cold_res, meta), cold_ms = _timed_call(
                    lambda: _run_invariant_uncached(mmp, mp, gd, mf, data, case, invariant_kind, args.degree)
                )
            notes = notes + f";builder={meta.get('builder', 'unknown')}"
            if invariant_kind == "slice_barcodes":
                output_term_count, output_abs_mass, output_measure_canonical = _slice_output_contract(cold_res)
                output_rank_query_axes_canonical = ""
                output_rank_table_canonical = ""
            elif invariant_kind == "mp_landscape":
                output_term_count, output_abs_mass, output_measure_canonical = _mp_landscape_output_contract(cold_res)
                output_rank_query_axes_canonical = ""
                output_rank_table_canonical = ""
            elif invariant_kind == "rank_invariant":
                terms = _sparse_rank_terms_from_measure(meta["rank_query_axes"], cold_res)
                output_term_count = len(terms)
                output_abs_mass = sum(abs(float(weight)) for _, weight in terms)
                output_measure_canonical = _serialize_measure_terms(terms)
                output_rank_query_axes_canonical = ""
                output_rank_table_canonical = ""
            elif invariant_kind == "restricted_hilbert":
                hilbert_axes, hilbert_measure = _normalize_hilbert_measure_contract(
                    meta["hilbert_measure_axes"], cold_res
                )
                _ = hilbert_axes
                terms = _canonical_measure_terms(hilbert_measure)
                output_term_count = len(terms)
                output_abs_mass = sum(abs(float(weight)) for _, weight in terms)
                output_measure_canonical = _serialize_measure_terms(terms)
                output_rank_query_axes_canonical = ""
                output_rank_table_canonical = ""
            else:
                if invariant_kind == "rank_signed_measure":
                    _, rank_measure = _normalize_rank_measure_contract(meta["rank_query_axes"], cold_res)
                    output_term_count, output_abs_mass = _extract_measure_stats(rank_measure)
                    output_measure_canonical = _serialize_measure_terms(_canonical_measure_terms(rank_measure))
                    output_rank_query_axes_canonical = ""
                    output_rank_table_canonical = ""
                else:
                    output_term_count, output_abs_mass = _extract_measure_stats(cold_res)
                    output_measure_canonical = _serialize_measure_terms(_canonical_measure_terms(cold_res))
                    output_rank_query_axes_canonical = ""
                    output_rank_table_canonical = ""

            warm_times: list[float] = []
            if args._hard_exit_after_write:
                warm_times.append(cold_ms)
            else:
                for _ in range(reps):
                    _, tms = _timed_call(
                        lambda: _run_invariant_uncached(mmp, mp, gd, mf, data, case, invariant_kind, args.degree)
                    )
                    warm_times.append(tms)
                    if defaults["trim_between_reps"] and allow_memory_relief:
                        _memory_relief()

            warm_median_ms = statistics.median(warm_times)
            warm_p90_ms = _p90(warm_times)

            print(
                f"{case_id:<32} inv={invariant_kind} cold_ms={cold_ms:.3f} "
                f"warm_med_ms={warm_median_ms:.3f} terms={output_term_count}"
            )

            rows.append(
                {
                    "tool": "multipers",
                    "case_id": case_id,
                    "regime": regime,
                    "invariant_kind": invariant_kind,
                    "degree": args.degree,
                    "n_points": n_points,
                    "ambient_dim": ambient_dim,
                    "max_dim": max_dim,
                    "cold_ms": cold_ms,
                    "cold_alloc_kib": "",
                    "warm_median_ms": warm_median_ms,
                    "warm_p90_ms": warm_p90_ms,
                    "warm_alloc_median_kib": "",
                    "output_term_count": output_term_count,
                    "output_abs_mass": output_abs_mass,
                    "output_measure_canonical": output_measure_canonical,
                    "output_rank_query_axes_canonical": output_rank_query_axes_canonical,
                    "output_rank_table_canonical": output_rank_table_canonical,
                    "notes": notes,
                    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
            )

            if defaults["trim_between_cases"] and allow_memory_relief:
                _memory_relief()

    if not rows:
        raise RuntimeError("No invariant benchmark rows were produced.")
    _write_rows(args.out, rows)
    print(f"Wrote multipers invariant results: {args.out}")
    if args._hard_exit_after_write:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
