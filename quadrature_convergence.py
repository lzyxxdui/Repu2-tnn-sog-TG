#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quadrature convergence without TNN training or checkpoints.

The analytic state (for example ``flower``) is sampled once on a fixed uniform
3D tensor grid.  The resulting magnetization tensor is kept in full: there is
no CP/Tucker/TNN compression.  Between grid nodes, tensor-product piecewise
linear interpolation is used.

The SOG operator is evaluated with the same narrow-Gaussian local integration
used in the original TNN code.  Therefore the A/B/C studies below change only
the target/energy Gauss--Legendre rule; the magnetization representation, SOG
parameters and local convolution quadrature remain fixed.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from dense_grid_sog import (
    DenseGridMagnetization,
    dense_grid_demag_energy_from_potential_weak,
    dense_grid_sog_potential_and_faces,
)
from quadrature import composite_gauss_legendre, gauss_legendre_interval
from sog import build_sog
from states import case_spec, state_function

Counts3 = Tuple[int, int, int]


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def resolve_counts(
    isotropic: int,
    directional: Sequence[int | None],
) -> Counts3:
    if isotropic < 2:
        raise ValueError("source-grid-n must be at least 2")
    out = []
    for value in directional:
        out.append(int(isotropic if value is None else value))
    if any(v < 2 for v in out):
        raise ValueError("all source-grid counts must be at least 2")
    return tuple(out)  # type: ignore[return-value]


def compact_xyz(values: Sequence[int]) -> int | str:
    if values[0] == values[1] == values[2]:
        return int(values[0])
    return ",".join(str(int(v)) for v in values)


def build_target_rule(
    bounds: Sequence[Tuple[float, float]],
    mode: str,
    M: int,
    k: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    if mode not in ("global", "composite"):
        raise ValueError(f"unsupported Gauss mode: {mode}")
    if M < 1 or k < 2:
        raise ValueError("require M>=1 and k>=2")
    if mode == "global" and M != 1:
        raise ValueError("global mode requires M=1")

    nodes: List[Tensor] = []
    weights: List[Tensor] = []
    for a, b in bounds:
        if mode == "global":
            x, w = gauss_legendre_interval(k, a, b, device=device, dtype=dtype)
        else:
            x, w = composite_gauss_legendre(
                a, b, M, k, device=device, dtype=dtype
            )
        nodes.append(x)
        weights.append(w)
    return tuple(nodes), tuple(weights)  # type: ignore[return-value]


def study_configs(study: str) -> Iterable[Tuple[str, str, int, int]]:
    # A: same N_1d=32, different spatial distributions.
    if study in ("A", "all"):
        yield "A", "global", 1, 32
        yield "A", "composite", 2, 16
        yield "A", "composite", 4, 8
        yield "A", "composite", 8, 4

    # B: fixed local order k=4, h-refinement.
    if study in ("B", "all"):
        for M in (4, 6, 8, 12, 16):
            yield "B", "composite", M, 4

    # C: fixed M=8, local p-refinement.
    if study in ("C", "all"):
        for k in (2, 4, 6, 8):
            yield "C", "composite", 8, k


def upsert_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "study",
        "state",
        "representation",
        "source_grid_n",
        "mode",
        "M",
        "k",
        "points_1d",
        "total_points",
        "Ed",
        "Ed_literature",
        "relative_literature_error",
        "evaluation_seconds",
        "sog_h",
        "sog_K",
        "conv_gauss_order",
        "conv_tail_tol",
        "literature_label",
    ]

    existing: List[Dict[str, str]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames == fieldnames:
                existing = list(reader)

    key_names = (
        "study",
        "state",
        "representation",
        "source_grid_n",
        "mode",
        "M",
        "k",
        "sog_h",
        "sog_K",
        "conv_gauss_order",
    )

    def key(row: Dict[str, object]) -> Tuple[str, ...]:
        return tuple(str(row.get(name, "")) for name in key_names)

    by_key: Dict[Tuple[str, ...], Dict[str, object]] = {
        key(row): dict(row) for row in existing
    }
    for row in rows:
        by_key[key(row)] = dict(row)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in by_key.values():
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Composite/global target Gauss convergence using an uncompressed "
            "dense tensor-grid magnetization; no TNN training/checkpoint."
        )
    )
    parser.add_argument(
        "--state",
        choices=("flower", "vortex_cube", "homo_film", "vortex_film"),
        default="flower",
    )
    parser.add_argument("--study", choices=("A", "B", "C", "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output-dir", type=Path, default=Path("quadrature_exact"))

    # Fixed, uncompressed source representation.  These nodes include the box
    # boundary and are NOT Gauss nodes.  They stay unchanged throughout A/B/C.
    parser.add_argument("--source-grid-n", type=int, default=33)
    parser.add_argument("--source-grid-n-x", type=int, default=None)
    parser.add_argument("--source-grid-n-y", type=int, default=None)
    parser.add_argument("--source-grid-n-z", type=int, default=None)

    # SOG and its local narrow-Gaussian integration: fixed during A/B/C.
    parser.add_argument("--sog-h", type=float, default=0.2)
    parser.add_argument("--sog-K", type=int, default=120)
    parser.add_argument("--conv-gauss-order", type=int, default=28)
    parser.add_argument("--conv-tail-tol", type=float, default=1.0e-15)
    parser.add_argument("--mu0", type=float, default=1.0)
    parser.add_argument("--Ms", type=float, default=1.0)

    # Memory/performance batching only.  These do not change the mathematics.
    parser.add_argument("--dense-alpha-chunk", type=int, default=8)
    parser.add_argument("--dense-x-chunk", type=int, default=64)
    parser.add_argument("--dense-contraction-alpha-chunk", type=int, default=4)

    # Analytic state parameters, matching main.py.
    parser.add_argument("--flower-a", type=float, default=1.0)
    parser.add_argument("--flower-b", type=float, default=2.0)
    parser.add_argument("--flower-c", type=float, default=1.0)
    parser.add_argument("--vortex-core-radius", type=float, default=0.14)
    parser.add_argument("--vortex-center-x", type=float, default=0.5)
    parser.add_argument("--vortex-center-y", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    spec = case_spec(args.state)
    exact = state_function(args)

    if args.sog_h <= 0.0 or args.sog_K < 1:
        raise ValueError("SOG requires --sog-h>0 and --sog-K>=1")
    if args.conv_gauss_order < 2:
        raise ValueError("--conv-gauss-order must be at least 2")
    if not (0.0 < args.conv_tail_tol < 1.0):
        raise ValueError("--conv-tail-tol must lie in (0,1)")
    if min(
        args.dense_alpha_chunk,
        args.dense_x_chunk,
        args.dense_contraction_alpha_chunk,
    ) < 1:
        raise ValueError("dense chunk sizes must be positive")

    source_counts = resolve_counts(
        args.source_grid_n,
        (args.source_grid_n_x, args.source_grid_n_y, args.source_grid_n_z),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "quadrature_convergence.csv"

    print("\n=== Exact-state / dense-grid quadrature convergence ===")
    print("No TNN, no CP/Tucker, no checkpoint.")
    print(f"state = {spec.name}")
    print(f"device / dtype = {device} / {dtype}")
    print(f"source dense grid = {compact_xyz(source_counts)} nodes/direction")
    print("source representation = full tensor + trilinear interpolation")
    print(
        f"SOG fixed: h={args.sog_h}, K={args.sog_K}, "
        f"local conv order={args.conv_gauss_order}, tail_tol={args.conv_tail_tol}"
    )
    if spec.literature_energy is not None:
        print(
            f"literature Ed = {spec.literature_energy:.12e} "
            f"({spec.literature_label})"
        )

    # The analytic state is sampled ONCE.  This representation remains fixed
    # for all target quadrature rules in A/B/C.
    build_start = time.perf_counter()
    dense_m = DenseGridMagnetization.from_exact_state(
        exact,
        spec.bounds,
        source_counts,
        device=device,
        dtype=dtype,
    )
    build_seconds = time.perf_counter() - build_start
    print(
        f"dense m tensor shape = {tuple(dense_m.values.shape)}, "
        f"build_seconds = {build_seconds:.6f}"
    )

    sog = build_sog(args.sog_h, args.sog_K, device=device, dtype=dtype)
    rows: List[Dict[str, object]] = []

    for study, mode, M, k in study_configs(args.study):
        points_1d = k if mode == "global" else M * k
        total_points = points_1d**3
        print(f"\nStudy {study}: target Gauss = {mode}")
        print(f"  subintervals = {M}")
        print(f"  order/subinterval = {k}")
        print(f"  points/direction = {points_1d}")
        print(f"  total tensor-product points = {total_points}")

        target_nodes, target_weights = build_target_rule(
            spec.bounds,
            mode,
            M,
            k,
            device=device,
            dtype=dtype,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        potential, face_potentials = dense_grid_sog_potential_and_faces(
            dense_m,
            target_nodes,
            sog,
            conv_gauss_order=args.conv_gauss_order,
            tail_tolerance=args.conv_tail_tol,
            alpha_chunk=args.dense_alpha_chunk,
            x_chunk=args.dense_x_chunk,
            contraction_alpha_chunk=args.dense_contraction_alpha_chunk,
        )
        Ed = dense_grid_demag_energy_from_potential_weak(
            dense_m,
            target_nodes,
            target_weights,
            potential,
            face_potentials,
            mu0=args.mu0,
            Ms=args.Ms,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        evaluation_seconds = time.perf_counter() - start

        lit = spec.literature_energy
        if lit is None:
            rel_lit = float("nan")
        else:
            rel_lit = abs(Ed - lit) / max(abs(lit), 1.0e-30)

        print(f"  Ed = {Ed:.12e}")
        if lit is not None:
            print(f"  relative literature error = {rel_lit:.6e}")
        print(f"  evaluation_seconds = {evaluation_seconds:.6f}")

        rows.append(
            {
                "study": study,
                "state": spec.name,
                "representation": "dense_grid_exact_state_no_compression",
                "source_grid_n": compact_xyz(source_counts),
                "mode": mode,
                "M": M,
                "k": k,
                "points_1d": points_1d,
                "total_points": total_points,
                "Ed": Ed,
                "Ed_literature": "" if lit is None else lit,
                "relative_literature_error": rel_lit,
                "evaluation_seconds": evaluation_seconds,
                "sog_h": args.sog_h,
                "sog_K": args.sog_K,
                "conv_gauss_order": args.conv_gauss_order,
                "conv_tail_tol": args.conv_tail_tol,
                "literature_label": spec.literature_label or "",
            }
        )

        # Do not retain target-sized tensors between cases.
        del potential, face_potentials, target_nodes, target_weights
        if device.type == "cuda":
            torch.cuda.empty_cache()

    upsert_rows(csv_path, rows)
    print(f"\nquadrature_convergence.csv: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
