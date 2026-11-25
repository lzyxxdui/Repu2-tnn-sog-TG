#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line driver for TNN/SOG demagnetization benchmarks.

Available cases
---------------
* flower          : 1 x 1 x 1 cube (CP baseline + TNN)
* vortex_cube     : 1 x 1 x 1 cube (CP baseline + TNN)
* homo_film       : 1 x 1 x 0.1 thin film, m=(0,0,1) (TNN only)
* vortex_film     : 1 x 1 x 0.1 thin film, z-invariant vortex (TNN only)

The thin-film cases use anisotropic Gauss rules.  Directional orders can be
set independently from the command line, while the SOG local convolution
order remains one common parameter because the t=sqrt(alpha)(y-x) change of
variables already resolves the physical interval length in each direction.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from cp import fit_componentwise_cp
from diagnostics import make_plots, save_componentwise_cp
from metrics import point_metrics, weighted_relative_scalar
from quadrature import composite_gauss_legendre, gauss_legendre_interval
from knot_diagnostics import KnotRecorder, initialize_uniform_knots, parse_epoch_list
from sog import (
    build_sog,
    demag_energy_from_potential_weak,
    representation_on_product_grid,
    sog_potential_and_faces,
    sog_potential_on_product_grid,
)
from states import (
    CaseSpec,
    case_spec,
    random_points_in_box,
    state_function,
    state_tensor,
)
from tnn import (
    ComponentwiseTNN,
    MarginalGradientTargets,
    build_marginal_gradient_targets,
    expand_tnn_rank,
    train_tnn_deterministic,
)

Orders3 = Tuple[int, int, int]
Counts3 = Tuple[int, int, int]


def parse_int_list(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError("rank list must contain positive integers")
    return sorted(set(values))


def parse_positive_int_sequence(text: str, *, name: str) -> Tuple[int, ...]:
    """Parse an ordered comma-separated list while preserving duplicates."""
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value < 1 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def parse_quadrature_sweep(text: str) -> List[Tuple[int, int]]:
    """Parse a comma-separated target quadrature sweep such as ``4x4,8x2``.

    Each pair is ``m x k``, where ``m`` is the number of composite
    subintervals per spatial direction and ``k`` is the Gauss order on every
    subinterval.  The input order is preserved.
    """
    pairs: List[Tuple[int, int]] = []
    for item in text.split(","):
        token = item.strip().lower().replace("*", "x").replace(":", "x")
        if not token:
            continue
        parts = [part.strip() for part in token.split("x") if part.strip()]
        if len(parts) != 2:
            raise ValueError(
                "target-quadrature-sweep entries must have the form m x k, "
                "for example 4x4,8x2,16x4"
            )
        m, k = int(parts[0]), int(parts[1])
        if m < 1 or k < 2:
            raise ValueError(
                "target quadrature sweep requires m>=1 and k>=2 for every pair"
            )
        pair = (m, k)
        if pair not in pairs:
            pairs.append(pair)
    if not pairs:
        raise ValueError("target-quadrature-sweep must contain at least one m x k pair")
    return pairs


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def resolve_orders(
    isotropic: int | None,
    directional: Sequence[int | None],
    defaults: Orders3,
) -> Orders3:
    if isotropic is not None and isotropic < 2:
        raise ValueError("Gauss order must be at least 2")
    out = []
    for dim in range(3):
        value = directional[dim]
        if value is None:
            value = isotropic if isotropic is not None else defaults[dim]
        if value < 2:
            raise ValueError("Gauss order must be at least 2")
        out.append(int(value))
    return tuple(out)  # type: ignore[return-value]


def resolve_counts(
    isotropic: int | None,
    directional: Sequence[int | None],
    defaults: Counts3 = (1, 1, 1),
) -> Counts3:
    if isotropic is not None and isotropic < 1:
        raise ValueError("Gauss subinterval count must be at least 1")
    out = []
    for dim in range(3):
        value = directional[dim]
        if value is None:
            value = isotropic if isotropic is not None else defaults[dim]
        if value < 1:
            raise ValueError("Gauss subinterval count must be at least 1")
        out.append(int(value))
    return tuple(out)  # type: ignore[return-value]


def build_axis_rules(
    orders: Orders3,
    bounds,
    *,
    mode: str = "global",
    subintervals: Counts3 = (1, 1, 1),
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """Build x/y/z rules, independently supporting global or composite Gauss."""
    if mode not in ("global", "composite"):
        raise ValueError(f"unsupported Gauss mode: {mode}")
    if mode == "global" and any(m != 1 for m in subintervals):
        raise ValueError(
            "global Gauss mode requires one subinterval per direction; "
            "use --*-gauss-subintervals 1 or switch to composite mode"
        )
    nodes = []
    weights = []
    for order, n_sub, (a, b) in zip(orders, subintervals, bounds):
        if mode == "global":
            x, w = gauss_legendre_interval(order, a, b, device=device, dtype=dtype)
        else:
            x, w = composite_gauss_legendre(
                a, b, n_sub, order, device=device, dtype=dtype
            )
        nodes.append(x)
        weights.append(w)
    return tuple(nodes), tuple(weights)  # type: ignore[return-value]


def quadrature_points_1d(mode: str, subintervals: Counts3, orders: Orders3) -> Counts3:
    if mode == "global":
        return orders
    return tuple(int(m * k) for m, k in zip(subintervals, orders))  # type: ignore[return-value]


def _compact_xyz(values: Sequence[int]) -> int | str:
    return int(values[0]) if values[0] == values[1] == values[2] else ",".join(map(str, values))


def print_gauss_summary(label: str, mode: str, subintervals: Counts3, orders: Orders3) -> None:
    points = quadrature_points_1d(mode, subintervals, orders)
    total = int(np.prod(points))
    print(f"{label} Gauss: {mode}")
    print(f"  subintervals = {_compact_xyz(subintervals)}")
    print(f"  order/subinterval = {_compact_xyz(orders)}")
    print(f"  points/direction = {_compact_xyz(points)}")
    print(f"  total points = {total}")


def upsert_quadrature_convergence(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Create/update the compact target-quadrature convergence table."""
    fieldnames = [
        "study",
        "state",
        "method",
        "mode",
        "M",
        "k",
        "points_1d",
        "total_points",
        "Ed",
        "evaluation_seconds",
        "rank",
        "sog_h",
        "sog_K",
        "conv_gauss_order",
    ]
    existing: List[Dict[str, str]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames == fieldnames:
                existing = list(reader)

    def key(row: Dict[str, object]) -> Tuple[str, str, str, str, str, str]:
        return tuple(
            str(row[name]) for name in ("study", "state", "method", "mode", "M", "k")
        )  # type: ignore[return-value]

    by_key: Dict[Tuple[str, str, str, str, str, str], Dict[str, object]] = {
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
        description="Cube/thin-film TNN--SOG demagnetization benchmarks."
    )
    parser.add_argument(
        "--state",
        choices=("flower", "vortex_cube", "homo_film", "vortex_film"),
        default="flower",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--flower-a", type=float, default=1.0)
    parser.add_argument("--flower-b", type=float, default=2.0)
    parser.add_argument("--flower-c", type=float, default=1.0)
    parser.add_argument("--vortex-core-radius", type=float, default=0.14)
    parser.add_argument(
        "--vortex-center-x",
        type=float,
        default=0.5,
        help="x-coordinate of the vortex center (default: 0.5).",
    )
    parser.add_argument(
        "--vortex-center-y",
        type=float,
        default=0.5,
        help="y-coordinate of the vortex center (default: 0.5).",
    )
    parser.add_argument(
        "--vortex-z-invariant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Impose exact z-invariance for vortex_cube and vortex_film.",
    )

    # CP remains a cube-only baseline. Thin-film cases intentionally use TNN only.
    parser.add_argument("--grid-n", type=int, default=33)
    parser.add_argument("--cp-rank", type=int, default=6)
    parser.add_argument("--cp-restarts", type=int, default=5)
    parser.add_argument("--cp-max-sweeps", type=int, default=600)
    parser.add_argument("--cp-tol", type=float, default=1.0e-10)
    parser.add_argument("--cp-reg", type=float, default=1.0e-12)
    parser.add_argument(
        "--no-cp",
        action="store_true",
        help="Disable the CP baseline for cube cases. Thin-film cases are TNN-only regardless.",
    )

    parser.add_argument(
        "--tnn-ranks",
        type=str,
        default=None,
        help="Comma-separated ranks. If omitted, a case-specific default is used.",
    )
    parser.add_argument("--tnn-width", type=int, default=64)
    parser.add_argument("--tnn-depth", type=int, default=1)
    parser.add_argument(
        "--tnn-widths",
        type=str,
        default=None,
        help=(
            "Optional comma-separated hidden widths, e.g. 64,128 for "
            "[1,64,128,P]. When supplied, its length must equal --tnn-depth. "
            "If omitted, the legacy behavior is preserved and --tnn-width is "
            "repeated for every hidden layer."
        ),
    )
    parser.add_argument(
        "--activation",
        choices=("relu", "repu2", "repu3", "tanh", "sin"),
        default="repu2",
    )
    parser.add_argument(
        "--loss-function",
        choices=("default", "gradM"),
        default="default",
        help="Thin-film anisotropic rules currently support default loss; gradM is retained for cube runs.",
    )
    parser.add_argument("--gradm-mean-weight", type=float, default=1.0)
    parser.add_argument("--gradm-target-chunk", type=int, default=8)

    # Train/target Gauss rules. ``--gauss-mode`` is a shorthand applying to both;
    # the more specific train/target mode flags override it when supplied.
    parser.add_argument(
        "--gauss-mode",
        choices=("global", "composite"),
        default="global",
        help="Shorthand Gauss mode for both training and target quadrature.",
    )
    parser.add_argument(
        "--train-gauss-mode",
        choices=("global", "composite"),
        default=None,
        help="Override --gauss-mode for TNN training quadrature only.",
    )
    parser.add_argument(
        "--target-gauss-mode",
        choices=("global", "composite"),
        default=None,
        help="Override --gauss-mode for target/energy quadrature only.",
    )

    # Isotropic shorthands + optional per-axis overrides.
    parser.add_argument("--train-gauss-subintervals", type=int, default=1)
    parser.add_argument("--train-gauss-subintervals-x", type=int, default=None)
    parser.add_argument("--train-gauss-subintervals-y", type=int, default=None)
    parser.add_argument("--train-gauss-subintervals-z", type=int, default=None)
    parser.add_argument("--train-gauss-order", type=int, default=None)
    parser.add_argument("--train-gauss-order-x", type=int, default=None)
    parser.add_argument("--train-gauss-order-y", type=int, default=None)
    parser.add_argument("--train-gauss-order-z", type=int, default=None)

    parser.add_argument("--target-gauss-subintervals", type=int, default=1)
    parser.add_argument("--target-gauss-subintervals-x", type=int, default=None)
    parser.add_argument("--target-gauss-subintervals-y", type=int, default=None)
    parser.add_argument("--target-gauss-subintervals-z", type=int, default=None)
    parser.add_argument("--target-gauss-order", type=int, default=None)
    parser.add_argument("--target-gauss-order-x", type=int, default=None)
    parser.add_argument("--target-gauss-order-y", type=int, default=None)
    parser.add_argument("--target-gauss-order-z", type=int, default=None)
    parser.add_argument(
        "--target-quadrature-sweep",
        type=str,
        default=None,
        help=(
            "Optional isotropic composite target-quadrature sweep. Example: "
            "4x4,8x2,4x8. Each pair is m x k, with m subintervals per "
            "direction and k Gauss points per subinterval. The TNN/CP models "
            "are trained/fitted only once and reused for every target rule."
        ),
    )
    parser.add_argument(
        "--sog-reference-energy",
        type=float,
        default=None,
        help=(
            "Optional internal SOG reference energy used only to report a "
            "controlled relative energy error in the target quadrature sweep."
        ),
    )

    parser.add_argument("--factor-norm-gauss-order", type=int, default=64)
    parser.add_argument("--factor-norm-gauss-order-x", type=int, default=None)
    parser.add_argument("--factor-norm-gauss-order-y", type=int, default=None)
    parser.add_argument("--factor-norm-gauss-order-z", type=int, default=None)

    parser.add_argument("--adam-epochs", type=int, default=10000)
    parser.add_argument("--adam-lr", type=float, default=5.0e-4)
    parser.add_argument("--lbfgs-iters", type=int, default=500)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--test-points", type=int, default=100000)

    # Optional one-hidden-layer ReLU/RePU knot tracking. This is diagnostic only
    # and does not change the loss, optimizer, quadrature, SOG, or energy.
    parser.add_argument(
        "--track-knots",
        action="store_true",
        help="Track first-layer knots during Adam training (vortex, depth=1, ReLU/RePU only).",
    )
    parser.add_argument(
        "--knot-snapshot-epochs",
        type=str,
        default="0,500,1000,1500,3000,5000,7000",
        help="Comma-separated fixed Adam epochs for knot snapshots.",
    )
    parser.add_argument(
        "--knot-init",
        choices=("uniform", "current"),
        default="uniform",
        help=(
            "Initialization used only when --track-knots is enabled. "
            "uniform starts from an evenly distributed knot partition; "
            "current preserves the network's ordinary initialization."
        ),
    )
    parser.add_argument(
        "--knot-loss-drop",
        type=float,
        default=0.0,
        help=(
            "Optional extra snapshot rule: save when best loss has fallen by this "
            "fraction relative to the previous loss-triggered snapshot, e.g. 0.30. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--knot-min-epoch-gap",
        type=int,
        default=200,
        help="Minimum epoch gap between loss-triggered knot snapshots.",
    )
    parser.add_argument(
        "--knot-topk-per-factor",
        type=int,
        default=12,
        help="Number of most important knots per component/direction shown in the partition figure.",
    )
    parser.add_argument(
        "--knot-save-model-snapshots",
        action="store_true",
        help="Also save a small .pt checkpoint at every knot snapshot.",
    )

    parser.add_argument("--sog-h", type=float, default=0.2)
    parser.add_argument("--sog-K", type=int, default=120)
    parser.add_argument(
        "--conv-gauss-order",
        type=int,
        default=28,
        help="Gauss order for each local 1D SOG convolution after the narrow-Gaussian variable change.",
    )
    parser.add_argument("--conv-tail-tol", type=float, default=1.0e-15)
    parser.add_argument("--mu0", type=float, default=1.0)
    parser.add_argument("--Ms", type=float, default=1.0)
    parser.add_argument("--make-plot", action="store_true")
    return parser.parse_args()


def initialize_homogeneous_tnn(model: ComponentwiseTNN) -> None:
    """Make homo_film exactly m=(0,0,1) at initialization when rank >= 1."""
    with torch.no_grad():
        model.components[0].coefficients.zero_()
        model.components[1].coefficients.zero_()
        model.components[2].coefficients.zero_()
        model.components[2].coefficients[0] = 1.0


def main() -> None:
    args = parse_args()
    spec: CaseSpec = case_spec(args.state)
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    if args.tnn_width < 1 or args.tnn_depth < 1:
        raise ValueError("tnn-width and tnn-depth must be positive")

    if args.tnn_widths is None:
        tnn_hidden_widths = tuple([args.tnn_width] * args.tnn_depth)
    else:
        tnn_hidden_widths = parse_positive_int_sequence(
            args.tnn_widths, name="tnn-widths"
        )
        if len(tnn_hidden_widths) != args.tnn_depth:
            raise ValueError(
                "--tnn-widths must contain exactly --tnn-depth entries; "
                f"got widths={tnn_hidden_widths} with depth={args.tnn_depth}"
            )

    if args.track_knots:
        if args.state not in ("vortex_cube", "vortex_film"):
            raise ValueError("--track-knots is currently intended for vortex_cube/vortex_film")
        if args.tnn_depth != 1:
            raise ValueError("--track-knots requires --tnn-depth 1")
        if args.activation not in ("relu", "repu2", "repu3"):
            raise ValueError("--track-knots requires relu, repu2, or repu3")
        if not (0.0 <= args.knot_loss_drop < 1.0):
            raise ValueError("--knot-loss-drop must satisfy 0 <= value < 1")
        if args.knot_min_epoch_gap < 0:
            raise ValueError("--knot-min-epoch-gap must be non-negative")
        if args.knot_topk_per_factor < 1:
            raise ValueError("--knot-topk-per-factor must be positive")
    if args.conv_gauss_order < 2:
        raise ValueError("conv-gauss-order must be at least 2")
    if args.state in ("vortex_cube", "vortex_film"):
        xlo, xhi = spec.bounds[0]
        ylo, yhi = spec.bounds[1]
        if not (xlo <= args.vortex_center_x <= xhi):
            raise ValueError(
                f"--vortex-center-x must lie in [{xlo}, {xhi}] for {args.state}"
            )
        if not (ylo <= args.vortex_center_y <= yhi):
            raise ValueError(
                f"--vortex-center-y must lie in [{ylo}, {yhi}] for {args.state}"
            )
    torch.set_num_threads(args.cpu_threads)

    device = choose_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    ranks_text = args.tnn_ranks or spec.default_tnn_ranks
    tnn_ranks = parse_int_list(ranks_text)
    exact = state_function(args)

    # The published vortex energies correspond to a vortex centered at (0.5, 0.5).
    # A shifted vortex is a different finite-domain benchmark, so the centered
    # literature value must not be reported as its reference energy.
    standard_vortex_center = (
        abs(args.vortex_center_x - 0.5) <= 1.0e-14
        and abs(args.vortex_center_y - 0.5) <= 1.0e-14
    )
    literature_energy = spec.literature_energy
    literature_label = spec.literature_label
    if args.state in ("vortex_cube", "vortex_film") and not standard_vortex_center:
        literature_energy = None
        literature_label = None

    train_mode = args.train_gauss_mode or args.gauss_mode
    target_mode = args.target_gauss_mode or args.gauss_mode
    train_subintervals = resolve_counts(
        args.train_gauss_subintervals,
        (
            args.train_gauss_subintervals_x,
            args.train_gauss_subintervals_y,
            args.train_gauss_subintervals_z,
        ),
    )
    target_subintervals = resolve_counts(
        args.target_gauss_subintervals,
        (
            args.target_gauss_subintervals_x,
            args.target_gauss_subintervals_y,
            args.target_gauss_subintervals_z,
        ),
    )
    train_orders = resolve_orders(
        args.train_gauss_order,
        (args.train_gauss_order_x, args.train_gauss_order_y, args.train_gauss_order_z),
        spec.default_train_orders,
    )
    target_orders = resolve_orders(
        args.target_gauss_order,
        (args.target_gauss_order_x, args.target_gauss_order_y, args.target_gauss_order_z),
        spec.default_target_orders,
    )
    if train_mode == "global" and any(m != 1 for m in train_subintervals):
        raise ValueError("train global Gauss requires --train-gauss-subintervals 1")
    if target_mode == "global" and any(m != 1 for m in target_subintervals):
        raise ValueError("target global Gauss requires --target-gauss-subintervals 1")
    train_points = quadrature_points_1d(train_mode, train_subintervals, train_orders)
    target_points = quadrature_points_1d(target_mode, target_subintervals, target_orders)
    target_quadrature_sweep = (
        parse_quadrature_sweep(args.target_quadrature_sweep)
        if args.target_quadrature_sweep is not None
        else []
    )
    if target_quadrature_sweep and target_mode != "composite":
        raise ValueError(
            "--target-quadrature-sweep uses composite Gauss rules; "
            "set --target-gauss-mode composite"
        )

    factor_orders = resolve_orders(
        args.factor_norm_gauss_order,
        (
            args.factor_norm_gauss_order_x,
            args.factor_norm_gauss_order_y,
            args.factor_norm_gauss_order_z,
        ),
        (64, 64, 64),
    )

    is_film = args.state in ("homo_film", "vortex_film")
    if is_film and args.loss_function != "default":
        raise ValueError(
            "Thin-film runs use anisotropic Gauss rules; use --loss-function default. "
            "The older gradM implementation is retained only for cube experiments."
        )

    z_invariant = spec.z_invariant
    if args.state in ("vortex_cube", "vortex_film"):
        z_invariant = args.vortex_z_invariant
    if is_film and not z_invariant:
        raise ValueError("The two implemented thin-film benchmarks are z-invariant; keep z-invariance enabled.")

    if args.output_dir is None:
        tag_r = "_".join(str(r) for r in tnn_ranks)
        center_tag = ""
        if args.state in ("vortex_cube", "vortex_film") and not standard_vortex_center:
            xc = str(args.vortex_center_x).replace("-", "m").replace(".", "p")
            yc = str(args.vortex_center_y).replace("-", "m").replace(".", "p")
            center_tag = f"_xc{xc}_yc{yc}"
        args.output_dir = Path(
            f"{args.state}{center_tag}_tnn{tag_r}_{args.activation}_{args.loss_function}_sog_results"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"\n=== {args.state}: TNN ranks {tnn_ranks} with SOG ===")
    print(f"domain = {spec.bounds}")
    print(f"device / dtype = {device} / {dtype}")
    print(
        f"TNN: activation={args.activation}, depth={args.tnn_depth}, "
        f"hidden_widths={tnn_hidden_widths}, loss={args.loss_function}, "
        f"z_invariant={z_invariant}"
    )
    if args.state in ("vortex_cube", "vortex_film"):
        print(
            f"vortex center = ({args.vortex_center_x:.6f}, {args.vortex_center_y:.6f}), "
            f"core radius = {args.vortex_core_radius:.6f}"
        )
    knot_fixed_epochs = parse_epoch_list(args.knot_snapshot_epochs) if args.track_knots else ()
    if args.track_knots:
        print(
            "knot tracking: fixed epochs="
            f"{knot_fixed_epochs}, init={args.knot_init}, loss_drop={args.knot_loss_drop}, "
            f"min_gap={args.knot_min_epoch_gap}, topk/factor={args.knot_topk_per_factor}"
        )
    print_gauss_summary("train", train_mode, train_subintervals, train_orders)
    if target_quadrature_sweep:
        print("target quadrature sweep (m,k,points/direction,total points):")
        for m, k in target_quadrature_sweep:
            n1d = m * k
            print(f"  ({m:2d},{k:2d}) -> {n1d:3d} per direction, {n1d**3} total")
    else:
        print_gauss_summary("target", target_mode, target_subintervals, target_orders)
    print(f"factor norm Gauss (global) orders (x,y,z) = {factor_orders}")
    print(f"SOG local convolution order = {args.conv_gauss_order}")
    if literature_energy is not None:
        print(
            f"literature energy = {literature_energy:.12e} "
            f"({literature_label})"
        )
    elif args.state in ("vortex_cube", "vortex_film") and not standard_vortex_center:
        print(
            "literature energy comparison disabled: the published reference is "
            "for the centered vortex at (0.5, 0.5)."
        )

    # ------------------------------------------------------------------
    # Optional CP baseline: cube cases only.
    # ------------------------------------------------------------------
    model_records: List[Dict[str, object]] = []
    run_cp = (not spec.tnn_only) and (not args.no_cp)
    if run_cp:
        cp_grid = torch.linspace(0.0, 1.0, args.grid_n, device=device, dtype=dtype)
        cp_target = state_tensor(cp_grid, exact)
        print(f"\n--- Componentwise CP, rank {args.cp_rank} per component ---")
        cp_start = time.perf_counter()
        cp_model = fit_componentwise_cp(
            cp_target,
            cp_grid,
            args.cp_rank,
            restarts=args.cp_restarts,
            max_sweeps=args.cp_max_sweeps,
            tol=args.cp_tol,
            reg=args.cp_reg,
            seed=args.seed,
            verbose=args.print_every > 0,
        )
        cp_seconds = time.perf_counter() - cp_start
        save_componentwise_cp(
            cp_model,
            args.output_dir / f"{args.state}_componentwise_cp_rank{args.cp_rank}.npz",
        )
        model_records.append(
            {
                "name": f"CP{args.cp_rank}",
                "model": cp_model,
                "parameters": cp_model.parameter_count,
                "stage_seconds": cp_seconds,
                "cumulative_seconds": cp_seconds,
                "factor_normalization_max_error": "",
            }
        )

    # ------------------------------------------------------------------
    # TNN training with anisotropic product rules.
    # ------------------------------------------------------------------
    train_nodes, train_weights = build_axis_rules(
        train_orders,
        spec.bounds,
        mode=train_mode,
        subintervals=train_subintervals,
        device=device,
        dtype=dtype,
    )
    factor_nodes, factor_weights = build_axis_rules(
        factor_orders, spec.bounds, device=device, dtype=dtype
    )

    marginal_targets: MarginalGradientTargets | None = None
    if args.loss_function == "default":
        train_exact: Tensor | None = state_tensor(train_nodes, exact)
        train_nodes_arg = train_nodes
        train_weights_arg = train_weights
    else:
        # Cube-only backward-compatible path: one common rule in all directions.
        if not (train_orders[0] == train_orders[1] == train_orders[2]):
            raise ValueError("gradM requires equal train orders in the current implementation")
        common_nodes, common_weights = train_nodes[0], train_weights[0]
        train_exact = None
        marginal_targets = build_marginal_gradient_targets(
            exact,
            common_nodes,
            common_weights,
            chunk_size=args.gradm_target_chunk,
        )
        train_nodes_arg = common_nodes
        train_weights_arg = common_weights

    current_model: ComponentwiseTNN | None = None
    cumulative_seconds = 0.0
    for rank in tnn_ranks:
        is_first_rank = current_model is None
        if current_model is None:
            print(f"\n--- Componentwise TNN rank {rank} ---")
            current_model = ComponentwiseTNN(
                rank,
                args.tnn_width,
                args.tnn_depth,
                z_invariant=z_invariant,
                dominant_constant_mz=spec.dominant_constant_mz,
                activation=args.activation,
                hidden_widths=tnn_hidden_widths,
            ).to(device=device, dtype=dtype)
            initialized_from = None
        else:
            print(f"\n--- Expand TNN rank {current_model.rank} -> {rank} ---")
            initialized_from = current_model.rank
            current_model = expand_tnn_rank(current_model, rank)

        if args.state == "homo_film":
            initialize_homogeneous_tnn(current_model)

        if args.track_knots and is_first_rank and args.knot_init == "uniform":
            initialize_uniform_knots(current_model, spec.bounds)

        current_model.set_factor_normalization_rule(factor_nodes, factor_weights)

        knot_recorder = None
        if args.track_knots:
            knot_recorder = KnotRecorder(
                output_dir=args.output_dir / f"knot_tracking_rank{rank}",
                bounds=spec.bounds,
                core_radius=args.vortex_core_radius,
                fixed_epochs=knot_fixed_epochs,
                loss_drop=args.knot_loss_drop,
                min_epoch_gap=args.knot_min_epoch_gap,
                topk_per_factor=args.knot_topk_per_factor,
                vortex_center=(args.vortex_center_x, args.vortex_center_y),
                save_model_snapshots=args.knot_save_model_snapshots,
            )

        info = train_tnn_deterministic(
            current_model,
            train_nodes_arg,
            train_weights_arg,
            train_exact,
            marginal_targets,
            loss_function=args.loss_function,
            gradm_mean_weight=args.gradm_mean_weight,
            adam_epochs=args.adam_epochs,
            adam_lr=args.adam_lr,
            lbfgs_iters=args.lbfgs_iters,
            print_every=args.print_every,
            progress_callback=knot_recorder,
        )
        if knot_recorder is not None:
            knot_recorder.capture_final(
                current_model,
                loss=float(info["best_training_loss"]),
                adam_epochs=args.adam_epochs,
            )
            knot_recorder.finalize()
            print(f"      knot figures: {(knot_recorder.output_dir / 'knot_partition_evolution.png').resolve()}")
        cumulative_seconds += info["training_seconds"]
        model_records.append(
            {
                "name": f"TNN{rank}",
                "model": current_model,
                "parameters": current_model.parameter_count,
                "stage_seconds": info["training_seconds"],
                "cumulative_seconds": cumulative_seconds,
                "factor_normalization_max_error": info["factor_normalization_max_error"],
            }
        )

    # ------------------------------------------------------------------
    # Independent random-point magnetization errors in the actual box.
    # ------------------------------------------------------------------
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 999)
    random_points = random_points_in_box(
        args.test_points,
        spec.bounds,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    exact_random = exact(random_points)
    print("\n=== Independent-point magnetization errors ===")
    for record in model_records:
        with torch.no_grad():
            predicted = record["model"].evaluate(random_points) if str(record["name"]).startswith("CP") else record["model"](random_points)
        metric = point_metrics(predicted, exact_random)
        record["metrics"] = metric
        print(
            f"{record['name']:>8s}: rel_L2={metric['relative_L2_m']:.6e}, "
            f"mx={metric['relative_L2_mx']:.3e}, my={metric['relative_L2_my']:.3e}, "
            f"mz={metric['relative_L2_mz']:.3e}, params={record['parameters']}"
        )

    # ------------------------------------------------------------------
    # Optional target-quadrature sweep.  The representation is constructed
    # only once above and then reused for all (m,k) target rules.  This isolates
    # target quadrature error from TNN optimization / CP fitting variability.
    # ------------------------------------------------------------------
    if target_quadrature_sweep:
        sog = build_sog(args.sog_h, args.sog_K, device=device, dtype=dtype)
        sweep_rows: List[Dict[str, object]] = []
        print(
            f"\n=== Target quadrature sweep: Common SOG h={args.sog_h}, "
            f"K={args.sog_K}, terms={sog.terms} ==="
        )

        for m, k in target_quadrature_sweep:
            sweep_subintervals: Counts3 = (m, m, m)
            sweep_orders: Orders3 = (k, k, k)
            sweep_nodes, sweep_weights = build_axis_rules(
                sweep_orders,
                spec.bounds,
                mode="composite",
                subintervals=sweep_subintervals,
                device=device,
                dtype=dtype,
            )
            points_1d = m * k
            total_points = points_1d ** 3
            print(
                f"\n--- target (m,k)=({m},{k}), points/direction={points_1d}, "
                f"total={total_points} ---"
            )

            for record in model_records:
                name = str(record["name"])
                model = record["model"]

                # Reset the CUDA peak-memory counter after training/fitting so the
                # reported peak belongs to this SOG target evaluation rather than
                # to the optimization stage.  The allocated baseline still contains
                # the model itself, which is useful when comparing CP and TNN.
                gpu_before_mb = ""
                gpu_peak_mb = ""
                gpu_peak_increment_mb = ""
                gpu_peak_reserved_mb = ""
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    gpu_before = torch.cuda.memory_allocated(device)
                    gpu_before_mb = gpu_before / (1024.0 ** 2)

                start_eval = time.perf_counter()
                potential, face_potentials = sog_potential_and_faces(
                    model,
                    sweep_nodes,
                    sog,
                    conv_gauss_order=args.conv_gauss_order,
                    tail_tolerance=args.conv_tail_tol,
                    domain_bounds=spec.bounds,
                )
                energy = demag_energy_from_potential_weak(
                    model,
                    sweep_nodes,
                    sweep_weights,
                    potential,
                    face_potentials,
                    mu0=args.mu0,
                    Ms=args.Ms,
                    domain_bounds=spec.bounds,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                evaluation_seconds = time.perf_counter() - start_eval

                if device.type == "cuda":
                    peak_allocated = torch.cuda.max_memory_allocated(device)
                    peak_reserved = torch.cuda.max_memory_reserved(device)
                    gpu_peak_mb = peak_allocated / (1024.0 ** 2)
                    gpu_peak_increment_mb = max(peak_allocated - gpu_before, 0) / (1024.0 ** 2)
                    gpu_peak_reserved_mb = peak_reserved / (1024.0 ** 2)

                lit_error: float | str = ""
                if literature_energy is not None:
                    lit_error = abs(energy - literature_energy) / abs(literature_energy)
                sog_ref_error: float | str = ""
                if args.sog_reference_energy is not None:
                    sog_ref_error = (
                        abs(energy - args.sog_reference_energy)
                        / abs(args.sog_reference_energy)
                    )

                print(
                    f"{name}: E_d={energy:.12e}, time={evaluation_seconds:.2f}s"
                    + (
                        f", eE(SOG-ref)={sog_ref_error:.6e}"
                        if sog_ref_error != ""
                        else ""
                    )
                    + (
                        f", peak GPU={float(gpu_peak_mb):.1f} MiB"
                        if gpu_peak_mb != ""
                        else ""
                    )
                )

                sweep_rows.append(
                    {
                        "state": args.state,
                        "method": name,
                        "rank_per_component": (
                            args.cp_rank if name.startswith("CP") else int(name[3:])
                        ),
                        "tnn_hidden_widths": (
                            ""
                            if name.startswith("CP")
                            else ",".join(map(str, tnn_hidden_widths))
                        ),
                        "seed": args.seed,
                        "target_gauss_mode": "composite",
                        "M": m,
                        "k": k,
                        "points_1d": points_1d,
                        "total_points": total_points,
                        "demag_energy": energy,
                        "literature_energy": (
                            literature_energy if literature_energy is not None else ""
                        ),
                        "relative_energy_error_vs_literature": lit_error,
                        "sog_reference_energy": (
                            args.sog_reference_energy
                            if args.sog_reference_energy is not None
                            else ""
                        ),
                        "relative_energy_error_vs_sog_reference": sog_ref_error,
                        "evaluation_seconds": evaluation_seconds,
                        "gpu_memory_before_mib": gpu_before_mb,
                        "gpu_peak_allocated_mib": gpu_peak_mb,
                        "gpu_peak_increment_mib": gpu_peak_increment_mb,
                        "gpu_peak_reserved_mib": gpu_peak_reserved_mb,
                        "parameter_count": record["parameters"],
                        "compression_stage_seconds": record["stage_seconds"],
                        "relative_L2_m": record["metrics"]["relative_L2_m"],
                        "sog_h": args.sog_h,
                        "sog_K": args.sog_K,
                        "conv_gauss_order": args.conv_gauss_order,
                    }
                )

                # Release the target-grid tensors before moving to the next rule.
                del potential, face_potentials
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        sweep_csv_path = args.output_dir / "target_quadrature_sweep.csv"
        with sweep_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sweep_rows)

        config = vars(args).copy()
        config["output_dir"] = str(args.output_dir)
        config["device_resolved"] = str(device)
        config["dtype_resolved"] = str(dtype)
        config["domain_bounds"] = spec.bounds
        config["train_gauss_mode_resolved"] = train_mode
        config["train_gauss_subintervals_xyz_resolved"] = train_subintervals
        config["train_gauss_orders_xyz_resolved"] = train_orders
        config["train_gauss_points_1d_xyz_resolved"] = train_points
        config["target_quadrature_sweep_resolved"] = target_quadrature_sweep
        config["factor_norm_gauss_orders_xyz_resolved"] = factor_orders
        config["tnn_ranks_resolved"] = tnn_ranks
        config["tnn_hidden_widths_resolved"] = tnn_hidden_widths
        config["literature_energy"] = literature_energy
        config["literature_reference"] = literature_label
        config["energy_formula"] = (
            "0.5*mu0*Ms^2*(boundary integral U m.n - volume integral U div(m))"
        )
        (args.output_dir / "run_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        print(f"\nTarget quadrature sweep CSV: {sweep_csv_path.resolve()}")
        return

    # ------------------------------------------------------------------
    # SOG potential and weak-form energy on anisotropic target rules.
    # ------------------------------------------------------------------
    sog = build_sog(args.sog_h, args.sog_K, device=device, dtype=dtype)
    target_nodes, target_weights = build_axis_rules(
        target_orders,
        spec.bounds,
        mode=target_mode,
        subintervals=target_subintervals,
        device=device,
        dtype=dtype,
    )
    exact_target = state_tensor(target_nodes, exact)
    demag_results: Dict[str, Dict[str, Tensor]] = {}
    summary_rows: List[Dict[str, object]] = []
    quadrature_rows: List[Dict[str, object]] = []

    print(f"\n=== Common SOG: h={args.sog_h}, K={args.sog_K}, terms={sog.terms} ===")
    for record in model_records:
        name = str(record["name"])
        model = record["model"]
        print(f"\n--- SOG potential and energy for {name} ---")
        start = time.perf_counter()
        magnetization = representation_on_product_grid(model, target_nodes)
        potential, face_potentials = sog_potential_and_faces(
            model,
            target_nodes,
            sog,
            conv_gauss_order=args.conv_gauss_order,
            tail_tolerance=args.conv_tail_tol,
            domain_bounds=spec.bounds,
        )
        energy = demag_energy_from_potential_weak(
            model,
            target_nodes,
            target_weights,
            potential,
            face_potentials,
            mu0=args.mu0,
            Ms=args.Ms,
            domain_bounds=spec.bounds,
        )
        sog_seconds = time.perf_counter() - start
        demag_results[name] = {
            "m": magnetization.detach().cpu(),
            "U": potential.detach().cpu(),
        }
        reference_error = ""
        if literature_energy is not None:
            reference_error = abs(energy - literature_energy) / abs(literature_energy)
        print(
            f"{name}: E_d={energy:.12e}, "
            + (f"rel.err(literature)={reference_error:.6e}, " if reference_error != "" else "")
            + f"SOG time={sog_seconds:.2f}s"
        )
        row = {
            "state": args.state,
            "geometry_Lx_Ly_Lz": "1,1,0.1" if is_film else "1,1,1",
            "method": name,
            "tnn_activation": "" if name.startswith("CP") else args.activation,
            "tnn_depth": "" if name.startswith("CP") else args.tnn_depth,
            "tnn_width": "" if name.startswith("CP") else args.tnn_width,
            "tnn_hidden_widths": (
                "" if name.startswith("CP") else ",".join(map(str, tnn_hidden_widths))
            ),
            "tnn_loss_function": "" if name.startswith("CP") else args.loss_function,
            "rank_per_component": args.cp_rank if name.startswith("CP") else int(name[3:]),
            "parameter_count": record["parameters"],
            "train_gauss_mode": train_mode,
            "train_gauss_subintervals": _compact_xyz(train_subintervals),
            "train_gauss_order": _compact_xyz(train_orders),
            "train_points_1d": _compact_xyz(train_points),
            "train_gauss_subintervals_x": train_subintervals[0],
            "train_gauss_subintervals_y": train_subintervals[1],
            "train_gauss_subintervals_z": train_subintervals[2],
            "train_gauss_order_x": train_orders[0],
            "train_gauss_order_y": train_orders[1],
            "train_gauss_order_z": train_orders[2],
            "train_points_1d_x": train_points[0],
            "train_points_1d_y": train_points[1],
            "train_points_1d_z": train_points[2],
            "target_gauss_mode": target_mode,
            "target_gauss_subintervals": _compact_xyz(target_subintervals),
            "target_gauss_order": _compact_xyz(target_orders),
            "target_points_1d": _compact_xyz(target_points),
            "target_gauss_subintervals_x": target_subintervals[0],
            "target_gauss_subintervals_y": target_subintervals[1],
            "target_gauss_subintervals_z": target_subintervals[2],
            "target_gauss_order_x": target_orders[0],
            "target_gauss_order_y": target_orders[1],
            "target_gauss_order_z": target_orders[2],
            "target_points_1d_x": target_points[0],
            "target_points_1d_y": target_points[1],
            "target_points_1d_z": target_points[2],
            "factor_norm_gauss_order_x": factor_orders[0],
            "factor_norm_gauss_order_y": factor_orders[1],
            "factor_norm_gauss_order_z": factor_orders[2],
            "compression_stage_seconds": record["stage_seconds"],
            "compression_cumulative_seconds": record["cumulative_seconds"],
            "factor_normalization_max_error": record["factor_normalization_max_error"],
            **record["metrics"],
            "demag_energy": energy,
            "vortex_center_x": args.vortex_center_x if args.state in ("vortex_cube", "vortex_film") else "",
            "vortex_center_y": args.vortex_center_y if args.state in ("vortex_cube", "vortex_film") else "",
            "vortex_core_radius": args.vortex_core_radius if args.state in ("vortex_cube", "vortex_film") else "",
            "literature_energy": literature_energy if literature_energy is not None else "",
            "relative_energy_error_vs_literature": reference_error,
            "literature_reference": literature_label or "",
            "sog_seconds": sog_seconds,
        }
        summary_rows.append(row)
        quadrature_rows.append(
            {
                "study": "single",
                "state": args.state,
                "method": name,
                "mode": target_mode,
                "M": _compact_xyz(target_subintervals),
                "k": _compact_xyz(target_orders),
                "points_1d": _compact_xyz(target_points),
                "total_points": int(np.prod(target_points)),
                "Ed": energy,
                "evaluation_seconds": sog_seconds,
                "rank": args.cp_rank if name.startswith("CP") else int(name[3:]),
                "sog_h": args.sog_h,
                "sog_K": args.sog_K,
                "conv_gauss_order": args.conv_gauss_order,
            }
        )

    # Pairwise CP comparisons remain available for cube experiments only.
    cp_records = [r for r in summary_rows if str(r["method"]).startswith("CP")]
    if cp_records:
        cp_name = str(cp_records[0]["method"])
        cp_energy = float(cp_records[0]["demag_energy"])
        cp_result = demag_results[cp_name]
        column_u = f"relative_U_difference_vs_{cp_name}"
        column_e = f"relative_energy_difference_vs_{cp_name}"
        for row in summary_rows:
            name = str(row["method"])
            if name == cp_name:
                row[column_u] = 0.0
                row[column_e] = 0.0
            else:
                row[column_u] = weighted_relative_scalar(
                    demag_results[name]["U"].to(device=device, dtype=dtype),
                    cp_result["U"].to(device=device, dtype=dtype),
                    target_weights,
                )
                row[column_e] = abs(float(row["demag_energy"]) - cp_energy) / max(abs(cp_energy), 1.0e-30)

    center_tag = ""
    if args.state in ("vortex_cube", "vortex_film") and not standard_vortex_center:
        xc = str(args.vortex_center_x).replace("-", "m").replace(".", "p")
        yc = str(args.vortex_center_y).replace("-", "m").replace(".", "p")
        center_tag = f"_xc{xc}_yc{yc}"
    tag = (
        f"{args.state}{center_tag}_tnn{'_'.join(map(str, tnn_ranks))}_"
        f"{args.activation}_{args.loss_function}"
    )
    csv_path = args.output_dir / f"{tag}_sog_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    quadrature_csv_path = args.output_dir / "quadrature_convergence.csv"
    upsert_quadrature_convergence(quadrature_csv_path, quadrature_rows)

    arrays: Dict[str, np.ndarray] = {
        "target_nodes_x": target_nodes[0].cpu().numpy(),
        "target_nodes_y": target_nodes[1].cpu().numpy(),
        "target_nodes_z": target_nodes[2].cpu().numpy(),
        "target_weights_x": target_weights[0].cpu().numpy(),
        "target_weights_y": target_weights[1].cpu().numpy(),
        "target_weights_z": target_weights[2].cpu().numpy(),
        "exact_m": exact_target.cpu().numpy(),
        "sog_alpha": sog.alpha.cpu().numpy(),
        "sog_beta": sog.beta.cpu().numpy(),
    }
    for name, data in demag_results.items():
        safe = name.lower()
        arrays[f"{safe}_m"] = data["m"].numpy()
        arrays[f"{safe}_U"] = data["U"].numpy()
    np.savez_compressed(args.output_dir / f"{tag}_sog_potential.npz", **arrays)

    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    config["device_resolved"] = str(device)
    config["dtype_resolved"] = str(dtype)
    config["domain_bounds"] = spec.bounds
    config["train_gauss_mode_resolved"] = train_mode
    config["train_gauss_subintervals_xyz_resolved"] = train_subintervals
    config["train_gauss_orders_xyz_resolved"] = train_orders
    config["train_gauss_points_1d_xyz_resolved"] = train_points
    config["target_gauss_mode_resolved"] = target_mode
    config["target_gauss_subintervals_xyz_resolved"] = target_subintervals
    config["target_gauss_orders_xyz_resolved"] = target_orders
    config["target_gauss_points_1d_xyz_resolved"] = target_points
    config["factor_norm_gauss_orders_xyz_resolved"] = factor_orders
    config["tnn_ranks_resolved"] = tnn_ranks
    config["tnn_hidden_widths_resolved"] = tnn_hidden_widths
    config["literature_energy"] = literature_energy
    config["literature_reference"] = literature_label
    config["standard_vortex_center"] = standard_vortex_center
    config["energy_formula"] = "0.5*mu0*Ms^2*(boundary integral U m.n - volume integral U div(m))"
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    if args.make_plot:
        nxp, nyp, nzp = (61, 61, 5) if is_film else (41, 41, 41)
        plot_nodes = tuple(
            torch.linspace(a, b, n, device=device, dtype=dtype)
            for (a, b), n in zip(spec.bounds, (nxp, nyp, nzp))
        )
        plot_exact = state_tensor(plot_nodes, exact)
        plot_results: Dict[str, Dict[str, Tensor]] = {}
        for record in model_records:
            name, model = str(record["name"]), record["model"]
            m_plot = representation_on_product_grid(model, plot_nodes)
            U_plot = sog_potential_on_product_grid(
                model,
                plot_nodes,
                sog,
                conv_gauss_order=args.conv_gauss_order,
                tail_tolerance=args.conv_tail_tol,
                domain_bounds=spec.bounds,
            )
            plot_results[name] = {"m": m_plot.cpu(), "U": U_plot.cpu()}
        make_plots(
            args.output_dir,
            args.state,
            tuple(node.cpu() for node in plot_nodes),
            plot_exact.cpu(),
            plot_results,
        )

    print(f"\nSummary CSV: {csv_path.resolve()}")
    print(f"Quadrature convergence CSV: {quadrature_csv_path.resolve()}")


if __name__ == "__main__":
    main()
