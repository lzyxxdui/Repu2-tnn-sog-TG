#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analytic magnetization states and geometry metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import torch
from torch import Tensor

Bounds3D = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]


@dataclass(frozen=True)
class CaseSpec:
    name: str
    bounds: Bounds3D
    tnn_only: bool
    z_invariant: bool
    dominant_constant_mz: bool
    literature_energy: float | None
    literature_label: str | None
    default_train_orders: Tuple[int, int, int]
    default_target_orders: Tuple[int, int, int]
    default_tnn_ranks: str


def case_spec(name: str) -> CaseSpec:
    cube = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    film = ((0.0, 1.0), (0.0, 1.0), (0.0, 0.1))
    if name == "flower":
        return CaseSpec(
            name=name,
            bounds=cube,
            tnn_only=False,
            z_invariant=False,
            dominant_constant_mz=True,
            literature_energy=1.528e-1,
            literature_label="Abert et al. DM 40x40x40 (Table 4)",
            default_train_orders=(18, 18, 18),
            default_target_orders=(18, 18, 18),
            default_tnn_ranks="20,30",
        )
    if name == "vortex_cube":
        return CaseSpec(
            name=name,
            bounds=cube,
            tnn_only=False,
            z_invariant=True,
            dominant_constant_mz=False,
            literature_energy=2.189e-2,
            literature_label="Abert et al. DM 40x40x40 (Table 5)",
            default_train_orders=(32, 32, 8),
            default_target_orders=(28, 28, 8),
            default_tnn_ranks="20,30",
        )
    if name == "homo_film":
        return CaseSpec(
            name=name,
            bounds=film,
            tnn_only=True,
            z_invariant=True,
            dominant_constant_mz=True,
            literature_energy=4.025e-2,
            literature_label="Abert et al. DM 80x80x8 (Table 6)",
            default_train_orders=(32, 32, 8),
            default_target_orders=(40, 40, 8),
            default_tnn_ranks="1",
        )
    if name == "vortex_film":
        return CaseSpec(
            name=name,
            bounds=film,
            tnn_only=True,
            z_invariant=True,
            dominant_constant_mz=False,
            literature_energy=1.569e-3,
            literature_label="Abert et al. DM 80x80x8 (Table 7)",
            default_train_orders=(32, 32, 8),
            default_target_orders=(40, 40, 8),
            default_tnn_ranks="20,30",
        )
    raise ValueError(f"unsupported state: {name}")


# =============================================================================
# Exact magnetization states
# =============================================================================

def homogeneous_out_of_plane(xyz: Tensor) -> Tensor:
    """Uniform out-of-plane state m=(0,0,1)."""
    if xyz.shape[-1] != 3:
        raise ValueError("xyz must have last dimension 3")
    out = torch.zeros_like(xyz)
    out[..., 2] = 1.0
    return out


def flower_state(
    xyz: Tensor,
    *,
    a: float = 1.0,
    b: float = 2.0,
    c: float = 1.0,
) -> Tensor:
    """Normalized flower state on the unit cube."""
    if xyz.shape[-1] != 3:
        raise ValueError("xyz must have last dimension 3")
    X = xyz[..., 0] - 0.5
    Y = xyz[..., 1] - 0.5
    Z = xyz[..., 2] - 0.5
    raw = torch.stack(
        (X * Z / a, Y * Z / c + (Y**3) * (Z**3) / (b**3), torch.ones_like(X)),
        dim=-1,
    )
    return raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True)


def vortex_state(
    xyz: Tensor,
    *,
    core_radius: float = 0.14,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> Tensor:
    r"""z-invariant vortex state with a user-selectable center in the x-y plane."""
    if xyz.shape[-1] != 3:
        raise ValueError("xyz must have last dimension 3")
    if core_radius <= 0.0:
        raise ValueError("core_radius must be positive")
    X = xyz[..., 0] - center_x
    Y = xyz[..., 1] - center_y
    r2 = X * X + Y * Y
    rc2 = core_radius * core_radius
    numerator = -torch.expm1(-4.0 * r2 / rc2)
    eps = torch.finfo(xyz.dtype).eps
    swirl = torch.sqrt(numerator / torch.clamp(r2, min=eps))
    swirl = torch.where(
        r2 > 64.0 * eps,
        swirl,
        torch.full_like(swirl, 2.0 / core_radius),
    )
    mx = Y * swirl
    my = -X * swirl
    mz = torch.exp(-2.0 * r2 / rc2)
    return torch.stack((mx, my, mz), dim=-1)


def state_function(args) -> Callable[[Tensor], Tensor]:
    """Resolve the command-line case name to its analytic magnetization."""
    if args.state == "flower":
        return lambda xyz: flower_state(
            xyz, a=args.flower_a, b=args.flower_b, c=args.flower_c
        )
    if args.state in ("vortex_cube", "vortex_film"):
        return lambda xyz: vortex_state(
            xyz,
            core_radius=args.vortex_core_radius,
            center_x=args.vortex_center_x,
            center_y=args.vortex_center_y,
        )
    if args.state == "homo_film":
        return homogeneous_out_of_plane
    raise ValueError(f"unsupported state: {args.state}")


def _as_nodes_xyz(nodes_xyz: Sequence[Tensor] | Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    if isinstance(nodes_xyz, Tensor):
        return nodes_xyz, nodes_xyz, nodes_xyz
    if len(nodes_xyz) != 3:
        raise ValueError("nodes_xyz must contain three one-dimensional node vectors")
    return nodes_xyz[0], nodes_xyz[1], nodes_xyz[2]


def state_tensor(
    nodes_xyz: Sequence[Tensor] | Tensor,
    exact: Callable[[Tensor], Tensor],
) -> Tensor:
    x, y, z = _as_nodes_xyz(nodes_xyz)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    xyz = torch.stack((X, Y, Z), dim=-1)
    return exact(xyz).permute(3, 0, 1, 2).contiguous()


def tensor_grid_points(nodes_xyz: Sequence[Tensor] | Tensor) -> Tensor:
    x, y, z = _as_nodes_xyz(nodes_xyz)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    return torch.stack((X, Y, Z), dim=-1).reshape(-1, 3)


def random_points_in_box(
    n: int,
    bounds: Bounds3D,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    unit = torch.rand((n, 3), generator=generator, device=device, dtype=dtype)
    lower = torch.tensor([b[0] for b in bounds], device=device, dtype=dtype)
    upper = torch.tensor([b[1] for b in bounds], device=device, dtype=dtype)
    return lower + (upper - lower) * unit
