#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Uncompressed dense tensor-grid magnetization with the existing SOG operator.

This module is intentionally independent of the TNN/CP representations.  The
magnetization is sampled from the analytic state on a fixed uniform tensor grid
(including the boundary) and represented by tensor-product piecewise-linear
hat functions.  No low-rank decomposition is performed.

The SOG Gaussian convolutions are still integrated with the same local
``t=sqrt(alpha)(y-x)`` change of variables used by the TNN code.  Therefore
very narrow Gaussian terms are not approximated by simply sampling the kernel
on the source grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from torch import Tensor

from quadrature import tensor_weights_3d
from sog import SOG, narrow_gaussian_convolution_01

Bounds3D = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
Counts3 = Tuple[int, int, int]


def _validate_counts(counts: Sequence[int]) -> Counts3:
    if len(counts) != 3:
        raise ValueError("source grid must have three directional counts")
    out = tuple(int(n) for n in counts)
    if any(n < 2 for n in out):
        raise ValueError("each source-grid direction needs at least two nodes")
    return out  # type: ignore[return-value]


def uniform_nodes(
    bounds: Sequence[Tuple[float, float]],
    counts: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor, Tensor]:
    counts3 = _validate_counts(counts)
    nodes: List[Tensor] = []
    for (a, b), n in zip(bounds, counts3):
        if not b > a:
            raise ValueError("invalid source-grid interval")
        nodes.append(torch.linspace(a, b, n, device=device, dtype=dtype))
    return tuple(nodes)  # type: ignore[return-value]


def _tensor_grid_values(
    nodes_xyz: Sequence[Tensor],
    exact_state: Callable[[Tensor], Tensor],
) -> Tensor:
    X, Y, Z = torch.meshgrid(*nodes_xyz, indexing="ij")
    xyz = torch.stack((X, Y, Z), dim=-1)
    values = exact_state(xyz)
    if values.shape != xyz.shape:
        raise ValueError(
            "exact_state must return the same leading grid shape and three components"
        )
    return values.permute(3, 0, 1, 2).contiguous()


def uniform_hat_basis_value_derivative(
    x: Tensor,
    n: int,
    lower: float,
    upper: float,
) -> Tuple[Tensor, Tensor]:
    """Piecewise-linear nodal basis values and d/dx values on a uniform grid.

    Returns matrices with shape ``x.shape + (n,)``.  At an interior grid node
    the derivative is the derivative from the cell on the right (except at the
    upper boundary, where the last cell is used).  Gauss nodes never coincide
    with interior uniform-grid knots in the intended experiments, so this
    one-sided convention does not affect the quadrature results.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if not upper > lower:
        raise ValueError("upper must exceed lower")

    scale = (n - 1) / (upper - lower)
    s = torch.clamp((x - lower) * scale, 0.0, float(n - 1))
    left = torch.floor(s).to(torch.long).clamp(0, n - 2)
    t = s - left.to(s.dtype)

    shape = x.shape + (n,)
    value = torch.zeros(shape, device=x.device, dtype=x.dtype)
    derivative = torch.zeros_like(value)

    left_idx = left.unsqueeze(-1)
    right_idx = (left + 1).unsqueeze(-1)
    value.scatter_add_(-1, left_idx, (1.0 - t).unsqueeze(-1))
    value.scatter_add_(-1, right_idx, t.unsqueeze(-1))

    slope = torch.full_like(t, scale)
    derivative.scatter_add_(-1, left_idx, (-slope).unsqueeze(-1))
    derivative.scatter_add_(-1, right_idx, slope.unsqueeze(-1))
    return value, derivative


@dataclass
class DenseGridMagnetization:
    """Full (uncompressed) magnetization tensor on a uniform source grid."""

    values: Tensor  # [3, Nx, Ny, Nz]
    nodes_xyz: Tuple[Tensor, Tensor, Tensor]
    bounds: Bounds3D

    @classmethod
    def from_exact_state(
        cls,
        exact_state: Callable[[Tensor], Tensor],
        bounds: Bounds3D,
        counts: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "DenseGridMagnetization":
        nodes = uniform_nodes(bounds, counts, device=device, dtype=dtype)
        with torch.no_grad():
            values = _tensor_grid_values(nodes, exact_state)
        return cls(values=values, nodes_xyz=nodes, bounds=bounds)

    @property
    def counts(self) -> Counts3:
        return tuple(int(x.numel()) for x in self.nodes_xyz)  # type: ignore[return-value]

    @property
    def parameter_count(self) -> int:
        return int(self.values.numel())

    def hat_factor(self, dim: int) -> Callable[[Tensor], Tensor]:
        n = self.counts[dim]
        lower, upper = self.bounds[dim]

        def factor(y: Tensor) -> Tensor:
            # narrow_gaussian_convolution_01 supplies (...,1) inputs.
            value, _ = uniform_hat_basis_value_derivative(
                y[..., 0], n, lower, upper
            )
            return value

        return factor

    @torch.no_grad()
    def component_on_product_grid(
        self,
        q: int,
        nodes_xyz: Sequence[Tensor],
        *,
        derivative_dim: int | None = None,
    ) -> Tensor:
        if q not in (0, 1, 2):
            raise ValueError("q must be 0, 1, or 2")
        if len(nodes_xyz) != 3:
            raise ValueError("need x/y/z target node vectors")

        matrices: List[Tensor] = []
        for dim, x in enumerate(nodes_xyz):
            n = self.counts[dim]
            lower, upper = self.bounds[dim]
            value, derivative = uniform_hat_basis_value_derivative(
                x, n, lower, upper
            )
            matrices.append(derivative if dim == derivative_dim else value)

        # Apply the three interpolation matrices successively.  This avoids
        # constructing a rank decomposition of the dense magnetization tensor.
        data = self.values[q]
        tmp = torch.einsum("ai,ijk->ajk", matrices[0], data)
        tmp = torch.einsum("bj,ajk->abk", matrices[1], tmp)
        return torch.einsum("ck,abk->abc", matrices[2], tmp)

    @torch.no_grad()
    def on_product_grid(self, nodes_xyz: Sequence[Tensor]) -> Tensor:
        return torch.stack(
            [self.component_on_product_grid(q, nodes_xyz) for q in range(3)],
            dim=0,
        )


def _convolution_cache_for_targets(
    dense_m: DenseGridMagnetization,
    target_sets: Sequence[Dict[str, Tensor]],
    sog: SOG,
    *,
    conv_gauss_order: int,
    tail_tolerance: float,
    alpha_chunk: int,
    x_chunk: int,
) -> List[Dict[str, Tuple[Tensor, Tensor]]]:
    """cache[dim][key] = (I0, I1) for every 1D nodal hat basis."""
    cache: List[Dict[str, Tuple[Tensor, Tensor]]] = []
    for dim in range(3):
        factor = dense_m.hat_factor(dim)
        lower, upper = dense_m.bounds[dim]
        dim_cache: Dict[str, Tuple[Tensor, Tensor]] = {}
        for key, targets in target_sets[dim].items():
            dim_cache[key] = narrow_gaussian_convolution_01(
                factor,
                targets,
                sog.alpha,
                gauss_order=conv_gauss_order,
                tail_tolerance=tail_tolerance,
                lower=lower,
                upper=upper,
                alpha_chunk=alpha_chunk,
                x_chunk=x_chunk,
            )
        cache.append(dim_cache)
    return cache


def _apply_three_batched_matrices(
    Ax: Tensor,
    Ay: Tensor,
    Az: Tensor,
    data: Tensor,
) -> Tensor:
    """Return [B,Tx,Ty,Tz] from batched matrices and dense [Sx,Sy,Sz] data."""
    # Distinct einsum labels: l=batch, a/b/c=target, i/j/k=source.
    tmp = torch.einsum("lai,ijk->lajk", Ax, data)
    tmp = torch.einsum("lbj,lajk->labk", Ay, tmp)
    return torch.einsum("lck,labk->labc", Az, tmp)


def _assemble_dense_sog_potential(
    dense_m: DenseGridMagnetization,
    cache: Sequence[Dict[str, Tuple[Tensor, Tensor]]],
    keys_xyz: Sequence[str],
    sog: SOG,
    *,
    contraction_alpha_chunk: int,
) -> Tensor:
    if contraction_alpha_chunk < 1:
        raise ValueError("contraction_alpha_chunk must be positive")

    target_shape = tuple(
        int(cache[d][keys_xyz[d]][0].shape[1]) for d in range(3)
    )
    potential = torch.zeros(
        target_shape,
        device=dense_m.values.device,
        dtype=dense_m.values.dtype,
    )

    L = sog.terms
    for l0 in range(0, L, contraction_alpha_chunk):
        l1 = min(l0 + contraction_alpha_chunk, L)
        beta = sog.beta[l0:l1]
        one_d = [cache[d][keys_xyz[d]] for d in range(3)]

        for q in range(3):
            matrices: List[Tensor] = []
            for dim in range(3):
                I0, I1 = one_d[dim]
                matrices.append(-I1[l0:l1] if dim == q else I0[l0:l1])
            block = _apply_three_batched_matrices(
                matrices[0], matrices[1], matrices[2], dense_m.values[q]
            )
            potential.add_(
                torch.sum(beta[:, None, None, None] * block, dim=0)
            )
    return potential


@torch.no_grad()
def dense_grid_sog_potential_and_faces(
    dense_m: DenseGridMagnetization,
    interior_nodes_xyz: Sequence[Tensor],
    sog: SOG,
    *,
    conv_gauss_order: int,
    tail_tolerance: float,
    alpha_chunk: int = 8,
    x_chunk: int = 64,
    contraction_alpha_chunk: int = 4,
) -> Tuple[Tensor, Dict[Tuple[int, int], Tensor]]:
    """Potential on the target tensor grid and on its six box faces."""
    if len(interior_nodes_xyz) != 3:
        raise ValueError("need three target node vectors")

    target_sets: List[Dict[str, Tensor]] = []
    for dim, nodes in enumerate(interior_nodes_xyz):
        lower, upper = dense_m.bounds[dim]
        target_sets.append(
            {
                "interior": nodes,
                "lower": torch.tensor(
                    [lower], device=nodes.device, dtype=nodes.dtype
                ),
                "upper": torch.tensor(
                    [upper], device=nodes.device, dtype=nodes.dtype
                ),
            }
        )

    cache = _convolution_cache_for_targets(
        dense_m,
        target_sets,
        sog,
        conv_gauss_order=conv_gauss_order,
        tail_tolerance=tail_tolerance,
        alpha_chunk=alpha_chunk,
        x_chunk=x_chunk,
    )

    interior = _assemble_dense_sog_potential(
        dense_m,
        cache,
        ("interior", "interior", "interior"),
        sog,
        contraction_alpha_chunk=contraction_alpha_chunk,
    )
    faces: Dict[Tuple[int, int], Tensor] = {}
    for q in range(3):
        for side, key in ((0, "lower"), (1, "upper")):
            keys = ["interior", "interior", "interior"]
            keys[q] = key
            faces[(q, side)] = _assemble_dense_sog_potential(
                dense_m,
                cache,
                keys,
                sog,
                contraction_alpha_chunk=contraction_alpha_chunk,
            )
    return interior, faces


@torch.no_grad()
def dense_grid_demag_energy_from_potential_weak(
    dense_m: DenseGridMagnetization,
    interior_nodes_xyz: Sequence[Tensor],
    interior_weights_xyz: Sequence[Tensor],
    interior_potential: Tensor,
    face_potentials: Dict[Tuple[int, int], Tensor],
    *,
    mu0: float,
    Ms: float,
) -> float:
    r"""Weak-form energy of the same trilinear dense-grid magnetization.

    E_d/(mu0 Ms^2) = 1/2 [ int_Gamma U (m.n) dS
                            - int_Omega U div(m) dx ].
    """
    if len(interior_nodes_xyz) != 3 or len(interior_weights_xyz) != 3:
        raise ValueError("need three target rules")

    w3 = tensor_weights_3d(interior_weights_xyz)
    divergence = torch.zeros_like(interior_potential)
    for q in range(3):
        divergence.add_(
            dense_m.component_on_product_grid(
                q, interior_nodes_xyz, derivative_dim=q
            )
        )
    volume = torch.sum(w3 * interior_potential * divergence)

    boundary = torch.zeros(
        (), device=interior_potential.device, dtype=interior_potential.dtype
    )
    for q in range(3):
        tangential_dims = [d for d in range(3) if d != q]
        w2 = torch.einsum(
            "i,j->ij",
            interior_weights_xyz[tangential_dims[0]],
            interior_weights_xyz[tangential_dims[1]],
        )
        lower, upper = dense_m.bounds[q]
        fixed = {
            0: torch.tensor(
                [lower],
                device=interior_nodes_xyz[q].device,
                dtype=interior_nodes_xyz[q].dtype,
            ),
            1: torch.tensor(
                [upper],
                device=interior_nodes_xyz[q].device,
                dtype=interior_nodes_xyz[q].dtype,
            ),
        }
        for side, normal_sign in ((0, -1.0), (1, 1.0)):
            face_nodes = list(interior_nodes_xyz)
            face_nodes[q] = fixed[side]
            m_face = dense_m.component_on_product_grid(q, face_nodes).squeeze(q)
            U_face = face_potentials[(q, side)].squeeze(q)
            boundary.add_(normal_sign * torch.sum(w2 * U_face * m_face))

    energy = 0.5 * mu0 * Ms**2 * (boundary - volume)
    return float(energy.cpu())
