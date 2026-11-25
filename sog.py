#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOG approximation of the Newton kernel, potential, and demagnetizing energy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from cp import ComponentwiseCP
from quadrature import tensor_weights_3d
from tnn import ComponentwiseTNN, ConstantFactor, OneDimFactorNet

# =============================================================================
# SOG and narrow Gaussian convolutions
# =============================================================================

@dataclass
class SOG:
    h: float
    K: int
    alpha: Tensor
    beta: Tensor

    @property
    def terms(self) -> int:
        return int(self.alpha.numel())


def build_sog(
    h: float,
    K: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> SOG:
    if h <= 0.0 or K < 1:
        raise ValueError("SOG requires h>0 and K>=1")
    k = torch.arange(-K, K + 1, device=device, dtype=dtype)
    s = h * k
    alpha = torch.exp(2.0 * s)
    beta = h * torch.exp(s) / (2.0 * math.pi ** 1.5)
    return SOG(h=h, K=K, alpha=alpha, beta=beta)


def narrow_gaussian_convolution_01(
    factor: Callable[[Tensor], Tensor],
    x: Tensor,
    alpha: Tensor,
    *,
    gauss_order: int,
    tail_tolerance: float,
    lower: float = 0.0,
    upper: float = 1.0,
    alpha_chunk: int = 32,
    x_chunk: int = 256,
) -> Tuple[Tensor, Tensor]:
    """Return I and dI/dx with shape [L,Nx,R].

    Only the first derivative is needed for the potential.  The demagnetizing
    energy is then evaluated by the weak identity

        E_d = 1/2 [ int_Gamma U (m.n) dS - int_Omega U div(m) dx ],

    which avoids the numerically delicate second derivative of extremely
    narrow Gaussians.
    """
    if not upper > lower:
        raise ValueError("Require upper>lower in the convolution interval")
    x = x.reshape(-1)
    alpha = alpha.reshape(-1)
    g, w = np.polynomial.legendre.leggauss(gauss_order)
    g = torch.as_tensor(g, device=x.device, dtype=x.dtype)
    w = torch.as_tensor(w, device=x.device, dtype=x.dtype)
    T = math.sqrt(-math.log(tail_tolerance))

    value_blocks: List[Tensor] = []
    first_blocks: List[Tensor] = []

    for a0 in range(0, alpha.numel(), alpha_chunk):
        a = alpha[a0 : a0 + alpha_chunk]
        sqrt_a = torch.sqrt(a)
        values_for_a: List[Tensor] = []
        first_for_a: List[Tensor] = []

        for x0 in range(0, x.numel(), x_chunk):
            xx = x[x0 : x0 + x_chunk]
            left = torch.maximum(
                sqrt_a[:, None] * (lower - xx[None, :]),
                torch.full(
                    (a.numel(), xx.numel()),
                    -T,
                    device=x.device,
                    dtype=x.dtype,
                ),
            )
            right = torch.minimum(
                sqrt_a[:, None] * (upper - xx[None, :]),
                torch.full(
                    (a.numel(), xx.numel()),
                    T,
                    device=x.device,
                    dtype=x.dtype,
                ),
            )
            half = 0.5 * (right - left)
            mid = 0.5 * (right + left)
            t = mid[..., None] + half[..., None] * g
            y = xx[None, :, None] + t / sqrt_a[:, None, None]
            y = torch.clamp(y, lower, upper)

            f = factor(y[..., None])
            if f.shape[:-1] != y.shape:
                raise ValueError(
                    "factor must return y.shape + (rank,), "
                    f"got y={tuple(y.shape)}, f={tuple(f.shape)}"
                )

            base = half[..., None] * w * torch.exp(-t * t)
            value = torch.sum(
                (base / sqrt_a[:, None, None])[..., None] * f,
                dim=-2,
            )
            first = torch.sum(
                (2.0 * base * t)[..., None] * f,
                dim=-2,
            )
            values_for_a.append(value)
            first_for_a.append(first)

        value_blocks.append(torch.cat(values_for_a, dim=1))
        first_blocks.append(torch.cat(first_for_a, dim=1))

    return torch.cat(value_blocks, dim=0), torch.cat(first_blocks, dim=0)


def representation_component_data(
    representation,
    q: int,
) -> Tuple[Tensor, List[Callable[[Tensor], Tensor]]]:
    """Return coefficient and x/y/z factor callables for CP or TNN."""
    if isinstance(representation, ComponentwiseCP):
        component = representation.components[q]
        coefficient = component.coefficient

        def make_factor(dim: int):
            def factor(y: Tensor) -> Tensor:
                return component.factor(dim, y[..., 0])
            return factor

        factors = [make_factor(dim) for dim in range(3)]
        return coefficient, factors

    if isinstance(representation, ComponentwiseTNN):
        component = representation.components[q]
        nets = (component.phi_x, component.phi_y, component.phi_z)

        def make_factor(net: nn.Module):
            def factor(y: Tensor) -> Tensor:
                return net(y)
            return factor

        return component.coefficients, [make_factor(net) for net in nets]

    raise TypeError(f"unsupported representation type {type(representation)}")


def cp_factor_value_derivative(
    factor: Tensor,
    x: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Piecewise-linear CP factor and its one-sided cell derivative."""
    n = factor.shape[0]
    s = torch.clamp(x, 0.0, 1.0) * (n - 1)
    left = torch.floor(s).to(torch.long).clamp(0, n - 2)
    t = (s - left.to(s.dtype)).unsqueeze(-1)
    value = (1.0 - t) * factor[left] + t * factor[left + 1]
    derivative = (n - 1) * (factor[left + 1] - factor[left])
    return value, derivative


def tnn_factor_value_derivative(
    net: nn.Module,
    x: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Evaluate all TNN channels and d/dx channels by the 1D chain rule."""
    if not isinstance(net, OneDimFactorNet):
        raise TypeError(type(net))
    with torch.no_grad():
        value, derivative = net.forward_with_derivative(x.reshape(-1, 1))
    return value, derivative


def representation_factor_value_derivative(
    representation,
    q: int,
    dim: int,
    nodes: Tensor,
) -> Tuple[Tensor, Tensor]:
    if isinstance(representation, ComponentwiseCP):
        component = representation.components[q]
        table = (component.A, component.B, component.C)[dim]
        return cp_factor_value_derivative(table, nodes)
    if isinstance(representation, ComponentwiseTNN):
        component = representation.components[q]
        net = (component.phi_x, component.phi_y, component.phi_z)[dim]
        if isinstance(net, ConstantFactor):
            value = net(nodes.reshape(-1, 1))
            return value, torch.zeros_like(value)
        return tnn_factor_value_derivative(net, nodes)
    raise TypeError(type(representation))


@torch.no_grad()
def representation_component_on_product_grid(
    representation,
    q: int,
    nodes_xyz: Sequence[Tensor],
    *,
    derivative_dim: int | None = None,
) -> Tensor:
    """Evaluate m_q or partial_{derivative_dim} m_q on a product grid."""
    coefficient, _ = representation_component_data(representation, q)
    factors = []
    for dim, nodes in enumerate(nodes_xyz):
        value, derivative = representation_factor_value_derivative(
            representation, q, dim, nodes
        )
        factors.append(derivative if dim == derivative_dim else value)
    return torch.einsum(
        "r,ir,jr,kr->ijk",
        coefficient,
        factors[0],
        factors[1],
        factors[2],
    )


@torch.no_grad()
def representation_on_product_grid(
    representation,
    nodes_xyz: Sequence[Tensor],
) -> Tensor:
    return torch.stack(
        [
            representation_component_on_product_grid(
                representation, q, nodes_xyz
            )
            for q in range(3)
        ],
        dim=0,
    )


@torch.no_grad()
def sog_potential_on_product_grid(
    representation,
    nodes_xyz: Sequence[Tensor],
    sog: SOG,
    *,
    conv_gauss_order: int,
    tail_tolerance: float,
    domain_bounds: Sequence[Tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
) -> Tensor:
    """Compute free-space scalar potential on X x Y x Z target nodes."""
    nx, ny, nz = (int(nodes.numel()) for nodes in nodes_xyz)
    potential = torch.zeros(
        (nx, ny, nz),
        device=nodes_xyz[0].device,
        dtype=nodes_xyz[0].dtype,
    )

    for q in range(3):
        coefficient, factor_callables = representation_component_data(
            representation, q
        )
        convolutions = [
            narrow_gaussian_convolution_01(
                factor,
                nodes_xyz[dim],
                sog.alpha,
                gauss_order=conv_gauss_order,
                tail_tolerance=tail_tolerance,
                lower=domain_bounds[dim][0],
                upper=domain_bounds[dim][1],
            )
            for dim, factor in enumerate(factor_callables)
        ]
        weighted_coefficient = sog.beta[:, None] * coefficient[None, :]
        factors = []
        for dim in range(3):
            I0, I1 = convolutions[dim]
            # int m_q(y) partial_{y_q} G(x-y) dy = -I1_q * I0_others.
            factors.append(-I1 if dim == q else I0)
        potential.add_(
            torch.einsum(
                "lr,lir,ljr,lkr->ijk",
                weighted_coefficient,
                factors[0],
                factors[1],
                factors[2],
            )
        )
    return potential


@torch.no_grad()
def sog_potential_and_faces(
    representation,
    interior_nodes_xyz: Sequence[Tensor],
    sog: SOG,
    *,
    conv_gauss_order: int,
    tail_tolerance: float,
    domain_bounds: Sequence[Tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
) -> Tuple[Tensor, Dict[Tuple[int, int], Tensor]]:
    """Compute interior and six-face potentials for a rectangular box."""
    if len(interior_nodes_xyz) != 3 or len(domain_bounds) != 3:
        raise ValueError("Need three node vectors and three domain intervals")

    # node_sets[dim][key] gives interior/lower/upper targets for that axis.
    node_sets: List[Dict[str, Tensor]] = []
    for dim in range(3):
        nodes = interior_nodes_xyz[dim]
        lower, upper = domain_bounds[dim]
        node_sets.append(
            {
                "interior": nodes,
                "lower": torch.tensor([lower], device=nodes.device, dtype=nodes.dtype),
                "upper": torch.tensor([upper], device=nodes.device, dtype=nodes.dtype),
            }
        )

    # cache[q][dim][key] = (I0,I1)
    cache: List[List[Dict[str, Tuple[Tensor, Tensor]]]] = []
    coefficients: List[Tensor] = []
    for q in range(3):
        coefficient, factor_callables = representation_component_data(representation, q)
        coefficients.append(coefficient)
        q_cache: List[Dict[str, Tuple[Tensor, Tensor]]] = []
        for dim, factor in enumerate(factor_callables):
            dim_cache: Dict[str, Tuple[Tensor, Tensor]] = {}
            lower, upper = domain_bounds[dim]
            for key, nodes in node_sets[dim].items():
                dim_cache[key] = narrow_gaussian_convolution_01(
                    factor,
                    nodes,
                    sog.alpha,
                    gauss_order=conv_gauss_order,
                    tail_tolerance=tail_tolerance,
                    lower=lower,
                    upper=upper,
                )
            q_cache.append(dim_cache)
        cache.append(q_cache)

    def assemble(keys_xyz: Sequence[str]) -> Tensor:
        shape = tuple(int(node_sets[dim][key].numel()) for dim, key in enumerate(keys_xyz))
        potential = torch.zeros(
            shape,
            device=interior_nodes_xyz[0].device,
            dtype=interior_nodes_xyz[0].dtype,
        )
        for q in range(3):
            weighted_coefficient = sog.beta[:, None] * coefficients[q][None, :]
            factors = []
            for dim in range(3):
                I0, I1 = cache[q][dim][keys_xyz[dim]]
                factors.append(-I1 if dim == q else I0)
            potential.add_(
                torch.einsum(
                    "lr,lir,ljr,lkr->ijk",
                    weighted_coefficient,
                    factors[0],
                    factors[1],
                    factors[2],
                )
            )
        return potential

    interior = assemble(("interior", "interior", "interior"))
    faces: Dict[Tuple[int, int], Tensor] = {}
    for q in range(3):
        for side, key in ((0, "lower"), (1, "upper")):
            keys = ["interior", "interior", "interior"]
            keys[q] = key
            faces[(q, side)] = assemble(keys)
    return interior, faces


@torch.no_grad()
def demag_energy_from_potential_weak(
    representation,
    interior_nodes_xyz: Sequence[Tensor],
    interior_weights_xyz: Sequence[Tensor],
    interior_potential: Tensor,
    face_potentials: Dict[Tuple[int, int], Tensor],
    *,
    mu0: float,
    Ms: float,
    domain_bounds: Sequence[Tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
) -> float:
    r"""Compute E_d on a rectangular box without second kernel derivatives.

        E_d/(mu0 Ms^2)
          = 1/2 [ integral_Gamma U (m dot n) dS
                   - integral_Omega U div(m) dx ].
    """
    if len(interior_nodes_xyz) != 3 or len(interior_weights_xyz) != 3:
        raise ValueError("Need three node vectors and three weight vectors")
    w3 = tensor_weights_3d(interior_weights_xyz)
    divergence = torch.zeros_like(interior_potential)
    for q in range(3):
        divergence.add_(
            representation_component_on_product_grid(
                representation, q, interior_nodes_xyz, derivative_dim=q
            )
        )
    volume = torch.sum(w3 * interior_potential * divergence)

    boundary = torch.zeros(
        (), device=interior_nodes_xyz[0].device, dtype=interior_nodes_xyz[0].dtype
    )

    for q in range(3):
        tangential_dims = [d for d in range(3) if d != q]
        w2 = torch.einsum(
            "i,j->ij",
            interior_weights_xyz[tangential_dims[0]],
            interior_weights_xyz[tangential_dims[1]],
        )
        lower, upper = domain_bounds[q]
        fixed = {
            0: torch.tensor([lower], device=interior_nodes_xyz[q].device, dtype=interior_nodes_xyz[q].dtype),
            1: torch.tensor([upper], device=interior_nodes_xyz[q].device, dtype=interior_nodes_xyz[q].dtype),
        }
        for side, normal_sign in ((0, -1.0), (1, 1.0)):
            face_nodes = list(interior_nodes_xyz)
            face_nodes[q] = fixed[side]
            U_face = face_potentials[(q, side)]
            m_face = representation_component_on_product_grid(
                representation, q, face_nodes
            )
            U_face_2d = U_face.squeeze(q)
            m_face_2d = m_face.squeeze(q)
            boundary.add_(normal_sign * torch.sum(w2 * U_face_2d * m_face_2d))

    energy = 0.5 * mu0 * Ms**2 * (boundary - volume)
    return float(energy.cpu())

