#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-dimensional and tensor-product Gauss quadrature utilities."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


def gauss_legendre_interval(
    n: int,
    a: float,
    b: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """n-point *global* Gauss--Legendre rule on [a,b].

    This function is intentionally unchanged from the original implementation
    so that ``gauss-mode=global`` reproduces the legacy nodes and weights.
    """
    if n < 2:
        raise ValueError("Gauss order must be at least 2")
    if not b > a:
        raise ValueError("Require b>a for an integration interval")
    x, w = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (b - a) * x + 0.5 * (a + b)
    w = 0.5 * (b - a) * w
    return (
        torch.as_tensor(x, device=device, dtype=dtype),
        torch.as_tensor(w, device=device, dtype=dtype),
    )


def composite_gauss_legendre(
    a: float,
    b: float,
    n_subintervals: int,
    order: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[Tensor, Tensor]:
    r"""Composite Gauss--Legendre rule on ``[a,b]``.

    The interval is split uniformly into ``n_subintervals`` pieces.  On each
    piece, the same ``order``-point reference Gauss--Legendre rule on [-1,1]
    is mapped by

        x = (left+right)/2 + (right-left)/2 * xi,
        w = (right-left)/2 * w_ref.

    Hence the returned vectors contain exactly
    ``n_subintervals * order`` nodes and weights.  This is *not* equivalent to
    constructing one global Gauss rule of that total order.
    """
    if n_subintervals < 1:
        raise ValueError("n_subintervals must be at least 1")
    if order < 2:
        raise ValueError("Gauss order per subinterval must be at least 2")
    if not b > a:
        raise ValueError("Require b>a for an integration interval")
    if device is None:
        device = torch.device("cpu")

    xi, wi = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, b, n_subintervals + 1, dtype=np.float64)
    left = edges[:-1]
    right = edges[1:]
    midpoint = 0.5 * (left + right)
    half_length = 0.5 * (right - left)

    # Shape (M, k), then flatten in physical subinterval order.
    nodes = midpoint[:, None] + half_length[:, None] * xi[None, :]
    weights = half_length[:, None] * wi[None, :]

    return (
        torch.as_tensor(nodes.reshape(-1), device=device, dtype=dtype),
        torch.as_tensor(weights.reshape(-1), device=device, dtype=dtype),
    )


def gauss_legendre_01(
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """Backward-compatible global Gauss--Legendre rule on [0,1]."""
    return gauss_legendre_interval(n, 0.0, 1.0, device=device, dtype=dtype)


def tensor_weights_3d(weights_xyz: Sequence[Tensor] | Tensor) -> Tensor:
    """Tensor-product weights for possibly anisotropic 1D rules."""
    if isinstance(weights_xyz, Tensor):
        wx = wy = wz = weights_xyz
    else:
        if len(weights_xyz) != 3:
            raise ValueError("weights_xyz must contain three one-dimensional rules")
        wx, wy, wz = weights_xyz
    return torch.einsum("i,j,k->ijk", wx, wy, wz)
