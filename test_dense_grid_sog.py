#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small consistency tests for the uncompressed dense-grid SOG path."""

import torch

from cp import ComponentwiseCP, ScalarCP
from dense_grid_sog import (
    DenseGridMagnetization,
    dense_grid_demag_energy_from_potential_weak,
    dense_grid_sog_potential_and_faces,
)
from quadrature import gauss_legendre_interval
from sog import build_sog, demag_energy_from_potential_weak, sog_potential_and_faces
from states import homogeneous_out_of_plane


def _constant_component(coeff: float, n: int, dtype: torch.dtype) -> ScalarCP:
    grid = torch.linspace(0.0, 1.0, n, dtype=dtype)
    ones = torch.ones((n, 1), dtype=dtype)
    return ScalarCP(
        coefficient=torch.tensor([coeff], dtype=dtype),
        A=ones.clone(),
        B=ones.clone(),
        C=ones.clone(),
        grid=grid,
        loss=0.0,
        sweeps=0,
    )


def test_dense_grid_matches_exact_rank1_constant() -> None:
    device = torch.device("cpu")
    dtype = torch.float64
    bounds = ((0.0, 1.0),) * 3

    dense = DenseGridMagnetization.from_exact_state(
        homogeneous_out_of_plane,
        bounds,
        (5, 5, 5),
        device=device,
        dtype=dtype,
    )
    cp = ComponentwiseCP(
        [
            _constant_component(0.0, 5, dtype),
            _constant_component(0.0, 5, dtype),
            _constant_component(1.0, 5, dtype),
        ]
    )

    nodes = []
    weights = []
    for _ in range(3):
        x, w = gauss_legendre_interval(
            4, 0.0, 1.0, device=device, dtype=dtype
        )
        nodes.append(x)
        weights.append(w)

    sog = build_sog(0.4, 3, device=device, dtype=dtype)
    Ud, Fd = dense_grid_sog_potential_and_faces(
        dense,
        nodes,
        sog,
        conv_gauss_order=12,
        tail_tolerance=1.0e-12,
        alpha_chunk=2,
        x_chunk=8,
        contraction_alpha_chunk=2,
    )
    Uc, Fc = sog_potential_and_faces(
        cp,
        nodes,
        sog,
        conv_gauss_order=12,
        tail_tolerance=1.0e-12,
        domain_bounds=bounds,
    )

    assert torch.max(torch.abs(Ud - Uc)).item() < 1.0e-13
    for key in Fd:
        assert torch.max(torch.abs(Fd[key] - Fc[key])).item() < 1.0e-13

    Ed = dense_grid_demag_energy_from_potential_weak(
        dense,
        nodes,
        weights,
        Ud,
        Fd,
        mu0=1.0,
        Ms=1.0,
    )
    Ec = demag_energy_from_potential_weak(
        cp,
        nodes,
        weights,
        Uc,
        Fc,
        mu0=1.0,
        Ms=1.0,
        domain_bounds=bounds,
    )
    assert abs(Ed - Ec) < 1.0e-13


if __name__ == "__main__":
    test_dense_grid_matches_exact_rank1_constant()
    print("dense-grid SOG consistency test passed")
