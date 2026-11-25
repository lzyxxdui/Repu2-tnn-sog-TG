#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Saving and plotting helpers for CP/TNN demagnetization experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from torch import Tensor

from cp import ComponentwiseCP


def save_componentwise_cp(model: ComponentwiseCP, path: Path) -> None:
    payload: Dict[str, np.ndarray] = {}
    for q, component in enumerate(model.components):
        payload[f"q{q}_coefficient"] = component.coefficient.cpu().numpy()
        payload[f"q{q}_A"] = component.A.cpu().numpy()
        payload[f"q{q}_B"] = component.B.cpu().numpy()
        payload[f"q{q}_C"] = component.C.cpu().numpy()
    payload["grid"] = model.components[0].grid.cpu().numpy()
    np.savez_compressed(path, **payload)


def make_plots(
    output_dir: Path,
    state_name: str,
    nodes_xyz: Sequence[Tensor],
    exact_m: Tensor,
    results: Dict[str, Dict[str, Tensor]],
) -> None:
    """Plot x-y slices at the target z node closest to the box mid-plane."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; plots skipped")
        return

    xnodes, ynodes, znodes = nodes_xyz
    zmid = 0.5 * (float(znodes[0]) + float(znodes[-1]))
    index = int(np.argmin(np.abs(znodes.cpu().numpy() - zmid)))
    x = xnodes.cpu().numpy()
    y = ynodes.cpu().numpy()

    fig, axes = plt.subplots(
        1, len(results) + 1,
        figsize=(4.2 * (len(results) + 1), 4.0),
        constrained_layout=True,
    )
    exact_mid = exact_m[:, :, :, index].permute(1, 2, 0).cpu().numpy()
    image = axes[0].pcolormesh(x, y, exact_mid[..., 2].T, shading="auto")
    axes[0].set_title(r"Exact $m_z$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    fig.colorbar(image, ax=axes[0])
    for axis, (name, data) in zip(axes[1:], results.items()):
        m_mid = data["m"][:, :, :, index].permute(1, 2, 0).cpu().numpy()
        error = np.linalg.norm(m_mid - exact_mid, axis=-1)
        image = axis.pcolormesh(x, y, error.T, shading="auto")
        axis.set_title(f"{name} $|m-m^*|$")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis)
    fig.savefig(output_dir / f"{state_name}_magnetization_midplane.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(results), figsize=(4.2 * len(results), 4.0), constrained_layout=True
    )
    if len(results) == 1:
        axes = [axes]
    for axis, (name, data) in zip(axes, results.items()):
        u_mid = data["U"][:, :, index].cpu().numpy()
        image = axis.pcolormesh(x, y, u_mid.T, shading="auto")
        axis.set_title(f"{name} $U$, mid-z")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis)
    fig.savefig(output_dir / f"{state_name}_potential_midplane.png", dpi=180)
    plt.close(fig)
