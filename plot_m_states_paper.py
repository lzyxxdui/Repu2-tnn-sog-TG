#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Publication-quality visualization of analytic magnetization states.

This script should be placed in the same directory as states.py.

Panels:
    (a) Flower state in a 1 x 1 x 1 cube
    (b) Vortex state in a 1 x 1 x 1 cube
    (c) Vortex state in a 1 x 1 x 0.1 thin film

The arrows are sampled only for visualization. They are unrelated to
the TNN training quadrature, SOG quadrature, or source-grid discretization.

Outputs:
    magnetization_states_paper.pdf   (vector, recommended for LaTeX)
    magnetization_states_paper.png   (600 dpi preview)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from states import flower_state, vortex_state


# ---------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.linewidth": 0.7,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


CUBE_BOUNDS = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
FILM_BOUNDS = ((0.0, 1.0), (0.0, 1.0), (0.0, 0.1))


def tensor_grid(bounds, nxyz, dtype=torch.float64):
    """Uniform display grid."""
    xs = [
        torch.linspace(a, b, n, dtype=dtype)
        for (a, b), n in zip(bounds, nxyz)
    ]
    X, Y, Z = torch.meshgrid(*xs, indexing="ij")
    xyz = torch.stack((X, Y, Z), dim=-1)
    return X, Y, Z, xyz


def normalize_m(m):
    """Normalize safely for arrow visualization."""
    norm = torch.linalg.vector_norm(m, dim=-1, keepdim=True)
    return m / torch.clamp(norm, min=1.0e-14)


def direction_rgb(m):
    """
    Encode the signed 3D magnetization direction directly in RGB.

    For a unit vector m=(mx,my,mz), each signed component is mapped from
    [-1,1] to [0,1]. A small neutral mixing is then applied to avoid overly
    saturated arrows in print.

    This is deterministic and uses all three components of m.
    """
    v = m.detach().cpu().numpy()
    rgb = 0.5 * (v + 1.0)

    # Slight neutral mixing improves print readability.
    neutral = 0.72
    rgb = neutral * rgb + (1.0 - neutral) * 0.55
    return np.clip(rgb, 0.0, 1.0).reshape(-1, 3)


def draw_box(ax, bounds, lw=0.65, alpha=0.65):
    """Draw only the 12 edges of the computational box."""
    (x0, x1), (y0, y1), (z0, z1) = bounds

    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for i, j in edges:
        p, q = vertices[i], vertices[j]
        ax.plot(
            [p[0], q[0]],
            [p[1], q[1]],
            [p[2], q[2]],
            color="0.35",
            lw=lw,
            alpha=alpha,
            zorder=0,
        )


def clean_axes(ax, bounds, *, film=False, film_z_exaggeration=0.28):
    """Paper-style 3D axes."""
    (x0, x1), (y0, y1), (z0, z1) = bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)

    # Transparent panes and axis lines.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))

    # Hide Matplotlib's default 3D axis frame; custom box is drawn above.
    try:
        ax.xaxis.line.set_color((1, 1, 1, 0))
        ax.yaxis.line.set_color((1, 1, 1, 0))
        ax.zaxis.line.set_color((1, 1, 1, 0))
    except Exception:
        pass

    # Similar viewing angle for all panels.
    ax.view_init(elev=22, azim=-56)

    if film:
        # Physical thickness is still z in [0,0.1]; only the displayed
        # box aspect is exaggerated for readability.
        ax.set_box_aspect((1.0, 1.0, film_z_exaggeration))
    else:
        ax.set_box_aspect((1.0, 1.0, 1.0))


def add_quiver(
    ax,
    X,
    Y,
    Z,
    m,
    *,
    arrow_length,
    linewidth=0.50,
    arrow_head=0.32,
):
    colors = direction_rgb(m)

    mx = m[..., 0].detach().cpu().numpy().ravel()
    my = m[..., 1].detach().cpu().numpy().ravel()
    mz = m[..., 2].detach().cpu().numpy().ravel()

    ax.quiver(
        X.detach().cpu().numpy().ravel(),
        Y.detach().cpu().numpy().ravel(),
        Z.detach().cpu().numpy().ravel(),
        mx,
        my,
        mz,
        length=arrow_length,
        normalize=True,
        colors=colors,
        linewidth=linewidth,
        arrow_length_ratio=arrow_head,
        pivot="middle",
    )


def plot_flower(ax):
    # Sparse enough for print, dense enough to show the 3D flower pattern.
    X, Y, Z, xyz = tensor_grid(CUBE_BOUNDS, (6, 6, 6))
    m = normalize_m(
        flower_state(
            xyz,
            a=1.0,
            b=2.0,
            c=1.0,
        )
    )

    add_quiver(
        ax,
        X, Y, Z, m,
        arrow_length=0.145,
        linewidth=0.48,
        arrow_head=0.34,
    )
    clean_axes(ax, CUBE_BOUNDS)
    draw_box(ax, CUBE_BOUNDS)


def plot_vortex_cube(ax):
    # More x-y points than z points because the implemented vortex is z-invariant.
    X, Y, Z, xyz = tensor_grid(CUBE_BOUNDS, (8, 8, 5))
    m = normalize_m(
        vortex_state(
            xyz,
            core_radius=0.14,
            center_x=0.5,
            center_y=0.5,
        )
    )

    add_quiver(
        ax,
        X, Y, Z, m,
        arrow_length=0.135,
        linewidth=0.52,
        arrow_head=0.34,
    )
    clean_axes(ax, CUBE_BOUNDS)
    draw_box(ax, CUBE_BOUNDS)


def plot_vortex_film(ax, film_z_exaggeration):
    # The state is z-invariant, so only three visual z layers are needed.
    X, Y, Z, xyz = tensor_grid(FILM_BOUNDS, (10, 10, 3))
    m = normalize_m(
        vortex_state(
            xyz,
            core_radius=0.14,
            center_x=0.5,
            center_y=0.5,
        )
    )

    add_quiver(
        ax,
        X, Y, Z, m,
        arrow_length=0.095,
        linewidth=0.50,
        arrow_head=0.33,
    )
    clean_axes(
        ax,
        FILM_BOUNDS,
        film=True,
        film_z_exaggeration=film_z_exaggeration,
    )
    draw_box(ax, FILM_BOUNDS)


def panel_label(ax, label):
    ax.text2D(
        0.015,
        0.965,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("magnetization_states_paper"),
        help="Output basename without extension.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution; PDF remains vector output.",
    )
    parser.add_argument(
        "--film-z-exaggeration",
        type=float,
        default=0.28,
        help=(
            "Displayed z aspect for the 1x1x0.1 film. "
            "This changes visualization only, not the physical coordinates."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.film_z_exaggeration <= 0:
        raise ValueError("--film-z-exaggeration must be positive")

    # Figure width is suitable for a two-column paper.
    fig = plt.figure(figsize=(10.8, 3.45))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    plot_flower(ax1)
    plot_vortex_cube(ax2)
    plot_vortex_film(ax3, args.film_z_exaggeration)

    panel_label(ax1, "(a)")
    panel_label(ax2, "(b)")
    panel_label(ax3, "(c)")

    # No in-panel titles: use the LaTeX figure caption instead.
    fig.subplots_adjust(
        left=0.01,
        right=0.995,
        bottom=0.02,
        top=0.995,
        wspace=0.02,
    )

    pdf_path = args.output.with_suffix(".pdf")
    png_path = args.output.with_suffix(".png")

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        png_path,
        dpi=args.dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    print(f"Saved vector PDF : {pdf_path.resolve()}")
    print(f"Saved {args.dpi} dpi PNG: {png_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
