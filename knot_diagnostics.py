#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Track and visualize one-hidden-layer ReLU/RePU TNN knots during training.

For the current OneDimFactorNet implementation the physical input x is first
mapped to u=2*x-1.  A first-layer neuron has preactivation

    a_h(x) = w_h (2*x - 1) + b_h,

hence its knot/breakpoint is

    x_h = (1 - b_h / w_h) / 2.

This module is intentionally diagnostic-only: it does not alter optimization,
quadrature, SOG convolution, or the demagnetization energy.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from tnn import ComponentwiseTNN, OneDimFactorNet


_COMPONENT_NAMES = ("mx", "my", "mz")
_AXIS_NAMES = ("x", "y", "z")


def parse_epoch_list(text: str) -> Tuple[int, ...]:
    values = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if any(value < 0 for value in values):
        raise ValueError("knot snapshot epochs must be non-negative")
    return tuple(values)


def _activation_power(name: str) -> int:
    if name == "relu":
        return 1
    if name == "repu2":
        return 2
    if name == "repu3":
        return 3
    raise ValueError(
        "Knot tracking is defined for relu/repu2/repu3. "
        "tanh and sin do not have finite activation knots."
    )


@torch.no_grad()
def initialize_uniform_knots(
    model: ComponentwiseTNN,
    bounds: Sequence[Tuple[float, float]],
) -> None:
    """Place first-layer knots uniformly in each learned physical interval.

    Only the first-layer biases are changed; Xavier-initialized input/output
    weights are retained.  This provides the clean analogue of starting a
    moving-mesh method from a uniform partition.  It is intended for the
    knot-adaptivity experiment and is not applied unless requested in main.py.
    """
    if model.depth != 1:
        raise ValueError("uniform knot initialization requires depth=1")
    _activation_power(model.activation)
    for component in model.components:
        factors = (component.phi_x, component.phi_y, component.phi_z)
        for dim, factor in enumerate(factors):
            if not isinstance(factor, OneDimFactorNet):
                continue
            lower, upper = bounds[dim]
            # Midpoints of W equal subintervals avoid knots exactly on boundaries.
            knots = torch.linspace(
                lower, upper, factor.width + 1,
                device=factor.layers[0].weight.device,
                dtype=factor.layers[0].weight.dtype,
            )
            knots = 0.5 * (knots[:-1] + knots[1:])
            w = factor.layers[0].weight[:, 0]
            # Current network preprocessing is u=2*x-1.
            factor.layers[0].bias.copy_(-w * (2.0 * knots - 1.0))
            factor.invalidate_normalization_cache()


@torch.no_grad()
def extract_knots(
    model: ComponentwiseTNN,
    bounds: Sequence[Tuple[float, float]],
    *,
    vortex_center: Tuple[float, float] = (0.5, 0.5),
    core_radius: float = 0.14,
) -> List[Dict[str, float | int | str | bool]]:
    """Return knot positions and importance for all learned 1D factor nets.

    The importance is designed to emphasize knots that influence spatial
    variation of the final TNN component.  For a one-hidden-layer factor,
    the hidden-to-rank weight V_{r,h} is weighted by the TNN coefficient c_r
    and the per-channel normalization.  It is then multiplied by |2w_h|^p,
    where p=1,2,3 for ReLU/RePU2/RePU3 respectively.  This corresponds to
    the scale of the derivative/curvature jump associated with the knot,
    up to activation-dependent constants.
    """
    if model.depth != 1:
        raise ValueError(
            "Knot tracking requires --tnn-depth 1. For deeper networks the "
            "effective partition is not described by -b/w of one layer."
        )
    power = _activation_power(model.activation)
    if len(bounds) != 3:
        raise ValueError("bounds must contain x/y/z intervals")

    rows: List[Dict[str, float | int | str | bool]] = []
    eps = torch.finfo(next(model.parameters()).dtype).eps

    for q, component in enumerate(model.components):
        coeff = component.coefficients.detach()
        factors = (component.phi_x, component.phi_y, component.phi_z)
        for dim, factor in enumerate(factors):
            if not isinstance(factor, OneDimFactorNet):
                continue
            if factor.depth != 1:
                raise ValueError("all tracked factor networks must have depth=1")

            first = factor.layers[0]
            w = first.weight[:, 0].detach()
            b = first.bias.detach()
            out_weight = factor.output.weight.detach()  # [rank, width]
            norms = factor.channel_norms().detach().clamp_min(64.0 * eps)

            effective = (coeff / norms)[:, None] * out_weight
            amplitude = torch.sqrt(torch.sum(effective.square(), dim=0))
            variation_scale = torch.abs(2.0 * w).pow(power)
            importance = amplitude * variation_scale

            finite_w = torch.abs(w) > 128.0 * eps
            position = torch.full_like(w, float("nan"))
            # raw input is u = 2*x - 1, so w*u+b=0 -> x=(1-b/w)/2.
            position[finite_w] = 0.5 * (1.0 - b[finite_w] / w[finite_w])

            lower, upper = bounds[dim]
            center_axis = vortex_center[dim] if dim < 2 else 0.5 * (lower + upper)
            for h in range(factor.width):
                pos = float(position[h].cpu())
                imp = float(importance[h].cpu())
                amp = float(amplitude[h].cpu())
                valid = math.isfinite(pos)
                inside = valid and lower <= pos <= upper
                distance = abs(pos - center_axis) if valid else float("nan")
                in_core_projection = (
                    inside and dim < 2 and distance <= core_radius
                )
                rows.append(
                    {
                        "component": _COMPONENT_NAMES[q],
                        "component_index": q,
                        "axis": _AXIS_NAMES[dim],
                        "axis_index": dim,
                        "neuron": h,
                        "position": pos,
                        "importance": imp,
                        "amplitude_importance": amp,
                        "input_weight": float(w[h].cpu()),
                        "bias": float(b[h].cpu()),
                        "inside_domain": inside,
                        "distance_to_core_center_axis": distance,
                        "inside_core_projection": in_core_projection,
                    }
                )
    return rows


def _xy_rows(rows: Iterable[Dict]) -> List[Dict]:
    return [
        row
        for row in rows
        if row["axis"] in ("x", "y")
        and bool(row["inside_domain"])
        and math.isfinite(float(row["importance"]))
    ]


def snapshot_metrics(rows: Sequence[Dict]) -> Dict[str, float]:
    xy = _xy_rows(rows)
    if not xy:
        return {
            "weighted_mean_distance": float("nan"),
            "weighted_core_fraction": float("nan"),
            "unweighted_mean_distance": float("nan"),
            "unweighted_core_fraction": float("nan"),
            "inside_xy_knots": 0.0,
        }
    weights = np.asarray([max(float(row["importance"]), 0.0) for row in xy])
    distances = np.asarray([float(row["distance_to_core_center_axis"]) for row in xy])
    core = np.asarray([bool(row["inside_core_projection"]) for row in xy], dtype=float)
    if float(weights.sum()) <= 0.0:
        weights = np.ones_like(weights)
    return {
        "weighted_mean_distance": float(np.sum(weights * distances) / np.sum(weights)),
        "weighted_core_fraction": float(np.sum(weights * core) / np.sum(weights)),
        "unweighted_mean_distance": float(np.mean(distances)),
        "unweighted_core_fraction": float(np.mean(core)),
        "inside_xy_knots": float(len(xy)),
    }


@dataclass
class KnotSnapshot:
    epoch: int
    loss: float
    trigger: str
    rows: List[Dict]
    metrics: Dict[str, float]


@dataclass
class KnotRecorder:
    output_dir: Path
    bounds: Sequence[Tuple[float, float]]
    core_radius: float
    fixed_epochs: Tuple[int, ...]
    loss_drop: float = 0.0
    min_epoch_gap: int = 200
    topk_per_factor: int = 12
    vortex_center: Tuple[float, float] = (0.5, 0.5)
    save_model_snapshots: bool = False
    snapshots: List[KnotSnapshot] = field(default_factory=list)
    _snapshot_epochs: set[int] = field(default_factory=set)
    _best_loss: float = math.inf
    _last_loss_trigger_value: float | None = None
    _last_loss_trigger_epoch: int = -10**9

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not (0.0 <= self.loss_drop < 1.0):
            raise ValueError("knot loss drop must satisfy 0 <= value < 1")
        if self.min_epoch_gap < 0:
            raise ValueError("knot min epoch gap must be non-negative")
        if self.topk_per_factor < 1:
            raise ValueError("knot top-k per factor must be positive")

    def __call__(self, epoch: int, loss: float, model: ComponentwiseTNN) -> None:
        if not math.isfinite(loss):
            return
        self._best_loss = min(self._best_loss, loss)
        triggers: List[str] = []
        if epoch in self.fixed_epochs:
            triggers.append("fixed_epoch")

        if epoch == 0 and self._last_loss_trigger_value is None:
            self._last_loss_trigger_value = loss
            self._last_loss_trigger_epoch = 0
        elif self.loss_drop > 0.0 and self._last_loss_trigger_value is not None:
            threshold = (1.0 - self.loss_drop) * self._last_loss_trigger_value
            if (
                self._best_loss <= threshold
                and epoch - self._last_loss_trigger_epoch >= self.min_epoch_gap
            ):
                triggers.append("loss_drop")
                self._last_loss_trigger_value = self._best_loss
                self._last_loss_trigger_epoch = epoch

        if triggers:
            self.capture(epoch, loss, model, "+".join(triggers))

    def capture(
        self,
        epoch: int,
        loss: float,
        model: ComponentwiseTNN,
        trigger: str,
        *,
        force: bool = False,
    ) -> None:
        if epoch in self._snapshot_epochs and not force:
            return
        rows = extract_knots(
            model,
            self.bounds,
            vortex_center=self.vortex_center,
            core_radius=self.core_radius,
        )
        snap = KnotSnapshot(
            epoch=epoch,
            loss=float(loss),
            trigger=trigger,
            rows=rows,
            metrics=snapshot_metrics(rows),
        )
        self.snapshots.append(snap)
        self._snapshot_epochs.add(epoch)
        self._write_snapshot_csv(snap)
        if self.save_model_snapshots:
            torch.save(
                {
                    "epoch": epoch,
                    "loss": float(loss),
                    "trigger": trigger,
                    "state_dict": model.state_dict(),
                },
                self.output_dir / f"model_epoch_{epoch:06d}.pt",
            )

    def capture_final(self, model: ComponentwiseTNN, loss: float, adam_epochs: int) -> None:
        # Use a distinct synthetic epoch only when the final/best model has already
        # been recorded at the last Adam epoch. This keeps the chronology clear.
        final_epoch = adam_epochs
        if final_epoch in self._snapshot_epochs:
            final_epoch = adam_epochs + 1
        self.capture(final_epoch, loss, model, "final_best", force=True)

    def _write_snapshot_csv(self, snap: KnotSnapshot) -> None:
        path = self.output_dir / f"knots_epoch_{snap.epoch:06d}.csv"
        if not snap.rows:
            return
        fieldnames = ["epoch", "loss", "trigger"] + list(snap.rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in snap.rows:
                writer.writerow(
                    {"epoch": snap.epoch, "loss": snap.loss, "trigger": snap.trigger, **row}
                )

    def finalize(self) -> None:
        if not self.snapshots:
            return
        self.snapshots.sort(key=lambda snap: snap.epoch)
        self._write_summary_csv()
        self._plot_partition_evolution()
        self._plot_metric_evolution()

    def _write_summary_csv(self) -> None:
        path = self.output_dir / "knot_snapshot_summary.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "epoch",
                "loss",
                "trigger",
                "weighted_mean_distance",
                "weighted_core_fraction",
                "unweighted_mean_distance",
                "unweighted_core_fraction",
                "inside_xy_knots",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for snap in self.snapshots:
                writer.writerow(
                    {
                        "epoch": snap.epoch,
                        "loss": snap.loss,
                        "trigger": snap.trigger,
                        **snap.metrics,
                    }
                )

    def _selected_rows(self, rows: Sequence[Dict]) -> List[Dict]:
        selected: List[Dict] = []
        groups: Dict[Tuple[str, str], List[Dict]] = {}
        for row in _xy_rows(rows):
            groups.setdefault((str(row["component"]), str(row["axis"])), []).append(row)
        for group_rows in groups.values():
            group_rows.sort(key=lambda row: float(row["importance"]), reverse=True)
            selected.extend(group_rows[: self.topk_per_factor])
        return selected

    def _plot_partition_evolution(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Rectangle

        n = len(self.snapshots)
        cols = min(4, n)
        rows_n = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(4.0 * cols, 3.9 * rows_n), squeeze=False)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        component_colors = {name: colors[i % len(colors)] for i, name in enumerate(_COMPONENT_NAMES)}
        x0, y0 = self.vortex_center
        rc = self.core_radius

        for ax, snap in zip(axes.flat, self.snapshots):
            # Highlight the x/y projections of the physical vortex core.
            ax.add_patch(
                Rectangle(
                    (x0 - rc, 0.0),
                    2.0 * rc,
                    1.0,
                    fill=False,
                    linestyle=":",
                    linewidth=0.8,
                )
            )
            ax.add_patch(
                Rectangle(
                    (0.0, y0 - rc),
                    1.0,
                    2.0 * rc,
                    fill=False,
                    linestyle=":",
                    linewidth=0.8,
                )
            )
            ax.add_patch(Circle((x0, y0), rc, fill=False, linewidth=1.4))
            ax.plot([x0], [y0], marker="o", markersize=3.0)

            all_xy = _xy_rows(snap.rows)
            for row in all_xy:
                pos = float(row["position"])
                if row["axis"] == "x":
                    ax.axvline(pos, alpha=0.055, linewidth=0.30, color="0.35")
                else:
                    ax.axhline(pos, alpha=0.055, linewidth=0.30, color="0.35")

            selected = self._selected_rows(snap.rows)
            max_imp = max((float(row["importance"]) for row in selected), default=1.0)
            max_imp = max(max_imp, 1.0e-30)
            for row in selected:
                pos = float(row["position"])
                strength = math.sqrt(max(float(row["importance"]), 0.0) / max_imp)
                alpha = 0.20 + 0.75 * strength
                linewidth = 0.45 + 1.55 * strength
                color = component_colors[str(row["component"])]
                if row["axis"] == "x":
                    ax.axvline(pos, alpha=alpha, linewidth=linewidth, color=color)
                else:
                    ax.axhline(pos, alpha=alpha, linewidth=linewidth, color=color)

            ax.set_xlim(self.bounds[0])
            ax.set_ylim(self.bounds[1])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            core_fraction = snap.metrics["weighted_core_fraction"]
            mean_distance = snap.metrics["weighted_mean_distance"]
            label_epoch = "final" if snap.trigger == "final_best" else f"epoch {snap.epoch}"
            ax.set_title(
                f"{label_epoch}\n"
                f"L={snap.loss:.2e},  core={core_fraction:.2f},  d={mean_distance:.3f}"
            )

        for ax in axes.flat[n:]:
            ax.axis("off")

        handles = [
            Line2D([0], [0], color=component_colors[name], lw=1.8, label=name)
            for name in _COMPONENT_NAMES
        ]
        handles.append(Line2D([0], [0], color="black", lw=1.4, label="vortex core"))
        fig.suptitle(
            "One-hidden-layer TNN knot evolution",
            y=0.995,
        )
        fig.legend(
            handles=handles, loc="upper center", ncol=4, frameon=False,
            bbox_to_anchor=(0.5, 0.965),
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
        fig.savefig(self.output_dir / "knot_partition_evolution.png", dpi=350, bbox_inches="tight")
        fig.savefig(self.output_dir / "knot_partition_evolution.pdf", bbox_inches="tight")
        plt.close(fig)

    def _plot_metric_evolution(self) -> None:
        import matplotlib.pyplot as plt

        epochs = [snap.epoch for snap in self.snapshots]
        mean_distance = [snap.metrics["weighted_mean_distance"] for snap in self.snapshots]
        core_fraction = [snap.metrics["weighted_core_fraction"] for snap in self.snapshots]
        losses = [snap.loss for snap in self.snapshots]

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(epochs, mean_distance, marker="o")
        ax.set_xlabel("Adam epoch / final snapshot index")
        ax.set_ylabel("importance-weighted mean knot distance")
        ax.set_title("Knot distance to vortex-core center")
        fig.tight_layout()
        fig.savefig(self.output_dir / "knot_mean_distance_vs_epoch.png", dpi=350, bbox_inches="tight")
        fig.savefig(self.output_dir / "knot_mean_distance_vs_epoch.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(epochs, core_fraction, marker="o")
        ax.set_xlabel("Adam epoch / final snapshot index")
        ax.set_ylabel("importance fraction in core projection")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Knot importance entering vortex-core projection")
        fig.tight_layout()
        fig.savefig(self.output_dir / "knot_core_fraction_vs_epoch.png", dpi=350, bbox_inches="tight")
        fig.savefig(self.output_dir / "knot_core_fraction_vs_epoch.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.semilogy(epochs, losses, marker="o")
        ax.set_xlabel("Adam epoch / final snapshot index")
        ax.set_ylabel("training loss")
        ax.set_title("Training loss at knot snapshots")
        fig.tight_layout()
        fig.savefig(self.output_dir / "knot_snapshot_loss.png", dpi=350, bbox_inches="tight")
        fig.savefig(self.output_dir / "knot_snapshot_loss.pdf", bbox_inches="tight")
        plt.close(fig)
