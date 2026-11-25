#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Componentwise CP decomposition and continuous factor interpolation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

# =============================================================================
# Componentwise CP decomposition
# =============================================================================

@dataclass
class ScalarCP:
    coefficient: Tensor  # [R]
    A: Tensor            # [N,R]
    B: Tensor            # [N,R]
    C: Tensor            # [N,R]
    grid: Tensor         # [N]
    loss: float
    sweeps: int

    @property
    def rank(self) -> int:
        return int(self.coefficient.numel())

    @property
    def parameter_count(self) -> int:
        return int(
            self.coefficient.numel()
            + self.A.numel()
            + self.B.numel()
            + self.C.numel()
        )

    def reconstruct(self) -> Tensor:
        return torch.einsum(
            "r,ir,jr,kr->ijk", self.coefficient, self.A, self.B, self.C
        )

    def factor(self, dim: int, x: Tensor) -> Tensor:
        table = (self.A, self.B, self.C)[dim]
        return interpolate_uniform_factor(table, x)

    def evaluate(self, xyz: Tensor) -> Tensor:
        fx = self.factor(0, xyz[..., 0])
        fy = self.factor(1, xyz[..., 1])
        fz = self.factor(2, xyz[..., 2])
        return torch.sum(
            fx * fy * fz * self.coefficient,
            dim=-1,
        )


@dataclass
class ComponentwiseCP:
    components: List[ScalarCP]

    @property
    def parameter_count(self) -> int:
        return sum(model.parameter_count for model in self.components)

    def evaluate(self, xyz: Tensor) -> Tensor:
        return torch.stack([model.evaluate(xyz) for model in self.components], dim=-1)

    def evaluate_tensor_grid(self, nodes: Tensor) -> Tensor:
        outputs = []
        for model in self.components:
            fx = model.factor(0, nodes)
            fy = model.factor(1, nodes)
            fz = model.factor(2, nodes)
            outputs.append(
                torch.einsum(
                    "r,ir,jr,kr->ijk",
                    model.coefficient,
                    fx,
                    fy,
                    fz,
                )
            )
        return torch.stack(outputs, dim=0)


def interpolate_uniform_factor(factor: Tensor, x: Tensor) -> Tensor:
    """Piecewise-linear interpolation of an N-by-R factor on [0,1]."""
    n = factor.shape[0]
    s = torch.clamp(x, 0.0, 1.0) * (n - 1)
    left = torch.floor(s).to(torch.long).clamp(0, n - 2)
    t = (s - left.to(s.dtype)).unsqueeze(-1)
    return (1.0 - t) * factor[left] + t * factor[left + 1]


def _regularized_solve(mttkrp: Tensor, gram: Tensor, reg: float) -> Tensor:
    rank = gram.shape[0]
    scale = torch.trace(gram).abs() / max(rank, 1)
    ridge = reg * torch.clamp(scale, min=torch.finfo(gram.dtype).eps)
    system = gram + ridge * torch.eye(
        rank, device=gram.device, dtype=gram.dtype
    )
    try:
        return torch.linalg.solve(system, mttkrp.T).T
    except RuntimeError:
        return mttkrp @ torch.linalg.pinv(system)


def _balance_scalar_cp(
    coefficient: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
) -> None:
    eps = torch.finfo(A.dtype).eps
    for factor in (A, B, C):
        norm = torch.linalg.vector_norm(factor, dim=0).clamp_min(eps)
        factor.div_(norm)
        coefficient.mul_(norm)


def fit_scalar_cp_als(
    target: Tensor,
    grid: Tensor,
    rank: int,
    *,
    max_sweeps: int,
    tol: float,
    reg: float,
    seed: int,
    verbose: bool,
) -> ScalarCP:
    if target.ndim != 3:
        raise ValueError("scalar CP target must have shape (Nx,Ny,Nz)")
    nx, ny, nz = target.shape
    gen = torch.Generator(device=target.device)
    gen.manual_seed(seed)

    coefficient = torch.randn(
        rank, generator=gen, device=target.device, dtype=target.dtype
    )
    A = torch.randn(
        nx, rank, generator=gen, device=target.device, dtype=target.dtype
    )
    B = torch.randn(
        ny, rank, generator=gen, device=target.device, dtype=target.dtype
    )
    C = torch.randn(
        nz, rank, generator=gen, device=target.device, dtype=target.dtype
    )
    coefficient.mul_(max(float(torch.std(target).cpu()), 1.0e-2))
    _balance_scalar_cp(coefficient, A, B, C)

    target_norm = torch.linalg.vector_norm(target).clamp_min(
        torch.finfo(target.dtype).eps
    )
    best_loss = math.inf
    best_state = None
    previous = math.inf
    best_sweep = 0

    with torch.no_grad():
        for sweep in range(1, max_sweeps + 1):
            # Update coefficient by treating it as a one-row mode.
            mtt = torch.einsum("ijk,ir,jr,kr->r", target, A, B, C)[None, :]
            gram = (A.T @ A) * (B.T @ B) * (C.T @ C)
            coefficient = _regularized_solve(mtt, gram, reg)[0]

            mtt_A = torch.einsum(
                "ijk,r,jr,kr->ir", target, coefficient, B, C
            )
            gram_A = (
                torch.outer(coefficient, coefficient)
                * (B.T @ B)
                * (C.T @ C)
            )
            A = _regularized_solve(mtt_A, gram_A, reg)

            mtt_B = torch.einsum(
                "ijk,r,ir,kr->jr", target, coefficient, A, C
            )
            gram_B = (
                torch.outer(coefficient, coefficient)
                * (A.T @ A)
                * (C.T @ C)
            )
            B = _regularized_solve(mtt_B, gram_B, reg)

            mtt_C = torch.einsum(
                "ijk,r,ir,jr->kr", target, coefficient, A, B
            )
            gram_C = (
                torch.outer(coefficient, coefficient)
                * (A.T @ A)
                * (B.T @ B)
            )
            C = _regularized_solve(mtt_C, gram_C, reg)
            _balance_scalar_cp(coefficient, A, B, C)

            if sweep == 1 or sweep % 5 == 0 or sweep == max_sweeps:
                approx = torch.einsum(
                    "r,ir,jr,kr->ijk", coefficient, A, B, C
                )
                loss = float(
                    (torch.linalg.vector_norm(approx - target) / target_norm).cpu()
                )
                if loss < best_loss:
                    best_loss = loss
                    best_state = tuple(
                        tensor.clone() for tensor in (coefficient, A, B, C)
                    )
                    best_sweep = sweep
                relative_change = abs(previous - loss) / max(loss, 1.0e-30)
                if verbose:
                    print(
                        f"      sweep {sweep:4d}: scalar relative L2={loss:.6e}"
                    )
                if loss < tol or relative_change < tol * 0.1:
                    break
                previous = loss

    if best_state is None:
        raise RuntimeError("CP ALS failed")
    return ScalarCP(
        *best_state,
        grid=grid,
        loss=best_loss,
        sweeps=best_sweep,
    )


def fit_componentwise_cp(
    target: Tensor,
    grid: Tensor,
    rank: int,
    *,
    restarts: int,
    max_sweeps: int,
    tol: float,
    reg: float,
    seed: int,
    verbose: bool,
) -> ComponentwiseCP:
    components: List[ScalarCP] = []
    for q, name in enumerate(("mx", "my", "mz")):
        print(f"    CP component {name}")
        best = None
        for restart in range(restarts):
            model = fit_scalar_cp_als(
                target[q],
                grid,
                rank,
                max_sweeps=max_sweeps,
                tol=tol,
                reg=reg,
                seed=seed + 1009 * restart + 101 * q,
                verbose=verbose,
            )
            if best is None or model.loss < best.loss:
                best = model
        assert best is not None
        components.append(best)
    return ComponentwiseCP(components)


