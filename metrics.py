#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Accuracy metrics for magnetization and scalar fields."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import Tensor

from quadrature import tensor_weights_3d

# Metrics
# =============================================================================

def relative_l2_tensor(
    approximate: Tensor,
    exact: Tensor,
    weight3: Tensor,
) -> Tensor:
    numerator = torch.sum(weight3 * torch.sum((approximate - exact) ** 2, dim=0))
    denominator = torch.sum(weight3 * torch.sum(exact**2, dim=0))
    return torch.sqrt(numerator / denominator.clamp_min(torch.finfo(exact.dtype).eps))


def point_metrics(approximate: Tensor, exact: Tensor) -> Dict[str, float]:
    error = approximate - exact
    point_error = torch.linalg.vector_norm(error, dim=-1)
    global_relative = (
        torch.linalg.vector_norm(error)
        / torch.linalg.vector_norm(exact).clamp_min(torch.finfo(exact.dtype).eps)
    )
    component_relative = []
    for q in range(3):
        component_relative.append(
            float(
                (
                    torch.linalg.vector_norm(error[:, q])
                    / torch.linalg.vector_norm(exact[:, q]).clamp_min(
                        torch.finfo(exact.dtype).eps
                    )
                ).cpu()
            )
        )
    norm_defect = torch.abs(torch.sum(approximate**2, dim=-1) - 1.0)
    return {
        "relative_L2_m": float(global_relative.cpu()),
        "relative_L2_mx": component_relative[0],
        "relative_L2_my": component_relative[1],
        "relative_L2_mz": component_relative[2],
        "max_pointwise_m_error": float(torch.max(point_error).cpu()),
        "mean_pointwise_m_error": float(torch.mean(point_error).cpu()),
        "unit_norm_rms": float(torch.sqrt(torch.mean(norm_defect**2)).cpu()),
        "unit_norm_max": float(torch.max(norm_defect).cpu()),
    }



def weighted_relative_scalar(
    a: Tensor,
    b: Tensor,
    weights: Sequence[Tensor] | Tensor,
) -> float:
    weight3 = tensor_weights_3d(weights)
    numerator = torch.sum(weight3 * (a - b) ** 2)
    denominator = torch.sum(weight3 * b**2).clamp_min(
        torch.finfo(a.dtype).eps
    )
    return float(torch.sqrt(numerator / denominator).cpu())


