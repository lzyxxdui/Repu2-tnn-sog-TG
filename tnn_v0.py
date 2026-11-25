#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalized componentwise TNN models, losses, and training routines."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn

from quadrature import tensor_weights_3d

# =============================================================================
# Componentwise TNN
# =============================================================================

class Sin(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


class RePU(nn.Module):
    """Rectified power unit: RePU_p(x) = max(x,0)^p."""

    def __init__(self, power: int) -> None:
        super().__init__()
        if power not in (2, 3):
            raise ValueError("Only RePU powers 2 and 3 are supported")
        self.power = power

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(x).pow(self.power)


def make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "repu2":
        return RePU(2)
    if name == "repu3":
        return RePU(3)
    if name == "tanh":
        return nn.Tanh()
    if name == "sin":
        return Sin()
    raise ValueError(f"unsupported activation: {name}")


def activation_derivative(name: str, preactivation: Tensor) -> Tensor:
    """Pointwise derivative used by the exact one-dimensional chain rule."""
    if name == "relu":
        # PyTorch also uses zero as the derivative at the ReLU kink.
        return (preactivation > 0.0).to(preactivation.dtype)
    if name == "repu2":
        return 2.0 * torch.relu(preactivation)
    if name == "repu3":
        return 3.0 * torch.relu(preactivation).square()
    if name == "tanh":
        value = torch.tanh(preactivation)
        return 1.0 - value.square()
    if name == "sin":
        return torch.cos(preactivation)
    raise ValueError(f"unsupported activation: {name}")


class ConstantFactor(nn.Module):
    """A parameter-free rank-channel factor identically equal to one."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = rank

    def forward(self, x: Tensor) -> Tensor:
        return torch.ones((*x.shape[:-1], self.rank), device=x.device, dtype=x.dtype)

    def forward_with_derivative(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        value = self.forward(x)
        return value, torch.zeros_like(value)


class OneDimFactorNet(nn.Module):
    r"""One-dimensional rank-channel network with weighted L2 normalization.

    The raw network produces ``rank`` channels psi_r.  Before a factor is used
    in the tensor product, each channel is normalized by a fixed one-dimensional
    quadrature rule,

        phi_r(x) = psi_r(x) / sqrt(sum_i w_i psi_r(x_i)^2).

    ``forward_with_derivative`` propagates the scalar input derivative through
    the one-dimensional network analytically.  It therefore returns all rank
    channels and their first derivatives in one forward graph, avoiding one
    autograd call per rank channel.  Backpropagation through this graph still
    differentiates with respect to all trainable network parameters.
    """

    def __init__(
        self,
        rank: int,
        width: int,
        depth: int,
        activation: str,
    ) -> None:
        super().__init__()
        if rank < 1 or width < 1 or depth < 1:
            raise ValueError("rank, width and depth must be positive")
        self.rank = rank
        self.width = width
        self.depth = depth
        self.activation_name = activation
        self.layers = nn.ModuleList()
        in_features = 1
        for _ in range(depth):
            layer = nn.Linear(in_features, width)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            in_features = width
        self.activation = make_activation(activation)
        self.output = nn.Linear(in_features, rank)
        nn.init.xavier_uniform_(self.output.weight, gain=0.35)
        nn.init.uniform_(self.output.bias, a=-0.25, b=0.25)

        self.register_buffer("normalization_nodes", torch.empty(0), persistent=True)
        self.register_buffer("normalization_weights", torch.empty(0), persistent=True)
        self.register_buffer("cached_channel_norms", torch.ones(rank), persistent=True)
        self._normalization_cache_valid = False

    def raw_forward(self, x: Tensor) -> Tensor:
        value = 2.0 * x - 1.0
        for layer in self.layers:
            value = self.activation(layer(value))
        return self.output(value)

    def raw_forward_with_derivative(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return raw channels and exact d/dx channels by a 1D chain rule."""
        value = 2.0 * x - 1.0
        derivative = torch.full_like(value, 2.0)
        for layer in self.layers:
            preactivation = layer(value)
            derivative = (derivative @ layer.weight.T) * activation_derivative(
                self.activation_name, preactivation
            )
            value = self.activation(preactivation)
        raw = self.output(value)
        raw_derivative = derivative @ self.output.weight.T
        return raw, raw_derivative

    def set_normalization_rule(self, nodes: Tensor, weights: Tensor) -> None:
        nodes = nodes.reshape(-1, 1)
        weights = weights.reshape(-1)
        if nodes.shape[0] != weights.shape[0]:
            raise ValueError("normalization nodes and weights must have equal length")
        if nodes.numel() == 0:
            raise ValueError("normalization quadrature must not be empty")
        if torch.any(weights <= 0):
            raise ValueError("normalization quadrature weights must be positive")
        self.normalization_nodes = nodes.detach().clone()
        self.normalization_weights = weights.detach().clone()
        self.invalidate_normalization_cache()

    def invalidate_normalization_cache(self) -> None:
        self._normalization_cache_valid = False

    def channel_norms(self) -> Tensor:
        if self.normalization_nodes.numel() == 0:
            raise RuntimeError(
                "factor normalization rule is unset; call "
                "ComponentwiseTNN.set_factor_normalization_rule first"
            )
        raw = self.raw_forward(self.normalization_nodes)
        squared_norm = torch.sum(
            self.normalization_weights[:, None] * raw.square(), dim=0
        )
        eps = torch.finfo(raw.dtype).eps
        return torch.sqrt(squared_norm.clamp_min(64.0 * eps))

    @torch.no_grad()
    def refresh_normalization_cache(self) -> None:
        self.cached_channel_norms.copy_(self.channel_norms())
        self._normalization_cache_valid = True

    def _current_norms(self) -> Tensor:
        if self.training:
            return self.channel_norms()
        if not self._normalization_cache_valid:
            self.refresh_normalization_cache()
        return self.cached_channel_norms

    def forward(self, x: Tensor) -> Tensor:
        return self.raw_forward(x) / self._current_norms()

    def forward_with_derivative(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        raw, raw_derivative = self.raw_forward_with_derivative(x)
        norms = self._current_norms()
        return raw / norms, raw_derivative / norms


class ScalarTNN(nn.Module):
    def __init__(
        self,
        rank: int,
        width: int,
        depth: int,
        *,
        component: int,
        z_invariant: bool,
        dominant_constant_mz: bool,
        activation: str,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.component = component
        self.z_invariant = z_invariant
        self.phi_x = OneDimFactorNet(rank, width, depth, activation)
        self.phi_y = OneDimFactorNet(rank, width, depth, activation)
        self.phi_z: nn.Module = (
            ConstantFactor(rank)
            if z_invariant
            else OneDimFactorNet(rank, width, depth, activation)
        )
        self.coefficients = nn.Parameter(torch.empty(rank))

        with torch.no_grad():
            self.coefficients.normal_(mean=0.0, std=2.0e-2)
            if component == 2 and dominant_constant_mz:
                for net in (self.phi_x, self.phi_y):
                    net.output.weight[0].zero_()
                    net.output.bias[0] = 1.0
                if isinstance(self.phi_z, OneDimFactorNet):
                    self.phi_z.output.weight[0].zero_()
                    self.phi_z.output.bias[0] = 1.0
                self.coefficients[0] = 1.0

    def factors(
        self, nodes_xyz: Sequence[Tensor] | Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if isinstance(nodes_xyz, Tensor):
            xnodes = ynodes = znodes = nodes_xyz
        else:
            if len(nodes_xyz) != 3:
                raise ValueError("nodes_xyz must contain three node vectors")
            xnodes, ynodes, znodes = nodes_xyz
        return (
            self.phi_x(xnodes.reshape(-1, 1)),
            self.phi_y(ynodes.reshape(-1, 1)),
            self.phi_z(znodes.reshape(-1, 1)),
        )

    def evaluate_tensor_grid(self, nodes_xyz: Sequence[Tensor] | Tensor) -> Tensor:
        fx, fy, fz = self.factors(nodes_xyz)
        return torch.einsum("r,ir,jr,kr->ijk", self.coefficients, fx, fy, fz)

    def forward(self, xyz: Tensor) -> Tensor:
        fx = self.phi_x(xyz[..., 0:1])
        fy = self.phi_y(xyz[..., 1:2])
        fz = self.phi_z(xyz[..., 2:3])
        return torch.sum(fx * fy * fz * self.coefficients, dim=-1)


class ComponentwiseTNN(nn.Module):
    def __init__(
        self,
        rank: int,
        width: int,
        depth: int,
        *,
        z_invariant: bool = False,
        dominant_constant_mz: bool = False,
        activation: str = "sin",
    ) -> None:
        super().__init__()
        self.rank = rank
        self.width = width
        self.depth = depth
        self.activation = activation
        self.z_invariant = z_invariant
        self.dominant_constant_mz = dominant_constant_mz
        self.components = nn.ModuleList(
            [
                ScalarTNN(
                    rank,
                    width,
                    depth,
                    component=q,
                    z_invariant=z_invariant,
                    dominant_constant_mz=dominant_constant_mz,
                    activation=activation,
                )
                for q in range(3)
            ]
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def learned_factor_nets(self) -> Iterable[OneDimFactorNet]:
        for component in self.components:
            for factor in (component.phi_x, component.phi_y, component.phi_z):
                if isinstance(factor, OneDimFactorNet):
                    yield factor

    def set_factor_normalization_rule(
        self,
        nodes_xyz: Sequence[Tensor] | Tensor,
        weights_xyz: Sequence[Tensor] | Tensor,
    ) -> None:
        if isinstance(nodes_xyz, Tensor):
            nodes3 = (nodes_xyz, nodes_xyz, nodes_xyz)
        else:
            if len(nodes_xyz) != 3:
                raise ValueError("nodes_xyz must contain three node vectors")
            nodes3 = tuple(nodes_xyz)
        if isinstance(weights_xyz, Tensor):
            weights3 = (weights_xyz, weights_xyz, weights_xyz)
        else:
            if len(weights_xyz) != 3:
                raise ValueError("weights_xyz must contain three weight vectors")
            weights3 = tuple(weights_xyz)
        for component in self.components:
            for dim, factor in enumerate((component.phi_x, component.phi_y, component.phi_z)):
                if isinstance(factor, OneDimFactorNet):
                    factor.set_normalization_rule(nodes3[dim], weights3[dim])

    def invalidate_factor_normalization_cache(self) -> None:
        for factor in self.learned_factor_nets():
            factor.invalidate_normalization_cache()

    @torch.no_grad()
    def refresh_factor_normalization_cache(self) -> None:
        for factor in self.learned_factor_nets():
            factor.refresh_normalization_cache()

    @torch.no_grad()
    def maximum_factor_normalization_error(self) -> float:
        errors: List[Tensor] = []
        for factor in self.learned_factor_nets():
            values = factor(factor.normalization_nodes)
            integral = torch.sum(
                factor.normalization_weights[:, None] * values.square(), dim=0
            )
            errors.append(torch.max(torch.abs(integral - 1.0)))
        if not errors:
            return 0.0
        return float(torch.max(torch.stack(errors)).cpu())

    def forward(self, xyz: Tensor) -> Tensor:
        return torch.stack([component(xyz) for component in self.components], dim=-1)

    def evaluate_tensor_grid(self, nodes_xyz: Sequence[Tensor] | Tensor) -> Tensor:
        return torch.stack(
            [component.evaluate_tensor_grid(nodes_xyz) for component in self.components], dim=0
        )


def copy_factor_module(old: nn.Module, new: nn.Module, old_rank: int) -> None:
    if isinstance(old, ConstantFactor) and isinstance(new, ConstantFactor):
        return
    if not isinstance(old, OneDimFactorNet) or not isinstance(new, OneDimFactorNet):
        raise TypeError("incompatible factor modules during rank expansion")
    if old.activation_name != new.activation_name or old.depth != new.depth:
        raise ValueError("rank continuation requires identical factor architectures")
    new.layers.load_state_dict(old.layers.state_dict())
    with torch.no_grad():
        new.output.weight[:old_rank].copy_(old.output.weight)
        new.output.bias[:old_rank].copy_(old.output.bias)


def expand_tnn_rank(old_model: ComponentwiseTNN, new_rank: int) -> ComponentwiseTNN:
    if new_rank <= old_model.rank:
        raise ValueError("new rank must exceed old rank")
    device = next(old_model.parameters()).device
    dtype = next(old_model.parameters()).dtype
    new_model = ComponentwiseTNN(
        new_rank,
        old_model.width,
        old_model.depth,
        z_invariant=old_model.z_invariant,
        dominant_constant_mz=old_model.dominant_constant_mz,
        activation=old_model.activation,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        for old_component, new_component in zip(old_model.components, new_model.components):
            copy_factor_module(old_component.phi_x, new_component.phi_x, old_model.rank)
            copy_factor_module(old_component.phi_y, new_component.phi_y, old_model.rank)
            copy_factor_module(old_component.phi_z, new_component.phi_z, old_model.rank)
            new_component.coefficients[: old_model.rank].copy_(old_component.coefficients)
            new_component.coefficients[old_model.rank :].normal_(mean=0.0, std=5.0e-3)
    return new_model


def weighted_component_loss(
    approximate: Tensor,
    exact: Tensor,
    weight3: Tensor,
) -> Tensor:
    """Original weighted componentwise relative-L2 loss, without |m|=1 penalty."""
    errors = []
    for q in range(3):
        numerator = torch.sum(weight3 * (approximate[q] - exact[q]) ** 2)
        denominator = torch.sum(weight3 * exact[q] ** 2).clamp_min(
            torch.finfo(approximate.dtype).eps
        )
        errors.append(numerator / denominator)
    return torch.stack(errors).mean()


@dataclass
class MarginalGradientTargets:
    r"""One-dimensional targets for the ``gradM`` training mode.

    For component q and coordinate d,

        bar m_{q,d}(s) = integral_{[0,1]^2} m_q(x) dx_perp.

    The training target stores bar m and d(bar m)/ds on the one-dimensional
    Gauss nodes.  This avoids forming a three-dimensional TNN prediction during
    optimization.  It is a marginal Sobolev surrogate, not the full volume
    seminorm ||grad(m_TNN-m)||_{L2(Omega)}.
    """

    values: Tensor       # [component, dimension, node]
    derivatives: Tensor  # [component, dimension, node]
    component_means: Tensor  # [component]


def build_marginal_gradient_targets(
    exact: Callable[[Tensor], Tensor],
    nodes: Tensor,
    weights: Tensor,
    *,
    chunk_size: int,
) -> MarginalGradientTargets:
    """Build exact 1D marginal values/derivatives with bounded temporary memory."""
    if chunk_size < 1:
        raise ValueError("gradM target chunk size must be positive")
    n = int(nodes.numel())
    values = torch.empty((3, 3, n), device=nodes.device, dtype=nodes.dtype)
    derivatives = torch.empty_like(values)

    for dim in range(3):
        other = [axis for axis in range(3) if axis != dim]
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            primary = nodes[start:stop]
            b = int(primary.numel())
            shape = (b, n, n)
            coordinate: List[Tensor | None] = [None, None, None]
            coordinate[dim] = primary[:, None, None].expand(shape)
            coordinate[other[0]] = nodes[None, :, None].expand(shape)
            coordinate[other[1]] = nodes[None, None, :].expand(shape)
            points = torch.stack(
                [coordinate[0], coordinate[1], coordinate[2]], dim=-1
            ).requires_grad_(True)
            exact_value = exact(points)
            marginal_value = torch.einsum(
                "bjkq,j,k->bq", exact_value, weights, weights
            )
            values[:, dim, start:stop] = marginal_value.T.detach()

            for q in range(3):
                point_gradient = torch.autograd.grad(
                    exact_value[..., q].sum(),
                    points,
                    retain_graph=(q < 2),
                    create_graph=False,
                )[0][..., dim]
                marginal_derivative = torch.einsum(
                    "bjk,j,k->b", point_gradient, weights, weights
                )
                derivatives[q, dim, start:stop] = marginal_derivative.detach()

    # The integral of any coordinate marginal is the same volume mean.  Average
    # the three numerical versions to suppress tiny quadrature asymmetries.
    means_by_dimension = torch.einsum("qdi,i->qd", values, weights)
    component_means = means_by_dimension.mean(dim=1)
    return MarginalGradientTargets(values, derivatives, component_means)


def component_factor_data_1d(
    component: ScalarTNN,
    nodes: Tensor,
    weights: Tensor,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """Values, derivatives and integrals of all factors using only 1D graphs."""
    x = nodes.reshape(-1, 1)
    factors = [component.phi_x, component.phi_y, component.phi_z]
    values: List[Tensor] = []
    derivatives: List[Tensor] = []
    integrals: List[Tensor] = []
    for factor in factors:
        if isinstance(factor, ConstantFactor):
            value, derivative = factor.forward_with_derivative(x)
        elif isinstance(factor, OneDimFactorNet):
            value, derivative = factor.forward_with_derivative(x)
        else:
            raise TypeError(type(factor))
        values.append(value)
        derivatives.append(derivative)
        integrals.append(torch.sum(weights[:, None] * value, dim=0))
    return values, derivatives, integrals


def marginal_gradient_loss(
    model: ComponentwiseTNN,
    nodes: Tensor,
    weights: Tensor,
    targets: MarginalGradientTargets,
    *,
    mean_weight: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""One-dimensional marginal derivative loss plus zero-mode closure.

    The derivative term is

      average_{q,d} ||d_s bar m^N_{q,d}-d_s bar m_{q,d}||^2 /
                      (||d_s bar m_{q,d}||^2 + small value scale).

    A derivative-only objective cannot determine additive constants.  The
    volume means of the three magnetization components are therefore matched
    with a separately reported zero-mode term.  Both terms use only 1D TNN
    factor values, derivatives and integrals during every optimizer step.
    """
    derivative_terms: List[Tensor] = []
    predicted_means: List[Tensor] = []
    eps = torch.finfo(nodes.dtype).eps
    target_derivative_energy = torch.sum(
        weights[None, None, :] * targets.derivatives.square(), dim=-1
    )
    # Symmetry can make some exact marginal derivatives identically zero.
    # Normalize those channels by the mean nonzero target energy instead of by
    # machine epsilon; otherwise a harmless O(1) initialization gives a loss of
    # order 1/eps and destabilizes the first optimizer steps.
    active = target_derivative_energy > 1024.0 * eps
    if torch.any(active):
        reference_energy = target_derivative_energy[active].mean()
    else:
        reference_energy = torch.ones((), device=nodes.device, dtype=nodes.dtype)

    for q, component in enumerate(model.components):
        values, derivatives, integrals = component_factor_data_1d(
            component, nodes, weights
        )
        coefficient = component.coefficients
        predicted_means.append(
            torch.sum(coefficient * integrals[0] * integrals[1] * integrals[2])
        )

        for dim in range(3):
            amplitude = coefficient
            for other_dim in range(3):
                if other_dim != dim:
                    amplitude = amplitude * integrals[other_dim]
            predicted_derivative = derivatives[dim] @ amplitude
            target_derivative = targets.derivatives[q, dim]
            numerator = torch.sum(
                weights * (predicted_derivative - target_derivative).square()
            )
            derivative_scale = target_derivative_energy[q, dim]
            denominator = torch.maximum(derivative_scale, reference_energy)
            derivative_terms.append(numerator / denominator.clamp_min(64.0 * eps))

    derivative_loss = torch.stack(derivative_terms).mean()
    predicted_mean_tensor = torch.stack(predicted_means)
    mean_loss = torch.mean(
        (predicted_mean_tensor - targets.component_means).square()
    )
    total = derivative_loss + mean_weight * mean_loss
    return total, derivative_loss, mean_loss


def train_tnn_deterministic(
    model: ComponentwiseTNN,
    nodes: Sequence[Tensor] | Tensor,
    weights: Sequence[Tensor] | Tensor,
    exact_tensor: Tensor | None,
    marginal_targets: MarginalGradientTargets | None,
    *,
    loss_function: str,
    gradm_mean_weight: float,
    adam_epochs: int,
    adam_lr: float,
    lbfgs_iters: int,
    print_every: int,
    progress_callback: Callable[[int, float, ComponentwiseTNN], None] | None = None,
) -> Dict[str, float]:
    model.train()
    model.invalidate_factor_normalization_cache()
    weight3 = tensor_weights_3d(weights) if loss_function == "default" else None
    if loss_function == "gradM" and (not isinstance(nodes, Tensor) or not isinstance(weights, Tensor)):
        raise ValueError(
            "gradM currently uses one common 1D rule in all directions. "
            "Use --loss-function default for anisotropic thin-film rules."
        )
    best_loss = math.inf
    best_state = None
    start = time.perf_counter()

    if loss_function == "default" and exact_tensor is None:
        raise ValueError("default loss requires the three-dimensional exact tensor")
    if loss_function == "gradM" and marginal_targets is None:
        raise ValueError("gradM loss requires one-dimensional marginal targets")

    def evaluate_loss() -> Tuple[Tensor, Dict[str, Tensor]]:
        if loss_function == "default":
            assert exact_tensor is not None and weight3 is not None
            approximate = model.evaluate_tensor_grid(nodes)
            fit = weighted_component_loss(approximate, exact_tensor, weight3)
            return fit, {"fit": fit}
        assert marginal_targets is not None
        total, grad_fit, mean_fit = marginal_gradient_loss(
            model,
            nodes,
            weights,
            marginal_targets,
            mean_weight=gradm_mean_weight,
        )
        return total, {"gradM": grad_fit, "mean": mean_fit}

    # Epoch 0 snapshot: initialized/continued model before any Adam update.
    if progress_callback is not None:
        with torch.no_grad():
            initial_total, _ = evaluate_loss()
        progress_callback(0, float(initial_total.detach().cpu()), model)

    if adam_epochs > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(adam_epochs, 1),
            eta_min=adam_lr * 0.05,
        )
        for epoch in range(1, adam_epochs + 1):
            total, parts = evaluate_loss()
            if not torch.isfinite(total):
                raise FloatingPointError("TNN training loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()
            scheduler.step()

            scalar_total = float(total.detach().cpu())
            if progress_callback is not None:
                # The model already contains the parameters after this Adam update.
                # The supplied loss is the loss evaluated immediately before that update;
                # for sparse visualization snapshots this one-step offset is negligible.
                progress_callback(epoch, scalar_total, model)
            if scalar_total < best_loss:
                best_loss = scalar_total
                best_state = {
                    key: tensor.detach().clone()
                    for key, tensor in model.state_dict().items()
                }
            if print_every > 0 and (
                epoch == 1 or epoch % print_every == 0 or epoch == adam_epochs
            ):
                part_text = ", ".join(
                    f"{name}={float(value.detach().cpu()):.6e}"
                    for name, value in parts.items()
                )
                print(
                    f"      Adam {epoch:6d}: total={scalar_total:.6e}, {part_text}"
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    if lbfgs_iters > 0:
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_iters,
            history_size=50,
            tolerance_grad=1.0e-12,
            tolerance_change=1.0e-14,
            line_search_fn="strong_wolfe",
        )
        closure_calls = 0

        def closure() -> Tensor:
            nonlocal closure_calls, best_loss, best_state
            optimizer_lbfgs.zero_grad(set_to_none=True)
            total, _ = evaluate_loss()
            total.backward()
            closure_calls += 1
            scalar_total = float(total.detach().cpu())
            if scalar_total < best_loss:
                best_loss = scalar_total
                best_state = {
                    key: tensor.detach().clone()
                    for key, tensor in model.state_dict().items()
                }
            if print_every > 0 and closure_calls % max(print_every // 10, 20) == 0:
                print(
                    f"      L-BFGS closure {closure_calls:5d}: "
                    f"total={scalar_total:.6e}"
                )
            return total

        optimizer_lbfgs.step(closure)
        if best_state is not None:
            model.load_state_dict(best_state)

    model.eval()
    model.refresh_factor_normalization_cache()
    normalization_error = model.maximum_factor_normalization_error()
    elapsed = time.perf_counter() - start
    print(
        f"      factor normalization max error={normalization_error:.3e} "
        "(quadrature L2 norms)"
    )
    return {
        "training_seconds": elapsed,
        "best_training_loss": best_loss,
        "factor_normalization_max_error": normalization_error,
    }


# =============================================================================
