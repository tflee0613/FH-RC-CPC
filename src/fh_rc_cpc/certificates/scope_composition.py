"""Bounded intersection of separately registered finite qualification scopes.

No rows are pooled and no old quota is recomputed after adding a scope. Costs
are exact represented binary-rational sums; feasibility retains each scope's
existing non-expanding IEEE-754 contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb
from operator import index
from types import MappingProxyType

import numpy as np

from ..qualification.failure_hypergraph import FailureHypergraph
from ..qualification.general_portfolio import (
    _uses_integer_weight_contract,
    _weighted_miss_allowance,
    _weighted_miss_is_feasible,
)


class ScopeCompositionError(RuntimeError):
    """Composition exceeds its finite-work domain or fails verification."""


@dataclass(frozen=True)
class ScopeContract:
    name: str
    graph: FailureHypergraph
    miss_budgets: Mapping[str, float]
    empty_groups: tuple[str, ...] = ()
    allowances: Mapping[str, float] = field(init=False)
    integer_weight_contract: bool = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("scope name must be a nonempty string")
        if not isinstance(self.graph, FailureHypergraph):
            raise ValueError("scope graph must be a FailureHypergraph")
        empty = tuple(self.empty_groups)
        if (
            any(not isinstance(group, str) or not group for group in empty)
            or len(set(empty)) != len(empty)
            or set(empty) & set(self.graph.groups)
        ):
            raise ValueError(
                "empty group names must be unique and disjoint from active groups"
            )
        if set(self.miss_budgets) != set(self.graph.groups) | set(empty):
            raise ValueError(
                "miss_budgets must register all active and empty groups exactly once"
            )
        budgets = {group: float(value) for group, value in self.miss_budgets.items()}
        if any(
            not np.isfinite(value) or not 0 <= value <= 1 for value in budgets.values()
        ):
            raise ValueError("miss budgets must be finite and lie in [0, 1]")
        # Copy even an already frozen graph: caller-owned arrays never alias
        # this registration snapshot, including its costs and group masks.
        graph = FailureHypergraph(
            self.graph.good,
            self.graph.controller_names,
            self.graph.controller_costs,
            self.graph.groups,
            self.graph.task_weights,
        )
        integral = _uses_integer_weight_contract(graph.task_weights)
        allowances = {}
        for group in sorted(budgets):
            if group in empty:
                allowances[group] = 0.0
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                allowance = _weighted_miss_allowance(
                    graph.task_weights[graph.groups[group]],
                    budgets[group],
                    integer_weight_contract=integral,
                )
            if not np.isfinite(allowance):
                raise ScopeCompositionError(
                    f"nonfinite allowance in {(self.name, group)}"
                )
            allowances[group] = allowance
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "miss_budgets", MappingProxyType(budgets))
        object.__setattr__(self, "empty_groups", tuple(sorted(empty)))
        object.__setattr__(self, "allowances", MappingProxyType(allowances))
        object.__setattr__(self, "integer_weight_contract", integral)


@dataclass(frozen=True)
class GroupCertificate:
    scope: str
    group: str
    population: int
    population_weight: float
    misses: int
    weighted_misses: float
    allowance: float
    slack: float
    passed: bool | None
    status: str

    @property
    def key(self) -> tuple[str, str]:
        return self.scope, self.group


@dataclass(frozen=True)
class ScopePortfolioCertificate:
    scope: str
    members: tuple[str, ...]
    integer_weight_contract: bool
    groups: tuple[GroupCertificate, ...]
    passed: bool


@dataclass(frozen=True)
class ScopeFrontier:
    scope: str
    feasible_masks: tuple[int, ...]
    minimum_cost: Fraction | None
    optimal_masks: tuple[int, ...]


@dataclass(frozen=True)
class CompositionPortfolio:
    mask: int
    members: tuple[str, ...]
    cost: Fraction
    scope_certificates: tuple[ScopePortfolioCertificate, ...]


@dataclass(frozen=True)
class ScopeCompositionResult:
    controller_names: tuple[str, ...]
    controller_costs: tuple[tuple[str, Fraction], ...]
    max_size: int
    allow_empty: bool
    enumerated_masks: int
    scope_frontiers: tuple[ScopeFrontier, ...]
    feasible_masks: tuple[int, ...]
    status: str
    minimum_cost: Fraction | None
    optimal_masks: tuple[int, ...]
    optimal_portfolios: tuple[CompositionPortfolio, ...]


def evaluate_scope(
    scope: ScopeContract,
    members: Sequence[str],
) -> ScopePortfolioCertificate:
    """Recompute per-scope evidence from original columns, including empty groups."""

    if isinstance(members, str):
        raise ValueError("members must be a sequence of controller IDs, not a string")
    selected = tuple(members)
    if (
        any(not isinstance(name, str) for name in selected)
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(scope.graph.controller_names)
    ):
        raise ValueError("members must be unique registered controller IDs")
    selected = tuple(sorted(selected))
    graph = scope.graph
    columns = [graph.controller_names.index(name) for name in selected]
    missed = (
        ~graph.good[:, columns].any(axis=1) if columns else np.ones(graph.n_tasks, bool)
    )
    groups = []
    for group, allowance in scope.allowances.items():
        if group in scope.empty_groups:
            groups.append(
                GroupCertificate(
                    scope.name,
                    group,
                    0,
                    0.0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    None,
                    "empty_not_exercised",
                )
            )
            continue
        domain = graph.groups[group]
        failure = domain & missed
        mass = float(graph.task_weights[failure].sum())
        passed = _weighted_miss_is_feasible(
            mass,
            allowance,
            integer_weight_contract=scope.integer_weight_contract,
        )
        groups.append(
            GroupCertificate(
                scope.name,
                group,
                int(domain.sum()),
                float(graph.task_weights[domain].sum()),
                int(failure.sum()),
                mass,
                allowance,
                allowance - mass,
                passed,
                "passed" if passed else "violated",
            )
        )
    return ScopePortfolioCertificate(
        scope.name,
        selected,
        scope.integer_weight_contract,
        tuple(groups),
        all(g.passed is not False for g in groups),
    )


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        result = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def compose_scopes(
    scopes: Sequence[ScopeContract],
    *,
    max_size: int | None = None,
    allow_empty: bool = False,
    max_controllers: int = 12,
    max_scope_checks: int = 1_000_000,
) -> ScopeCompositionResult:
    """Intersect complete finite feasible sets and return all exact cost optima.

    A cap/empty convention is common to every scope. Masks use sorted common
    controller IDs. Empty intersections and empty admissible families return
    an explicit infeasible result; exceeded work guards raise before enumeration.
    """

    scopes = tuple(scopes)
    if not scopes or any(not isinstance(scope, ScopeContract) for scope in scopes):
        raise ValueError(
            "composition requires a nonempty sequence of ScopeContract scopes"
        )
    if len({scope.name for scope in scopes}) != len(scopes):
        raise ValueError("scope names must be unique")
    guard = _integer(max_controllers, "max_controllers", 1)
    if guard > 20:
        raise ValueError("max_controllers cannot exceed the small-menu ceiling 20")
    check_limit = _integer(max_scope_checks, "max_scope_checks", 1)
    if not isinstance(allow_empty, (bool, np.bool_)):
        raise ValueError("allow_empty must be boolean")
    names = tuple(sorted(scopes[0].graph.controller_names))
    k = len(names)
    if k > guard:
        raise ScopeCompositionError(
            f"controller enumeration requires {k}; limit={guard}"
        )
    cap = k if max_size is None else _integer(max_size, "max_size")
    if cap > k:
        raise ValueError("max_size cannot exceed the controller count")
    cost_map = dict(
        zip(
            scopes[0].graph.controller_names,
            map(
                lambda cost: Fraction.from_float(float(cost)),
                scopes[0].graph.controller_costs,
            ),
        )
    )
    for scope in scopes:
        scope_costs = {
            name: Fraction.from_float(float(cost))
            for name, cost in zip(
                scope.graph.controller_names, scope.graph.controller_costs
            )
        }
        if set(scope_costs) != set(cost_map):
            raise ValueError("every scope must register the same controller IDs")
        if scope_costs != cost_map:
            raise ValueError(
                "every scope must register the same exact controller cost map"
            )
    count = sum(comb(k, size) for size in range(cap + 1)) - (not allow_empty)
    checks = count * sum(len(scope.graph.groups) for scope in scopes)
    if checks > check_limit:
        raise ScopeCompositionError(
            f"composition needs {checks} scope/group checks; limit={check_limit}"
        )
    masks = tuple(
        mask
        for mask in range(1 << k)
        if mask.bit_count() <= cap and (mask != 0 or allow_empty)
    )
    costs = {
        mask: sum(
            (cost_map[name] for i, name in enumerate(names) if mask & (1 << i)),
            Fraction(),
        )
        for mask in masks
    }
    frontiers = []
    intersection = set(masks)
    for scope in scopes:
        # A distinct computation from evaluate_scope: use common-ID row
        # signatures to construct the feasible mask set before verification.
        graph = scope.graph
        row_signatures = np.zeros(graph.n_tasks, dtype=np.uint32)
        for local, name in enumerate(graph.controller_names):
            row_signatures |= graph.good[:, local].astype(np.uint32) << names.index(
                name
            )
        feasible = []
        for mask in masks:
            missed = (row_signatures & mask) == 0
            if all(
                _weighted_miss_is_feasible(
                    float(graph.task_weights[domain & missed].sum()),
                    scope.allowances[group],
                    integer_weight_contract=scope.integer_weight_contract,
                )
                for group, domain in graph.groups.items()
            ):
                feasible.append(mask)
        minimum = min((costs[mask] for mask in feasible), default=None)
        optimal = tuple(mask for mask in feasible if costs[mask] == minimum)
        frontiers.append(ScopeFrontier(scope.name, tuple(feasible), minimum, optimal))
        intersection.intersection_update(feasible)
    feasible = tuple(sorted(intersection))
    minimum = min((costs[mask] for mask in feasible), default=None)
    optimal = tuple(mask for mask in feasible if costs[mask] == minimum)
    portfolios = []
    for mask in optimal:
        members = tuple(name for i, name in enumerate(names) if mask & (1 << i))
        certificates = tuple(evaluate_scope(scope, members) for scope in scopes)
        if not all(certificate.passed for certificate in certificates):
            raise ScopeCompositionError(
                "intersection optimum failed original-scope verification"
            )
        portfolios.append(
            CompositionPortfolio(mask, members, costs[mask], certificates)
        )
    return ScopeCompositionResult(
        names,
        tuple((name, cost_map[name]) for name in names),
        cap,
        bool(allow_empty),
        count,
        tuple(frontiers),
        feasible,
        "optimal" if optimal else "infeasible",
        minimum,
        optimal,
        tuple(portfolios),
    )
