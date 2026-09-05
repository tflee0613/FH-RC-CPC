"""Lossless compression of task outcomes by epsilon-good signatures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .failure_hypergraph import Controller, FailureHypergraph
from .general_portfolio import _uses_integer_weight_contract


@dataclass(frozen=True)
class PortfolioEvaluation:
    members: tuple[int, ...]
    cost: float
    total_miss_count: int
    total_weighted_miss: float
    group_misses: dict[str, int]
    group_weighted_misses: dict[str, float]


@dataclass(frozen=True)
class DecisionStabilityAudit:
    """Finite audit of the conditional uniform-approximation regret bound."""

    selected_index: int
    oracle_index: int
    uniform_error: float
    true_regret: float
    regret_bound: float
    valid: bool


@dataclass(frozen=True)
class ConstraintMarginAudit:
    """Check the sufficient group-miss margins used in the Supplement."""

    estimated_misses: dict[str, float]
    error_bounds: dict[str, float]
    allowances: dict[str, float]
    certified_groups: dict[str, bool]
    all_groups_certified: bool


@dataclass(frozen=True)
class FailureSignatureTable:
    controller_names: tuple[str, ...]
    controller_costs: np.ndarray
    signatures: np.ndarray
    signature_counts: np.ndarray
    signature_weights: np.ndarray
    group_counts: Mapping[str, np.ndarray]
    group_weights: Mapping[str, np.ndarray]
    task_signature_index: np.ndarray
    integer_weight_contract: bool = False

    def __post_init__(self) -> None:
        names = tuple(self.controller_names)
        signatures = np.asarray(self.signatures)
        if signatures.dtype != np.bool_ or signatures.ndim != 2:
            raise ValueError("signatures must be a boolean matrix")
        n_signatures, n_controllers = signatures.shape
        if (
            n_signatures <= 0
            or len(names) != n_controllers
            or any(not isinstance(name, str) or not name for name in names)
        ):
            raise ValueError("signature dimensions must align with controller names")
        if len(set(names)) != len(names):
            raise ValueError("controller names must be unique")
        rows = [tuple(bool(value) for value in row) for row in signatures]
        if len(set(rows)) != n_signatures:
            raise ValueError("signature rows must be unique")

        costs = np.asarray(self.controller_costs, dtype=float)
        counts = np.asarray(self.signature_counts)
        weights = np.asarray(self.signature_weights, dtype=float)
        task_index = np.asarray(self.task_signature_index)
        if (
            costs.shape != (n_controllers,)
            or not np.all(np.isfinite(costs))
            or np.any(costs <= 0)
        ):
            raise ValueError("controller costs must be finite, positive, and aligned")
        if (
            counts.shape != (n_signatures,)
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts <= 0)
        ):
            raise ValueError(
                "signature counts must be an aligned positive integer vector"
            )
        if (
            weights.shape != (n_signatures,)
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0)
        ):
            raise ValueError("signature weights must be finite, positive, and aligned")
        if (
            task_index.ndim != 1
            or not np.issubdtype(task_index.dtype, np.integer)
            or len(task_index) != int(counts.sum())
        ):
            raise ValueError("task_signature_index must reconstruct every task")
        if np.any(task_index < 0) or np.any(task_index >= n_signatures):
            raise ValueError("task signature indices are out of range")

        group_counts: dict[str, np.ndarray] = {}
        group_weights: dict[str, np.ndarray] = {}
        if set(self.group_counts) != set(self.group_weights) or not self.group_counts:
            raise ValueError(
                "group count and weight mappings must be nonempty and align"
            )
        for group in sorted(self.group_counts):
            if not isinstance(group, str) or not group:
                raise ValueError("group names must be nonempty strings")
            g_count = np.asarray(self.group_counts[group])
            g_weight = np.asarray(self.group_weights[group], dtype=float)
            if (
                g_count.shape != (n_signatures,)
                or not np.issubdtype(g_count.dtype, np.integer)
                or np.any(g_count < 0)
            ):
                raise ValueError("group counts must be aligned integer vectors")
            if (
                g_weight.shape != (n_signatures,)
                or not np.all(np.isfinite(g_weight))
                or np.any(g_weight < 0)
            ):
                raise ValueError(
                    "group weights must be finite, aligned nonnegative vectors"
                )
            if int(g_count.sum()) <= 0 or float(g_weight.sum()) <= 0:
                raise ValueError("groups cannot be empty")
            g_count = g_count.astype(np.int64, copy=True)
            g_weight = g_weight.astype(float, copy=True)
            if self.integer_weight_contract:
                if not _uses_integer_weight_contract(g_weight):
                    raise ValueError("integer-contract group weights must be integral")
            g_count.setflags(write=False)
            g_weight.setflags(write=False)
            group_counts[group] = g_count
            group_weights[group] = g_weight

        signatures = signatures.astype(bool, copy=True)
        costs = costs.astype(float, copy=True)
        counts = counts.astype(np.int64, copy=True)
        weights = weights.astype(float, copy=True)
        if self.integer_weight_contract:
            if not _uses_integer_weight_contract(weights):
                raise ValueError("integer-contract signature weights must be integral")
        task_index = task_index.astype(np.int64, copy=True)
        for value in (signatures, costs, counts, weights, task_index):
            value.setflags(write=False)
        object.__setattr__(self, "controller_names", names)
        object.__setattr__(self, "controller_costs", costs)
        object.__setattr__(self, "signatures", signatures)
        object.__setattr__(self, "signature_counts", counts)
        object.__setattr__(self, "signature_weights", weights)
        object.__setattr__(self, "group_counts", MappingProxyType(group_counts))
        object.__setattr__(self, "group_weights", MappingProxyType(group_weights))
        object.__setattr__(self, "task_signature_index", task_index)
        object.__setattr__(
            self, "integer_weight_contract", bool(self.integer_weight_contract)
        )

    @property
    def n_tasks(self) -> int:
        return int(self.signature_counts.sum())

    @property
    def n_signatures(self) -> int:
        return int(len(self.signatures))

    @property
    def n_controllers(self) -> int:
        return int(self.signatures.shape[1])


def _guard_signature_weight_arithmetic(
    graph: FailureHypergraph, table: FailureSignatureTable
) -> None:
    """Prove exact reductions or audit every portfolio within a fixed work cap."""

    fallback = "use the original-task solver or explicitly registered exact weights"
    with np.errstate(over="ignore"):
        total = float(graph.task_weights.sum())
    if not np.isfinite(total):
        raise ValueError(f"nonfinite compression weight total; {fallback}")
    ratios = [float(weight).as_integer_ratio() for weight in graph.task_weights]
    denominator = max(pair[1] for pair in ratios)
    units = [numerator * (denominator // divisor) for numerator, divisor in ratios]
    shift = min((value & -value).bit_length() - 1 for value in units)
    if (sum(units) >> shift) <= 2**53:
        return  # Every positive subset and every association is representable.

    channels = [(np.ones(graph.n_tasks, bool), table.signature_weights)] + [
        (domain, table.group_weights[group]) for group, domain in graph.groups.items()
    ]
    states = 1 << graph.n_controllers
    if states * graph.n_tasks * len(channels) > 1_000_000:
        raise ValueError(f"compression arithmetic audit exceeds work limit; {fallback}")
    for mask in range(states):  # Empty portfolio also checks every denominator.
        members = [i for i in range(graph.n_controllers) if mask & (1 << i)]
        missed = ~table.signatures[:, members].any(axis=1)
        source_missed = missed[table.task_signature_index]
        for domain, compressed_weights in channels:
            raw = float(graph.task_weights[domain & source_missed].sum())
            compressed = float(compressed_weights[missed].sum())
            if raw != compressed:
                raise ValueError(
                    f"compression changes a floating-weight sum at mask {mask}; {fallback}"
                )


def compress_failure_signatures(graph: FailureHypergraph) -> FailureSignatureTable:
    """Aggregate identical outcome bitmasks without changing any group measure."""

    signatures, inverse, counts = np.unique(
        graph.good, axis=0, return_inverse=True, return_counts=True
    )
    n_signatures = len(signatures)
    signature_weights = np.bincount(
        inverse, weights=graph.task_weights, minlength=n_signatures
    )
    group_counts: dict[str, np.ndarray] = {}
    group_weights: dict[str, np.ndarray] = {}
    for group, membership in graph.groups.items():
        group_counts[group] = np.bincount(
            inverse[membership], minlength=n_signatures
        ).astype(np.int64)
        group_weights[group] = np.bincount(
            inverse[membership],
            weights=graph.task_weights[membership],
            minlength=n_signatures,
        )
    table = FailureSignatureTable(
        controller_names=graph.controller_names,
        controller_costs=graph.controller_costs,
        signatures=signatures,
        signature_counts=counts,
        signature_weights=signature_weights,
        group_counts=group_counts,
        group_weights=group_weights,
        task_signature_index=inverse,
        integer_weight_contract=_uses_integer_weight_contract(graph.task_weights),
    )
    _guard_signature_weight_arithmetic(graph, table)
    return table


def _indices(
    members: Sequence[Controller], controller_names: tuple[str, ...]
) -> tuple[int, ...]:
    indices: list[int] = []
    for member in members:
        if isinstance(member, str):
            try:
                index = controller_names.index(member)
            except ValueError as error:
                raise KeyError(f"unknown controller: {member}") from error
        else:
            index = int(member)
            if index < 0 or index >= len(controller_names):
                raise IndexError(f"controller index out of range: {index}")
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError("portfolio members must be unique")
    return tuple(sorted(indices))


def evaluate_task_portfolio(
    graph: FailureHypergraph, members: Sequence[Controller]
) -> PortfolioEvaluation:
    indices = _indices(members, graph.controller_names)
    missed = graph.common_failure(indices)
    return PortfolioEvaluation(
        members=indices,
        cost=float(graph.controller_costs[list(indices)].sum()) if indices else 0.0,
        total_miss_count=int(missed.sum()),
        total_weighted_miss=float(graph.task_weights[missed].sum()),
        group_misses={
            group: int(np.sum(membership & missed))
            for group, membership in graph.groups.items()
        },
        group_weighted_misses={
            group: float(graph.task_weights[membership & missed].sum())
            for group, membership in graph.groups.items()
        },
    )


def evaluate_signature_portfolio(
    table: FailureSignatureTable, members: Sequence[Controller]
) -> PortfolioEvaluation:
    indices = _indices(members, table.controller_names)
    covered = (
        table.signatures[:, list(indices)].any(axis=1)
        if indices
        else np.zeros(table.n_signatures, dtype=bool)
    )
    missed = ~covered
    return PortfolioEvaluation(
        members=indices,
        cost=float(table.controller_costs[list(indices)].sum()) if indices else 0.0,
        total_miss_count=int(table.signature_counts[missed].sum()),
        total_weighted_miss=float(table.signature_weights[missed].sum()),
        group_misses={
            group: int(values[missed].sum())
            for group, values in table.group_counts.items()
        },
        group_weighted_misses={
            group: float(values[missed].sum())
            for group, values in table.group_weights.items()
        },
    )


def audit_decision_stability(
    true_values: np.ndarray,
    approximate_values: np.ndarray,
) -> DecisionStabilityAudit:
    """Verify the conditional ``2 * uniform_error`` selection-regret bound.

    The result is descriptive, not a statistical certificate: the theorem
    applies only when the supplied approximation errors are uniformly bounded
    over the complete finite decision set.
    """

    truth = np.asarray(true_values, dtype=float)
    approximation = np.asarray(approximate_values, dtype=float)
    if truth.ndim != 1 or not len(truth) or approximation.shape != truth.shape:
        raise ValueError("true and approximate values must be aligned nonempty vectors")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(approximation)):
        raise ValueError("decision values must be finite")
    selected = int(np.argmin(approximation))
    oracle = int(np.argmin(truth))
    uniform_error = float(np.max(np.abs(truth - approximation)))
    regret = float(truth[selected] - truth[oracle])
    bound = 2.0 * uniform_error
    return DecisionStabilityAudit(
        selected_index=selected,
        oracle_index=oracle,
        uniform_error=uniform_error,
        true_regret=regret,
        regret_bound=bound,
        valid=bool(regret <= bound + 1e-12),
    )


def audit_constraint_margins(
    estimated_misses: Mapping[str, float],
    error_bounds: Mapping[str, float],
    group_weights: Mapping[str, float],
    miss_budgets: Mapping[str, float],
) -> ConstraintMarginAudit:
    """Verify ``estimated_miss <= allowance - error_bound`` for every group.

    When supplied error bounds are valid for the true group misses, satisfying
    this condition is sufficient for true feasibility.  This function checks
    the deterministic margin premise; it does not estimate statistical error
    bounds from data.
    """

    keys = set(estimated_misses)
    if not keys or any(
        set(mapping) != keys for mapping in (error_bounds, group_weights, miss_budgets)
    ):
        raise ValueError("all group mappings must contain the same nonempty keys")
    estimates = {str(key): float(estimated_misses[key]) for key in sorted(keys)}
    errors = {str(key): float(error_bounds[key]) for key in sorted(keys)}
    totals = {str(key): float(group_weights[key]) for key in sorted(keys)}
    budgets = {str(key): float(miss_budgets[key]) for key in sorted(keys)}
    if any(
        not np.isfinite(value) or value < 0
        for value in (*estimates.values(), *errors.values())
    ):
        raise ValueError(
            "estimated misses and error bounds must be finite and nonnegative"
        )
    if any(not np.isfinite(value) or value <= 0 for value in totals.values()):
        raise ValueError("group weights must be finite and positive")
    if any(
        not np.isfinite(value) or value < 0 or value > 1 for value in budgets.values()
    ):
        raise ValueError("miss budgets must lie in [0, 1]")
    allowances = {key: budgets[key] * totals[key] for key in sorted(keys)}
    certified = {
        key: estimates[key] <= allowances[key] - errors[key] + 1e-12
        for key in sorted(keys)
    }
    return ConstraintMarginAudit(
        estimated_misses=estimates,
        error_bounds=errors,
        allowances=allowances,
        certified_groups=certified,
        all_groups_certified=all(certified.values()),
    )
