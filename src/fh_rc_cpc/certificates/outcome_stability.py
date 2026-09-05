"""Exact, bounded fixed-incumbent robustness to whole-task outcome changes.

The universe, memberships, weights, allowances, costs and admissible portfolios
are frozen. Distances count replaced rows, not flipped entries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import prod
from operator import index

import numpy as np

from ..qualification.failure_hypergraph import Controller, FailureHypergraph
from ..qualification.general_portfolio import (
    _uses_integer_weight_contract,
    _weighted_miss_allowance,
    _weighted_miss_is_feasible,
)


class OutcomeCertificationError(RuntimeError):
    """Exact certification is unsupported or an independent check failed."""


@dataclass(frozen=True)
class OutcomeAtom:
    outcome: tuple[bool, ...]
    groups: tuple[str, ...]
    weight: Fraction
    rows: tuple[int, ...]


@dataclass(frozen=True)
class RowChangeWitness:
    kind: str
    members: tuple[int, ...]
    rows: tuple[int, ...]
    replacement_rows: tuple[tuple[bool, ...], ...]
    violated_group: str | None = None


@dataclass(frozen=True)
class RowChangeDistance:
    """``distance=None`` means infinity, not an unsolved finite problem."""

    distance: int | None
    witness: RowChangeWitness | None
    method: str
    states_checked: int = 0


@dataclass(frozen=True)
class OutcomeStabilityCertificate:
    incumbent: tuple[int, ...]
    incumbent_cost: Fraction
    n_tasks: int
    max_size: int
    allow_empty: bool
    allowances: tuple[tuple[str, Fraction], ...]
    destruction: RowChangeDistance
    cheaper_repair: RowChangeDistance
    distance: int | None
    radius: int
    witness: RowChangeWitness | None
    cheaper_portfolios: int


@dataclass(frozen=True)
class ReserveQualifiedPortfolio:
    members: tuple[int, ...]
    destruction: RowChangeDistance


@dataclass(frozen=True)
class RowReserveQualification:
    """Exact reserve-feasibility optimum, not persistence of nominal optimality."""

    row_reserve: int
    n_tasks: int
    max_size: int
    allow_empty: bool
    allowances: tuple[tuple[str, Fraction], ...]
    status: str
    minimum_cost: Fraction | None
    optima: tuple[ReserveQualifiedPortfolio, ...]
    admissible_portfolios: int
    robust_feasible_portfolios: int


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


def _members(
    graph: FailureHypergraph, members: Sequence[Controller]
) -> tuple[int, ...]:
    result = tuple(
        graph.controller_index(
            member if isinstance(member, str) else _integer(member, "controller index")
        )
        for member in members
    )
    if len(set(result)) != len(result):
        raise ValueError("portfolio members must be unique")
    return tuple(sorted(result))


def joint_outcome_atoms(graph: FailureHypergraph) -> tuple[OutcomeAtom, ...]:
    """Partition *all* rows by outcomes, joint memberships and exact weight."""

    rows_by_key: dict[tuple, list[int]] = {}
    for row in range(graph.n_tasks):
        key = (
            tuple(bool(value) for value in graph.good[row]),
            tuple(group for group, domain in graph.groups.items() if domain[row]),
            Fraction.from_float(float(graph.task_weights[row])),
        )
        rows_by_key.setdefault(key, []).append(row)
    return tuple(OutcomeAtom(*key, tuple(rows)) for key, rows in rows_by_key.items())


@dataclass(frozen=True)
class _Contract:
    graph: FailureHypergraph
    weights: tuple[Fraction, ...]
    allowances: tuple[tuple[str, Fraction], ...]
    integer_weight_contract: bool

    def masses(self, missed: np.ndarray) -> dict[str, Fraction]:
        return {
            group: sum(
                (self.weights[row] for row in np.flatnonzero(domain & missed)),
                Fraction(),
            )
            for group, domain in self.graph.groups.items()
        }


def _subset_sums_exact(weights: tuple[Fraction, ...]) -> bool:
    """Sufficient dyadic-lattice proof for every positive subset/reduction."""

    denominator = max(weight.denominator for weight in weights)
    numerators = [
        weight.numerator * (denominator // weight.denominator) for weight in weights
    ]
    # Removing a common power of two also admits large integral row weights.
    shift = min((number & -number).bit_length() - 1 for number in numerators)
    return (sum(numerators) >> shift) <= 2**53


def _contract(
    graph: FailureHypergraph,
    miss_budgets: Mapping[str, float],
    max_arithmetic_states: int,
) -> _Contract:
    limit = _integer(max_arithmetic_states, "max_arithmetic_states", 1)
    if set(miss_budgets) != set(graph.groups):
        raise ValueError("miss_budgets must contain every group exactly once")
    budgets = {group: float(value) for group, value in miss_budgets.items()}
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in budgets.values()):
        raise ValueError("group miss budgets must lie in [0, 1]")
    weights = tuple(Fraction.from_float(float(weight)) for weight in graph.task_weights)
    integral = _uses_integer_weight_contract(graph.task_weights)
    allowances = []
    for group, domain in graph.groups.items():
        with np.errstate(over="ignore", invalid="ignore"):
            bound = _weighted_miss_allowance(
                graph.task_weights[domain],
                budgets[group],
                integer_weight_contract=integral,
            )
        if not np.isfinite(bound):
            raise OutcomeCertificationError(
                f"nonfinite frozen allowance in group {group}"
            )
        allowance = Fraction.from_float(bound)
        allowances.append((group, allowance))
        rows = tuple(int(row) for row in np.flatnonzero(domain))
        group_weights = tuple(weights[row] for row in rows)
        if _subset_sums_exact(group_weights):
            continue
        states = 1 << len(rows)
        if states > limit:
            raise OutcomeCertificationError(
                f"arithmetic audit for group {group} needs {states} subsets; limit={limit}"
            )
        # Witness-only checks cannot establish universal compatibility. Audit
        # every original-row subset when the sufficient lattice proof fails.
        for bits in range(states):
            chosen = [rows[i] for i in range(len(rows)) if bits & (1 << i)]
            exact = sum((weights[row] for row in chosen), Fraction()) <= allowance
            core = _weighted_miss_is_feasible(
                float(graph.task_weights[chosen].sum()),
                bound,
                integer_weight_contract=integral,
            )
            if exact != core:
                raise OutcomeCertificationError(
                    f"exact/core arithmetic disagreement in group {group} on rows {chosen}"
                )
    return _Contract(graph, weights, tuple(allowances), integral)


def _missed(good: np.ndarray, members: tuple[int, ...]) -> np.ndarray:
    return (
        ~good[:, list(members)].any(axis=1)
        if members
        else np.ones(len(good), dtype=bool)
    )


def apply_row_change_witness(
    graph: FailureHypergraph, witness: RowChangeWitness
) -> np.ndarray:
    """Apply an explicit witness to a copy, rejecting malformed/no-op changes."""

    _members(graph, witness.members)
    if witness.kind not in {"failure", "repair"}:
        raise ValueError("witness kind must be failure or repair")
    rows = tuple(_integer(row, "row index") for row in witness.rows)
    if len(set(rows)) != len(rows) or any(row >= graph.n_tasks for row in rows):
        raise ValueError("witness row indices must be unique and in bounds")
    if len(witness.replacement_rows) != len(rows):
        raise ValueError("replacement rows must align with row indices")
    changed = graph.good.copy()
    if rows:
        values = np.asarray(witness.replacement_rows)
        if values.dtype != np.bool_ or values.shape != (len(rows), graph.n_controllers):
            raise ValueError("replacement rows must be an aligned boolean matrix")
        if not np.all(np.any(values != graph.good[list(rows)], axis=1)):
            raise ValueError("every listed row must actually change")
        changed[list(rows)] = values
    return changed


def _verify_witness(contract: _Contract, witness: RowChangeWitness) -> bool:
    graph = contract.graph
    changed = apply_row_change_witness(graph, witness)
    missed = _missed(changed, _members(graph, witness.members))
    statuses = {}
    for group, allowance in contract.allowances:
        rows = np.flatnonzero(graph.groups[group] & missed)
        exact = sum((contract.weights[row] for row in rows), Fraction()) <= allowance
        core = _weighted_miss_is_feasible(
            float(graph.task_weights[rows].sum()),
            float(allowance),
            integer_weight_contract=contract.integer_weight_contract,
        )
        if exact != core:
            raise OutcomeCertificationError(
                "witness disagrees with the frozen core boundary"
            )
        statuses[group] = exact
    if witness.kind == "repair":
        return all(statuses.values()) and witness.violated_group is None
    if witness.violated_group is not None:
        return (
            witness.violated_group in statuses and not statuses[witness.violated_group]
        )
    return not all(statuses.values())


def verify_row_change_witness(
    graph: FailureHypergraph,
    witness: RowChangeWitness,
    *,
    miss_budgets: Mapping[str, float],
    max_arithmetic_states: int = 65_536,
) -> bool:
    """Independently verify attainment, not minimum distance; bad witnesses fail."""

    contract = _contract(graph, miss_budgets, max_arithmetic_states)
    try:
        return _verify_witness(contract, witness)
    except (ValueError, TypeError, IndexError, KeyError, AttributeError):
        return False


def _distance_with_witness(
    contract: _Contract,
    members: tuple[int, ...],
    rows: tuple[int, ...],
    kind: str,
    method: str,
    states: int = 0,
    violated_group: str | None = None,
) -> RowChangeDistance:
    rows = tuple(sorted(rows))
    replacements = []
    for row in rows:
        replacement = contract.graph.good[row].copy()
        if kind == "failure":
            replacement[:] = False
        else:
            replacement[members[0]] = True
        replacements.append(tuple(bool(value) for value in replacement))
    witness = RowChangeWitness(kind, members, rows, tuple(replacements), violated_group)
    if not _verify_witness(contract, witness):
        raise OutcomeCertificationError(
            "constructed row witness failed independent verification"
        )
    return RowChangeDistance(len(rows), witness, method, states)


def _failure_distance(
    contract: _Contract, members: tuple[int, ...]
) -> RowChangeDistance:
    graph = contract.graph
    missed = _missed(graph.good, members)
    masses = contract.masses(missed)
    best: tuple[tuple[int, ...], str] | None = None
    for group, allowance in contract.allowances:
        if masses[group] > allowance:
            return _distance_with_witness(
                contract,
                members,
                (),
                "failure",
                "already_infeasible",
                violated_group=group,
            )
        candidates = sorted(
            np.flatnonzero(graph.groups[group] & ~missed),
            key=lambda row: (-contract.weights[row], int(row)),
        )
        weight = masses[group]
        selected = []
        for row in candidates:
            selected.append(int(row))
            weight += contract.weights[row]
            if weight > allowance:
                candidate = (tuple(sorted(selected)), group)
                if best is None or (len(candidate[0]), candidate) < (
                    len(best[0]),
                    best,
                ):
                    best = candidate
                break
    if best is None:
        return RowChangeDistance(None, None, "no_group_can_be_violated")
    return _distance_with_witness(
        contract,
        members,
        best[0],
        "failure",
        "descending_weight_prefix",
        violated_group=best[1],
    )


def portfolio_failure_distance(
    graph: FailureHypergraph,
    members: Sequence[Controller],
    *,
    miss_budgets: Mapping[str, float],
    max_arithmetic_states: int = 65_536,
) -> RowChangeDistance:
    """First number of arbitrary row replacements making this portfolio infeasible."""

    return _failure_distance(
        _contract(graph, miss_budgets, max_arithmetic_states), _members(graph, members)
    )


def qualification_with_row_reserve(
    graph: FailureHypergraph,
    *,
    row_reserve: int,
    miss_budgets: Mapping[str, float],
    max_size: int | None = None,
    allow_empty: bool = True,
    max_controllers: int = 12,
    max_arithmetic_states: int = 65_536,
) -> RowReserveQualification:
    """All exact minimum-cost bases feasible after any <= reserve row changes.

    A basis passes iff its destruction distance is infinite or exceeds the
    requested reserve. No cheaper-competitor repair search is needed. Empty
    feasible families return status ``infeasible``, not an invented basis.
    """

    reserve = _integer(row_reserve, "row_reserve")
    if reserve > graph.n_tasks:
        raise ValueError("row_reserve cannot exceed the task count")
    controller_limit = _integer(max_controllers, "max_controllers", 1)
    if controller_limit > 20:
        raise ValueError("max_controllers cannot exceed the small-menu ceiling 20")
    if graph.n_controllers > controller_limit:
        raise OutcomeCertificationError(
            f"controller enumeration requires {graph.n_controllers}; limit={controller_limit}"
        )
    if not isinstance(allow_empty, (bool, np.bool_)):
        raise ValueError("allow_empty must be boolean")
    cap = graph.n_controllers if max_size is None else _integer(max_size, "max_size")
    if cap > graph.n_controllers:
        raise ValueError("max_size cannot exceed the controller count")
    contract = _contract(graph, miss_budgets, max_arithmetic_states)
    costs = tuple(Fraction.from_float(float(cost)) for cost in graph.controller_costs)
    minimum_cost = None
    optima = []
    admissible_count = robust_count = 0
    for mask in range(1 << graph.n_controllers):
        if mask.bit_count() > cap or (mask == 0 and not allow_empty):
            continue
        admissible_count += 1
        members = tuple(i for i in range(graph.n_controllers) if mask & (1 << i))
        destruction = _failure_distance(contract, members)
        if destruction.distance is not None and destruction.distance <= reserve:
            continue
        robust_count += 1
        cost = sum((costs[i] for i in members), Fraction())
        if minimum_cost is None or cost < minimum_cost:
            minimum_cost = cost
            optima.clear()
        if cost == minimum_cost:
            optima.append(ReserveQualifiedPortfolio(members, destruction))
    return RowReserveQualification(
        row_reserve=reserve,
        n_tasks=graph.n_tasks,
        max_size=cap,
        allow_empty=bool(allow_empty),
        allowances=contract.allowances,
        status="optimal" if optima else "infeasible",
        minimum_cost=minimum_cost,
        optima=tuple(optima),
        admissible_portfolios=admissible_count,
        robust_feasible_portfolios=robust_count,
    )


def _repair_distance(
    contract: _Contract,
    members: tuple[int, ...],
    max_atom_states: int,
) -> RowChangeDistance:
    limit = _integer(max_atom_states, "max_atom_states", 1)
    graph = contract.graph
    missed = _missed(graph.good, members)
    masses = contract.masses(missed)
    bounds = dict(contract.allowances)
    active = tuple(
        group for group, bound in contract.allowances if masses[group] > bound
    )
    if not active:
        return _distance_with_witness(
            contract, members, (), "repair", "already_feasible"
        )
    if not members:
        return RowChangeDistance(None, None, "infeasible_empty_portfolio")
    missed_rows = {
        group: set(map(int, np.flatnonzero(graph.groups[group] & missed)))
        for group in active
    }
    for governing in active:
        if all(
            missed_rows[group] <= missed_rows[governing]
            and bounds[governing] <= bounds[group]
            for group in active
        ):
            remaining = masses[governing]
            selected = []
            for row in sorted(
                missed_rows[governing], key=lambda row: (-contract.weights[row], row)
            ):
                selected.append(row)
                remaining -= contract.weights[row]
                if remaining <= bounds[governing]:
                    return _distance_with_witness(
                        contract,
                        members,
                        tuple(selected),
                        "repair",
                        "dominating_group_prefix",
                    )
            raise OutcomeCertificationError(
                "dominating-group repair did not attain its bound"
            )
    atoms = tuple(
        atom
        for atom in joint_outcome_atoms(graph)
        if missed[atom.rows[0]] and any(group in atom.groups for group in active)
    )
    states = prod(len(atom.rows) + 1 for atom in atoms)
    if states > limit:
        raise OutcomeCertificationError(
            f"atom repair needs {states} count states; limit={limit}"
        )
    best: tuple[int, ...] | None = None
    for counts in product(*(range(len(atom.rows) + 1) for atom in atoms)):
        if best is not None and sum(counts) >= len(best):
            continue
        if all(
            masses[group]
            - sum(
                (
                    atom.weight * count
                    for atom, count in zip(atoms, counts)
                    if group in atom.groups
                ),
                Fraction(),
            )
            <= bounds[group]
            for group in active
        ):
            best = tuple(
                sorted(
                    row
                    for atom, count in zip(atoms, counts)
                    for row in atom.rows[:count]
                )
            )
    if best is None:
        # A nonempty portfolio can always repair every originally missed row.
        raise OutcomeCertificationError(
            "exhaustive atom search failed to find the full repair"
        )
    return _distance_with_witness(
        contract, members, best, "repair", "exact_atom_counts", states
    )


def portfolio_repair_distance(
    graph: FailureHypergraph,
    members: Sequence[Controller],
    *,
    miss_budgets: Mapping[str, float],
    max_atom_states: int = 100_000,
    max_arithmetic_states: int = 65_536,
) -> RowChangeDistance:
    """Exact weighted overlapping multicover repair; limits fail closed."""

    return _repair_distance(
        _contract(graph, miss_budgets, max_arithmetic_states),
        _members(graph, members),
        max_atom_states,
    )


def _repair_bounds(
    contract: _Contract,
    members: tuple[int, ...],
) -> tuple[int | None, tuple[int, ...] | None, str]:
    """A proven lower bound and feasible upper rows, not an exact repair claim.

    Group prefixes are minimum-cardinality repairs for that group alone. Their
    union is feasible. Counts add as a lower bound only across disjoint FULL
    original missed-row sets; disjoint selected prefixes are insufficient.
    """

    missed = _missed(contract.graph.good, members)
    masses = contract.masses(missed)
    bounds = dict(contract.allowances)
    active = [group for group, bound in contract.allowances if masses[group] > bound]
    if not active:
        return 0, (), "already_feasible"
    if not members:
        return None, None, "infeasible_empty_portfolio"
    entries = []
    for group in active:
        rows = frozenset(
            map(int, np.flatnonzero(contract.graph.groups[group] & missed))
        )
        remaining = masses[group]
        prefix = []
        for row in sorted(rows, key=lambda row: (-contract.weights[row], row)):
            prefix.append(row)
            remaining -= contract.weights[row]
            if remaining <= bounds[group]:
                break
        entries.append((group, rows, tuple(prefix)))
    # Preserve the existing exact dominating-group shortcut and its witness.
    for group, rows, prefix in entries:
        if all(
            other_rows <= rows and bounds[group] <= bounds[other_group]
            for other_group, other_rows, _ in entries
        ):
            return len(prefix), tuple(sorted(prefix)), "dominating_group_prefix"
    used = set()
    packing_bound = 0
    for _, rows, prefix in sorted(
        entries, key=lambda entry: (-len(entry[2]), entry[0])
    ):
        if used.isdisjoint(rows):
            packing_bound += len(prefix)
            used.update(rows)
    lower = max(packing_bound, max(len(prefix) for _, _, prefix in entries))
    upper_rows = tuple(sorted({row for _, _, prefix in entries for row in prefix}))
    return lower, upper_rows, "group_prefix_bounds"


def certify_outcome_stability(
    graph: FailureHypergraph,
    *,
    incumbent: Sequence[Controller],
    miss_budgets: Mapping[str, float],
    max_size: int | None = None,
    allow_empty: bool = True,
    max_controllers: int = 12,
    max_atom_states: int = 100_000,
    max_arithmetic_states: int = 65_536,
) -> OutcomeStabilityCertificate:
    """Certify an allowed exact optimum against any bounded number of row changes.

    Strictly cheaper portfolios are enumerated without cost tolerances. The
    result concerns this incumbent, not persistence of all old optimal masks.
    ``None`` distance is infinity and gives the full finite-universe radius.
    """

    controller_limit = _integer(max_controllers, "max_controllers", 1)
    if controller_limit > 20:
        raise ValueError(
            "max_controllers cannot exceed the explicit small-menu ceiling 20"
        )
    if graph.n_controllers > controller_limit:
        raise OutcomeCertificationError(
            f"controller enumeration requires {graph.n_controllers}; limit={controller_limit}"
        )
    _integer(max_atom_states, "max_atom_states", 1)
    if not isinstance(allow_empty, (bool, np.bool_)):
        raise ValueError("allow_empty must be boolean")
    cap = graph.n_controllers if max_size is None else _integer(max_size, "max_size")
    if cap > graph.n_controllers:
        raise ValueError("max_size cannot exceed the controller count")
    selected = _members(graph, incumbent)
    if not selected and not allow_empty:
        raise ValueError("empty incumbent is forbidden by allow_empty=False")
    if len(selected) > cap:
        raise ValueError("incumbent exceeds max_size")
    contract = _contract(graph, miss_budgets, max_arithmetic_states)
    destruction = _failure_distance(contract, selected)
    if destruction.distance == 0:
        raise ValueError("incumbent must be nominally feasible")
    costs = tuple(Fraction.from_float(float(cost)) for cost in graph.controller_costs)
    selected_cost = sum((costs[i] for i in selected), Fraction())
    best = RowChangeDistance(None, None, "no_finite_cheaper_repair")
    cheaper_count = 0
    candidates = []
    for mask in range(1 << graph.n_controllers):
        if mask.bit_count() > cap or (mask == 0 and not allow_empty):
            continue
        members = tuple(i for i in range(graph.n_controllers) if mask & (1 << i))
        if sum((costs[i] for i in members), Fraction()) >= selected_cost:
            continue
        cheaper_count += 1
        lower, upper_rows, method = _repair_bounds(contract, members)
        if lower == 0:
            raise ValueError(
                f"incumbent is not exactly cost-optimal; cheaper portfolio={members}"
            )
        if lower is None:
            continue  # Only the infeasible empty portfolio has infinite repair.
        candidates.append((lower, len(upper_rows), mask, members))
        if best.distance is None or len(upper_rows) < best.distance:
            best = _distance_with_witness(
                contract, members, upper_rows, "repair", method
            )
    # All cheaper portfolios have now passed the nominal-infeasibility check.
    # The API reports their MINIMUM repair distance, not every individual one.
    # Every retained upper witness is independently checked on original rows.
    for lower, _, _, members in sorted(candidates):
        if lower >= best.distance:
            continue
        candidate = _repair_distance(contract, members, max_atom_states)
        if candidate.distance is not None and (
            best.distance is None or candidate.distance < best.distance
        ):
            best = candidate
    finite = [
        component for component in (destruction, best) if component.distance is not None
    ]
    first = min(finite, key=lambda component: component.distance) if finite else None
    distance = first.distance if first is not None else None
    radius = (
        graph.n_tasks if distance is None else min(graph.n_tasks, max(0, distance - 1))
    )
    return OutcomeStabilityCertificate(
        selected,
        selected_cost,
        graph.n_tasks,
        cap,
        bool(allow_empty),
        contract.allowances,
        destruction,
        best,
        distance,
        radius,
        first.witness if first is not None else None,
        cheaper_count,
    )
