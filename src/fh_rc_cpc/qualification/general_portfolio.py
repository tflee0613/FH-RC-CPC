"""General solvers for group-constrained controller portfolio compression."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


class PortfolioInfeasible(RuntimeError):
    """No controller portfolio satisfies all registered group budgets."""


@dataclass(frozen=True)
class GeneralPortfolioResult:
    """Immutable output shared by exact and deterministic greedy solvers."""

    members: tuple[int, ...]
    objective: float
    status: str
    group_misses: dict[str, int]
    group_weighted_misses: dict[str, float]
    tie_break: str
    optimality_certified: bool = field(default=False, kw_only=True)
    optimality_basis: str = field(default="none", kw_only=True)


def _uses_integer_weight_contract(weights: np.ndarray) -> bool:
    """Only exactly integer-valued represented weights define this contract."""

    values = np.asarray(weights, dtype=float)
    return bool(np.all(values == np.rint(values)))


def _weighted_miss_allowance(
    weights: np.ndarray,
    budget: float,
    *,
    integer_weight_contract: bool | None = None,
) -> float:
    """Translate a rate budget into its exact weighted-miss allowance.

    Integer-weight qualification contracts admit an integer number of misses,
    so ``floor(beta * W)`` is the operative bound.  General noninteger
    importance weights retain the continuous weighted allowance ``beta * W``.
    The optional flag lets a losslessly compressed representation preserve the
    weight semantics of its source tasks even when aggregation hides them.
    """

    values = np.asarray(weights, dtype=float)
    integral = (
        _uses_integer_weight_contract(values)
        if integer_weight_contract is None
        else bool(integer_weight_contract)
    )
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("weights must be a finite nonnegative vector")
    governed_values = np.rint(values) if integral else values
    allowance = float(budget) * float(governed_values.sum())
    return float(np.floor(allowance)) if integral else allowance


def _weighted_miss_is_feasible(
    weighted_miss: float,
    allowance: float,
    *,
    integer_weight_contract: bool,
) -> bool:
    """Apply the exact, non-expanding boundary for a miss contract.

    The governed inputs are IEEE-754 values.  Feasibility therefore compares
    their evaluated values directly: no absolute or ULP tolerance may enlarge
    a registered allowance.  In particular, a zero allowance rejects every
    positive weighted miss.
    """

    observed = float(weighted_miss)
    permitted = float(allowance)
    if (
        not np.isfinite(observed)
        or observed < 0.0
        or not np.isfinite(permitted)
        or permitted < 0.0
    ):
        return False
    return observed <= permitted


def _validated_inputs(
    coverage: np.ndarray,
    costs: np.ndarray,
    groups: np.ndarray | Mapping[str, np.ndarray],
    miss_budgets: Mapping[str, float],
    weights: np.ndarray | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, float],
    np.ndarray,
]:
    coverage_array = np.asarray(coverage, dtype=bool)
    costs_array = np.asarray(costs, dtype=float)
    if coverage_array.ndim != 2 or min(coverage_array.shape) <= 0:
        raise ValueError("coverage must be a nonempty context-by-controller matrix")
    n_contexts, n_controllers = coverage_array.shape
    if costs_array.shape != (n_controllers,):
        raise ValueError("costs must align with coverage columns")
    if not np.isfinite(costs_array).all() or np.any(costs_array <= 0):
        raise ValueError("controller costs must be finite and positive")
    budgets = {str(key): float(value) for key, value in miss_budgets.items()}
    if isinstance(groups, Mapping):
        memberships = {
            str(name): np.asarray(domain, dtype=bool) for name, domain in groups.items()
        }
        if any(domain.shape != (n_contexts,) for domain in memberships.values()):
            raise ValueError(
                "every group membership mask must align with coverage rows"
            )
        if any(not bool(domain.any()) for domain in memberships.values()):
            raise ValueError("group membership masks cannot be empty")
    else:
        groups_array = np.asarray(groups).astype(str)
        if groups_array.shape != (n_contexts,):
            raise ValueError("groups must align with coverage rows")
        memberships = {
            str(group): groups_array == group
            for group in sorted(np.unique(groups_array))
        }
    if set(budgets) != set(memberships):
        raise ValueError("miss_budgets must contain every observed group exactly once")
    if any(
        not np.isfinite(value) or value < 0 or value > 1 for value in budgets.values()
    ):
        raise ValueError("group miss budgets must lie in [0, 1]")
    if weights is None:
        weights_array = np.ones(n_contexts, dtype=float)
    else:
        weights_array = np.asarray(weights, dtype=float)
    if weights_array.shape != (n_contexts,):
        raise ValueError("weights must align with coverage rows")
    if not np.isfinite(weights_array).all() or np.any(weights_array <= 0):
        raise ValueError("context weights must be finite and positive")
    return coverage_array, costs_array, memberships, budgets, weights_array


def _result(
    *,
    members: tuple[int, ...],
    coverage: np.ndarray,
    costs: np.ndarray,
    group_membership: Mapping[str, np.ndarray],
    weights: np.ndarray,
    status: str,
    tie_break: str,
    optimality_certified: bool = False,
    optimality_basis: str = "none",
) -> GeneralPortfolioResult:
    covered = (
        coverage[:, list(members)].any(axis=1)
        if members
        else np.zeros(len(coverage), dtype=bool)
    )
    group_misses: dict[str, int] = {}
    weighted_misses: dict[str, float] = {}
    for group in sorted(group_membership):
        domain = group_membership[group]
        group_misses[str(group)] = int(np.sum(domain & ~covered))
        weighted_misses[str(group)] = float(np.sum(weights[domain & ~covered]))
    return GeneralPortfolioResult(
        members=members,
        objective=float(costs[list(members)].sum()) if members else 0.0,
        status=status,
        group_misses=group_misses,
        group_weighted_misses=weighted_misses,
        tie_break=tie_break,
        optimality_certified=optimality_certified,
        optimality_basis=optimality_basis,
    )


def _exact_zero_or_one_certificate(
    costs: np.ndarray,
    members: tuple[int, ...],
    contracts: list[tuple[np.ndarray, np.ndarray, float]],
    *,
    allow_empty: bool,
) -> bool:
    """A positive-cost lower-bound proof, independent of any numerical dual.

    Empty costs zero. When empty is forbidden, every admitted portfolio costs
    at least the cheapest singleton. When empty is allowed, a singleton can be
    optimal only if empty is infeasible. A feasible singleton at the global
    minimum singleton cost attains the corresponding lower bound. Both the
    governed IEEE predicate and exact represented sums of supplied mass vectors
    against the frozen allowance support the witness. Signature vectors may
    already be aggregated: this is not a certificate for original individual-
    weight rational sums. The claim concerns governed feasibility and exact
    represented costs, not all ties or numerical lexicographic minimality.
    """
    if len(members) > 1:
        return False
    exact_costs = tuple(Fraction.from_float(float(cost)) for cost in costs)
    if members and exact_costs[members[0]] != min(exact_costs):
        return False
    empty_ieee = empty_rational = True
    for coverage, weights, allowance in contracts:
        covered = (
            coverage[:, list(members)].any(axis=1)
            if members
            else np.zeros(len(weights), dtype=bool)
        )
        permitted = Fraction.from_float(float(allowance))
        masses = tuple(Fraction.from_float(float(weight)) for weight in weights)
        miss = sum(
            (weight for weight, hit in zip(masses, covered, strict=True) if not hit),
            Fraction(),
        )
        if miss > permitted or float(weights[~covered].sum()) > allowance:
            return False
        empty_rational = empty_rational and sum(masses, Fraction()) <= permitted
        empty_ieee = empty_ieee and float(weights.sum()) <= allowance
    if not members:
        return allow_empty
    return (not allow_empty) or (not empty_ieee and not empty_rational)


def solve_group_constrained_portfolio(
    coverage: np.ndarray,
    *,
    costs: np.ndarray,
    groups: np.ndarray | Mapping[str, np.ndarray],
    miss_budgets: Mapping[str, float],
    weights: np.ndarray | None = None,
    max_size: int | None = None,
    allow_empty: bool = True,
) -> GeneralPortfolioResult:
    """Solve weighted partial set cover with explicit optimality evidence.

    Controller variables are binary and uncovered-context auxiliaries are
    continuous on [0, 1].  A first MILP minimizes the unperturbed registered
    cost. A second MILP fixes that cost and applies a deterministic
    controller-index tie objective. At most 20 controllers additionally use
    exhaustive original-task verification with exact represented additive-cost
    ordering; this bounded authority is exponential, not a MILP-only path.
    Larger menus report numerical MILP closure, not exact-cost optimality.
    Only a separate empty/minimum-singleton lower-bound proof can certify a
    large-menu result. Individual cost spacing does not bound portfolio-sum
    spacing. Integer source weights impose the exact
    allowance ``floor(beta_g W_g)``; noninteger weights keep ``beta_g W_g``.
    Exact enumeration, where used, is independent of numerical MILP tolerances.
    """

    coverage_array, costs_array, memberships, budgets, weights_array = (
        _validated_inputs(coverage, costs, groups, miss_budgets, weights)
    )
    integer_weight_contract = _uses_integer_weight_contract(weights_array)
    n_contexts, n_controllers = coverage_array.shape
    if not isinstance(allow_empty, (bool, np.bool_)):
        raise ValueError("allow_empty must be boolean")
    if max_size is not None and (
        isinstance(max_size, (bool, np.bool_))
        or int(max_size) != max_size
        or max_size < 0
        or max_size > n_controllers
    ):
        raise ValueError("max_size must lie between zero and the controller count")
    n_variables = n_controllers + n_contexts
    primary_objective = np.concatenate([costs_array, np.zeros(n_contexts, dtype=float)])

    cover_constraints = np.zeros((n_contexts, n_variables), dtype=float)
    cover_constraints[:, :n_controllers] = coverage_array.astype(float)
    cover_constraints[:, n_controllers:] = np.eye(n_contexts, dtype=float)
    constraints: list[LinearConstraint] = [
        LinearConstraint(
            cover_constraints,
            lb=np.ones(n_contexts, dtype=float),
            ub=np.full(n_contexts, np.inf, dtype=float),
        )
    ]
    group_matrix = np.zeros((len(budgets), n_variables), dtype=float)
    group_upper = np.zeros(len(budgets), dtype=float)
    for row, group in enumerate(sorted(budgets)):
        domain = memberships[group]
        group_matrix[row, n_controllers:][domain] = weights_array[domain]
        group_upper[row] = _weighted_miss_allowance(
            weights_array[domain],
            budgets[group],
            integer_weight_contract=integer_weight_contract,
        )
    constraints.append(
        LinearConstraint(
            group_matrix,
            lb=np.full(len(budgets), -np.inf, dtype=float),
            ub=group_upper,
        )
    )
    if max_size is not None:
        size_row = np.zeros((1, n_variables), dtype=float)
        size_row[0, :n_controllers] = 1.0
        constraints.append(
            LinearConstraint(
                size_row,
                lb=np.array([-np.inf], dtype=float),
                ub=np.array([float(max_size)], dtype=float),
            )
        )
    if not allow_empty:
        nonempty_row = np.zeros((1, n_variables), dtype=float)
        nonempty_row[0, :n_controllers] = 1.0
        constraints.append(
            LinearConstraint(
                nonempty_row,
                lb=np.array([1.0], dtype=float),
                ub=np.array([np.inf], dtype=float),
            )
        )
    integrality = np.concatenate(
        [
            np.ones(n_controllers, dtype=np.uint8),
            np.zeros(n_contexts, dtype=np.uint8),
        ]
    )
    primary = milp(
        c=primary_objective,
        integrality=integrality,
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=constraints,
        options={"presolve": True},
    )
    if not bool(primary.success) or primary.x is None:
        raise PortfolioInfeasible(
            f"exact portfolio problem is infeasible or unsolved: {primary.message}"
        )
    primary_members = tuple(
        int(index) for index in np.flatnonzero(primary.x[:n_controllers] >= 0.5)
    )
    optimum = float(costs_array[list(primary_members)].sum())
    gap_value = getattr(primary, "mip_gap", None)
    numerical_closed = (
        int(primary.status) == 0
        and gap_value is not None
        and np.isfinite(float(gap_value))
        and float(gap_value) == 0.0
    )

    cost_row = np.zeros((1, n_variables), dtype=float)
    cost_row[0, :n_controllers] = costs_array
    tie_constraints = [
        *constraints,
        LinearConstraint(
            cost_row,
            lb=np.array([optimum], dtype=float),
            ub=np.array([optimum], dtype=float),
        ),
    ]
    members = primary_members
    if n_controllers <= 20:
        controller_tie = np.exp2(np.arange(n_controllers, dtype=float))
        tie_rule = "two_stage_cost_then_bitmask"
        tie_objective = np.concatenate(
            [controller_tie, np.zeros(n_contexts, dtype=float)]
        )
        tie = milp(
            c=tie_objective,
            integrality=integrality,
            bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
            constraints=tie_constraints,
            options={"presolve": True},
        )
        if bool(tie.success) and tie.x is not None:
            candidate = tuple(
                int(index) for index in np.flatnonzero(tie.x[:n_controllers] >= 0.5)
            )
            candidate_cost = float(costs_array[list(candidate)].sum())
            if candidate_cost == optimum:
                members = candidate
            else:
                tie_rule = "primary_optimum_after_tie_cost_mismatch"
        else:
            tie_rule = "primary_optimum_after_tie_failure"
    elif numerical_closed:
        locked_constraints = list(tie_constraints)
        locked_solution = None
        tie_rule = "two_stage_cost_then_numerical_binary_lexicographic_locking"
        for controller in reversed(range(n_controllers)):
            lex_objective = np.zeros(n_variables, dtype=float)
            lex_objective[controller] = 1.0
            lex = milp(
                c=lex_objective,
                integrality=integrality,
                bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
                constraints=locked_constraints,
                options={"presolve": True},
            )
            if not bool(lex.success) or lex.x is None:
                locked_solution = None
                tie_rule = "numerical_primary_after_lexicographic_tie_failure"
                break
            locked_value = float(lex.x[controller] >= 0.5)
            lock_row = np.zeros((1, n_variables), dtype=float)
            lock_row[0, controller] = 1.0
            locked_constraints.append(
                LinearConstraint(
                    lock_row,
                    lb=np.array([locked_value], dtype=float),
                    ub=np.array([locked_value], dtype=float),
                )
            )
            locked_solution = lex.x
        if locked_solution is not None:
            candidate = tuple(
                int(index)
                for index in np.flatnonzero(locked_solution[:n_controllers] >= 0.5)
            )
            candidate_cost = float(costs_array[list(candidate)].sum())
            if candidate_cost == optimum:
                members = candidate
            else:
                tie_rule = "numerical_primary_after_tie_cost_mismatch"
    else:
        tie_rule = "feasible_incumbent_without_secondary_tie_break"
    if n_controllers <= 20:
        # Use original row sums, not a compressed representation's reduction
        # order. This authority also catches primary MILP cost tolerances.
        exact_costs = tuple(Fraction.from_float(float(cost)) for cost in costs_array)
        allowances = dict(zip(sorted(budgets), group_upper, strict=True))
        best_key = None
        best_members = ()
        for mask in range(0 if allow_empty else 1, 1 << n_controllers):
            if max_size is not None and mask.bit_count() > max_size:
                continue
            candidate = tuple(i for i in range(n_controllers) if mask & (1 << i))
            covered = (
                coverage_array[:, list(candidate)].any(axis=1)
                if candidate
                else np.zeros(n_contexts, dtype=bool)
            )
            if not all(
                _weighted_miss_is_feasible(
                    float(weights_array[domain & ~covered].sum()),
                    allowances[group],
                    integer_weight_contract=integer_weight_contract,
                )
                for group, domain in memberships.items()
            ):
                continue
            key = (sum((exact_costs[i] for i in candidate), Fraction()), mask)
            if best_key is None or key < best_key:
                best_key, best_members = key, candidate
        if best_key is None:
            raise PortfolioInfeasible(
                "no enumerated task portfolio satisfies the budgets"
            )
        if members != best_members:
            tie_rule = "complete_enumeration_authority_after_milp_crosscheck"
        members = best_members
        optimum = float(costs_array[list(members)].sum()) if members else 0.0
    exact_trivial = n_controllers > 20 and _exact_zero_or_one_certificate(
        costs_array,
        members,
        [
            (
                coverage_array[memberships[group]],
                weights_array[memberships[group]],
                float(group_upper[index]),
            )
            for index, group in enumerate(sorted(budgets))
        ],
        allow_empty=bool(allow_empty),
    )
    certified = n_controllers <= 20 or exact_trivial
    basis = (
        "bounded_exhaustive"
        if n_controllers <= 20
        else "exact_empty_or_minimum_singleton"
        if exact_trivial
        else "numerical_milp"
    )
    result = _result(
        members=members,
        coverage=coverage_array,
        costs=costs_array,
        group_membership=memberships,
        weights=weights_array,
        status="optimal"
        if certified
        else "numerical_optimal"
        if numerical_closed
        else "feasible_uncertified",
        tie_break=tie_rule,
        optimality_certified=certified,
        optimality_basis=basis,
    )
    if result.objective != optimum:
        raise RuntimeError("tie-breaking stage changed the registered-cost optimum")
    for group, budget in budgets.items():
        domain = memberships[group]
        permitted = _weighted_miss_allowance(
            weights_array[domain],
            budget,
            integer_weight_contract=integer_weight_contract,
        )
        if not _weighted_miss_is_feasible(
            result.group_weighted_misses[group],
            permitted,
            integer_weight_contract=integer_weight_contract,
        ):
            raise RuntimeError("solver output violates a registered group budget")
    return result


def greedy_group_constrained_portfolio(
    coverage: np.ndarray,
    *,
    costs: np.ndarray,
    groups: np.ndarray | Mapping[str, np.ndarray],
    miss_budgets: Mapping[str, float],
    weights: np.ndarray | None = None,
    max_size: int | None = None,
    allow_empty: bool = True,
) -> GeneralPortfolioResult:
    """Deterministically reduce weighted group deficits per controller cost.

    Integer source weights use ``floor(beta_g W_g)`` permitted misses in both
    marginal gains and termination.  Noninteger importance weights retain the
    continuous ``beta_g W_g`` allowance.
    """

    coverage_array, costs_array, memberships, budgets, weights_array = (
        _validated_inputs(coverage, costs, groups, miss_budgets, weights)
    )
    integer_weight_contract = _uses_integer_weight_contract(weights_array)
    n_contexts, n_controllers = coverage_array.shape
    if not isinstance(allow_empty, (bool, np.bool_)):
        raise ValueError("allow_empty must be boolean")
    if max_size is not None and (
        isinstance(max_size, (bool, np.bool_))
        or int(max_size) != max_size
        or max_size < 0
        or max_size > n_controllers
    ):
        raise ValueError("max_size must lie between zero and the controller count")
    selected: list[int] = []
    covered = np.zeros(n_contexts, dtype=bool)

    def deficits() -> dict[str, float]:
        values: dict[str, float] = {}
        for group, budget in budgets.items():
            domain = memberships[group]
            uncovered_weight = float(weights_array[domain & ~covered].sum())
            permitted = _weighted_miss_allowance(
                weights_array[domain],
                budget,
                integer_weight_contract=integer_weight_contract,
            )
            values[group] = (
                0.0
                if _weighted_miss_is_feasible(
                    uncovered_weight,
                    permitted,
                    integer_weight_contract=integer_weight_contract,
                )
                else uncovered_weight - permitted
            )
        return values

    while any(value > 0.0 for value in deficits().values()):
        if max_size is not None and len(selected) >= max_size:
            raise PortfolioInfeasible(
                "greedy solver reached max_size before satisfying group budgets"
            )
        before = deficits()
        choices: list[tuple[float, float, int, np.ndarray]] = []
        for controller in range(n_controllers):
            if controller in selected:
                continue
            candidate_covered = covered | coverage_array[:, controller]
            reduction = 0.0
            for group, deficit in before.items():
                if deficit <= 0:
                    continue
                domain = memberships[group]
                newly_covered = domain & ~covered & candidate_covered
                reduction += min(deficit, float(weights_array[newly_covered].sum()))
            score = reduction / float(costs_array[controller])
            choices.append(
                (score, -float(costs_array[controller]), -controller, candidate_covered)
            )
        if not choices:
            raise PortfolioInfeasible("greedy solver exhausted the controller library")
        best = max(choices, key=lambda item: item[:3])
        if best[0] <= 0:
            raise PortfolioInfeasible("greedy solver cannot reduce remaining deficits")
        controller = -int(best[2])
        selected.append(controller)
        covered = best[3]
    if not selected and not allow_empty:
        if max_size == 0:
            raise PortfolioInfeasible(
                "nonempty policy conflicts with a zero portfolio-size cap"
            )
        controller = min(
            range(n_controllers),
            key=lambda index: (float(costs_array[index]), index),
        )
        selected.append(controller)
        covered = coverage_array[:, controller].copy()
    members = tuple(sorted(selected))
    return _result(
        members=members,
        coverage=coverage_array,
        costs=costs_array,
        group_membership=memberships,
        weights=weights_array,
        status="feasible_greedy",
        tie_break="gain_per_cost_then_cost_then_index",
    )
