"""Exact and deterministic greedy solvers on lossless failure signatures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, eye, hstack

from .general_portfolio import (
    PortfolioInfeasible,
    _exact_zero_or_one_certificate,
    _weighted_miss_allowance,
    _weighted_miss_is_feasible,
)
from .signature_compression import (
    FailureSignatureTable,
    PortfolioEvaluation,
    evaluate_signature_portfolio,
)


@dataclass(frozen=True)
class SignaturePortfolioResult:
    members: tuple[int, ...]
    objective: float
    status: str
    group_misses: dict[str, int]
    group_weighted_misses: dict[str, float]
    total_miss_count: int
    total_weighted_miss: float
    tie_break: str
    optimality_certified: bool
    mip_gap: float | None
    solver_message: str
    mip_node_count: int | None = None
    mip_dual_bound: float | None = None
    lp_relaxation_objective: float | None = None
    lp_relative_gap: float | None = None
    solver_status_code: int | None = None
    presolve: bool | None = None
    optimality_basis: str = field(default="none", kw_only=True)


@dataclass(frozen=True)
class LPRoundingResult:
    """Deterministic threshold rounding followed by ranked feasibility repair."""

    members: tuple[int, ...]
    initial_members: tuple[int, ...]
    repair_added: tuple[int, ...]
    objective: float
    lp_objective: float
    lp_controller_values: tuple[float, ...]
    threshold: float
    constraint_feasible: bool
    group_misses: dict[str, int]
    group_weighted_misses: dict[str, float]
    total_miss_count: int
    status: str


def _budgets(
    table: FailureSignatureTable, miss_budgets: Mapping[str, float]
) -> dict[str, float]:
    budgets = {str(group): float(value) for group, value in miss_budgets.items()}
    if set(budgets) != set(table.group_weights):
        raise ValueError("miss_budgets must contain every group exactly once")
    if any(
        not np.isfinite(value) or value < 0 or value > 1 for value in budgets.values()
    ):
        raise ValueError("group miss budgets must lie in [0, 1]")
    return budgets


def _constraints(
    table: FailureSignatureTable,
    budgets: Mapping[str, float],
    *,
    max_size: int | None,
    allow_empty: bool,
) -> list[LinearConstraint]:
    k = table.n_controllers
    s = table.n_signatures
    cover = hstack(
        (csr_matrix(table.signatures.astype(float)), eye(s, dtype=float, format="csr")),
        format="csr",
    )
    constraints = [LinearConstraint(cover, np.ones(s), np.full(s, np.inf, dtype=float))]
    group_matrix = np.zeros((len(budgets), k + s), dtype=float)
    group_upper = np.zeros(len(budgets), dtype=float)
    for row, group in enumerate(sorted(budgets)):
        weights = table.group_weights[group]
        group_matrix[row, k:] = weights
        group_upper[row] = _weighted_miss_allowance(
            weights,
            budgets[group],
            integer_weight_contract=table.integer_weight_contract,
        )
    constraints.append(
        LinearConstraint(
            csr_matrix(group_matrix),
            np.full(len(budgets), -np.inf, dtype=float),
            group_upper,
        )
    )
    if not isinstance(allow_empty, (bool, np.bool_)):
        raise ValueError("allow_empty must be boolean")
    if max_size is not None:
        if (
            isinstance(max_size, (bool, np.bool_))
            or int(max_size) != max_size
            or max_size < 0
            or max_size > k
        ):
            raise ValueError("max_size must lie between zero and controller count")
        size = np.zeros((1, k + s), dtype=float)
        size[0, :k] = 1.0
        constraints.append(
            LinearConstraint(
                csr_matrix(size), np.array([-np.inf]), np.array([float(max_size)])
            )
        )
    if not allow_empty:
        nonempty = np.zeros((1, k + s), dtype=float)
        nonempty[0, :k] = 1.0
        constraints.append(
            LinearConstraint(csr_matrix(nonempty), np.array([1.0]), np.array([np.inf]))
        )
    return constraints


def _result(
    evaluation: PortfolioEvaluation,
    *,
    status: str,
    tie_break: str,
    optimality_certified: bool,
    mip_gap: float | None,
    solver_message: str,
    mip_node_count: int | None = None,
    mip_dual_bound: float | None = None,
    lp_relaxation_objective: float | None = None,
    lp_relative_gap: float | None = None,
    solver_status_code: int | None = None,
    presolve: bool | None = None,
    optimality_basis: str = "none",
) -> SignaturePortfolioResult:
    return SignaturePortfolioResult(
        members=evaluation.members,
        objective=evaluation.cost,
        status=status,
        group_misses=evaluation.group_misses,
        group_weighted_misses=evaluation.group_weighted_misses,
        total_miss_count=evaluation.total_miss_count,
        total_weighted_miss=evaluation.total_weighted_miss,
        tie_break=tie_break,
        optimality_certified=optimality_certified,
        mip_gap=mip_gap,
        solver_message=solver_message,
        mip_node_count=mip_node_count,
        mip_dual_bound=mip_dual_bound,
        lp_relaxation_objective=lp_relaxation_objective,
        lp_relative_gap=lp_relative_gap,
        solver_status_code=solver_status_code,
        presolve=presolve,
        optimality_basis=optimality_basis,
    )


def _assert_feasible(
    table: FailureSignatureTable,
    evaluation: PortfolioEvaluation,
    budgets: Mapping[str, float],
) -> None:
    for group, budget in budgets.items():
        permitted = _weighted_miss_allowance(
            table.group_weights[group],
            budget,
            integer_weight_contract=table.integer_weight_contract,
        )
        if not _weighted_miss_is_feasible(
            evaluation.group_weighted_misses[group],
            permitted,
            integer_weight_contract=table.integer_weight_contract,
        ):
            raise RuntimeError("solver output violates a registered group budget")


def solve_signature_portfolio(
    table: FailureSignatureTable,
    *,
    miss_budgets: Mapping[str, float],
    max_size: int | None = None,
    allow_empty: bool = True,
    presolve: bool = True,
    time_limit: float | None = None,
    mip_rel_gap: float = 0.0,
    compute_lp_relaxation: bool = False,
    deterministic_tie_break: bool = True,
) -> SignaturePortfolioResult:
    """Solve the signature-sufficient problem with explicit optimality evidence.

    Successful menus of at most 20 controllers always use complete enumeration
    as authority, including when ``deterministic_tie_break=False`` skips the
    secondary MILP. Above 20, a finite zero primary gap gives only
    ``numerical_optimal``. Exact certification requires a separate feasible
    empty/minimum-singleton lower-bound proof. Other successful feasible
    outputs have status ``feasible_uncertified``. Numerical index locking is
    a reproducibility refinement, not an exact all-optima or minimum-mask proof.
    Integer source weights impose the exact allowance
    ``floor(beta_g W_g)``; noninteger weights keep ``beta_g W_g``.
    Individual coefficient separation does not ensure separation of portfolio
    cost sums; no numerical dual or zero gap alone is an exact certificate.
    """

    budgets = _budgets(table, miss_budgets)
    relative_gap = float(mip_rel_gap)
    if not np.isfinite(relative_gap) or relative_gap < 0.0:
        raise ValueError("mip_rel_gap must be finite and nonnegative")
    constraints = _constraints(
        table, budgets, max_size=max_size, allow_empty=allow_empty
    )
    k = table.n_controllers
    s = table.n_signatures
    bounds = Bounds(np.zeros(k + s), np.ones(k + s))
    # Controller choices are binary.  Uncovered-signature auxiliaries can be
    # continuous: an uncovered signature is forced to one, while a covered
    # signature can be set to zero without changing the projected portfolio.
    integrality = np.concatenate(
        [np.ones(k, dtype=np.uint8), np.zeros(s, dtype=np.uint8)]
    )
    primary_objective = np.concatenate(
        [table.controller_costs, np.zeros(s, dtype=float)]
    )
    options: dict[str, float | bool] = {
        "presolve": bool(presolve),
        "mip_rel_gap": relative_gap,
    }
    if time_limit is not None:
        if not np.isfinite(time_limit) or float(time_limit) <= 0:
            raise ValueError("time_limit must be finite and positive")
        options["time_limit"] = float(time_limit)
    primary = milp(
        c=primary_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options=options,
    )
    if not bool(primary.success) or primary.x is None or primary.fun is None:
        raise PortfolioInfeasible(
            f"signature portfolio is infeasible or unsolved: {primary.message}"
        )

    primary_members = tuple(
        int(index) for index in np.flatnonzero(primary.x[:k] >= 0.5)
    )
    primary_evaluation = evaluate_signature_portfolio(table, primary_members)
    gap_value = getattr(primary, "mip_gap", None)
    gap = None if gap_value is None else float(gap_value)
    numerical_closed = (
        int(primary.status) == 0 and gap is not None and np.isfinite(gap) and gap == 0.0
    )
    optimum = primary_evaluation.cost
    evaluation = primary_evaluation
    tie_rule = (
        "primary_optimum_without_secondary_tie_break"
        if k <= 20
        else "numerical_primary_without_secondary_tie_break"
        if numerical_closed
        else "feasible_incumbent_without_secondary_tie_break"
    )
    if deterministic_tie_break and (numerical_closed or k <= 20):
        cost_row = np.zeros((1, k + s), dtype=float)
        cost_row[0, :k] = table.controller_costs
        tie_constraints = [
            *constraints,
            LinearConstraint(
                csr_matrix(cost_row),
                np.array([optimum]),
                np.array([optimum]),
            ),
        ]
        if k <= 20:
            controller_tie = np.exp2(np.arange(k, dtype=float))
            tie_rule = "two_stage_cost_then_bitmask"
            tie_objective = np.concatenate(
                [controller_tie, np.full(s, 1.0 / (s + 1), dtype=float)]
            )
            tie = milp(
                c=tie_objective,
                integrality=integrality,
                bounds=bounds,
                constraints=tie_constraints,
                options={**options, "mip_rel_gap": 0.0},
            )
            if bool(tie.success) and tie.x is not None:
                tie_members = tuple(
                    int(index) for index in np.flatnonzero(tie.x[:k] >= 0.5)
                )
                tie_evaluation = evaluate_signature_portfolio(table, tie_members)
                if tie_evaluation.cost == optimum:
                    evaluation = tie_evaluation
        else:
            # An index sum can collide, for example on {0, 3} and {1, 2}.
            # Sequentially lock x_(K-1), ..., x_0 to its smallest feasible
            # value. This is a numerical refinement; solver tolerances can
            # still prevent an exact minimum-mask guarantee.
            locked_constraints = list(tie_constraints)
            locked_solution = None
            tie_rule = "two_stage_cost_then_numerical_binary_lexicographic_locking"
            for controller in reversed(range(k)):
                lex_objective = np.zeros(k + s, dtype=float)
                lex_objective[controller] = 1.0
                lex = milp(
                    c=lex_objective,
                    integrality=integrality,
                    bounds=bounds,
                    constraints=locked_constraints,
                    options={**options, "mip_rel_gap": 0.0},
                )
                if not bool(lex.success) or lex.x is None:
                    locked_solution = None
                    tie_rule = "numerical_primary_after_lexicographic_tie_failure"
                    break
                locked_value = float(lex.x[controller] >= 0.5)
                lock_row = np.zeros((1, k + s), dtype=float)
                lock_row[0, controller] = 1.0
                locked_constraints.append(
                    LinearConstraint(
                        csr_matrix(lock_row),
                        np.array([locked_value]),
                        np.array([locked_value]),
                    )
                )
                locked_solution = lex.x
            if locked_solution is not None:
                tie_members = tuple(
                    int(index) for index in np.flatnonzero(locked_solution[:k] >= 0.5)
                )
                tie_evaluation = evaluate_signature_portfolio(table, tie_members)
                if tie_evaluation.cost == optimum:
                    evaluation = tie_evaluation
    enumeration_certified = False
    if k <= 20:
        exhaustive = solve_signature_portfolio_by_enumeration(
            table,
            miss_budgets=budgets,
            max_size=max_size,
            allow_empty=allow_empty,
        )
        enumeration_certified = True
        if evaluation.members != exhaustive.members:
            evaluation = evaluate_signature_portfolio(table, exhaustive.members)
            tie_rule = "complete_enumeration_authority_after_milp_crosscheck"
        optimum = exhaustive.objective
    _assert_feasible(table, evaluation, budgets)
    if evaluation.cost != optimum:
        raise RuntimeError("tie-breaking stage changed the registered-cost optimum")
    node_value = getattr(primary, "mip_node_count", None)
    node_count = None if node_value is None else int(node_value)
    dual_value = getattr(primary, "mip_dual_bound", None)
    dual_bound = None if dual_value is None else float(dual_value)
    lp_objective: float | None = None
    lp_relative_gap: float | None = None
    if compute_lp_relaxation:
        lp = milp(
            c=primary_objective,
            integrality=np.zeros(k + s, dtype=np.uint8),
            bounds=bounds,
            constraints=constraints,
            options={
                key: value for key, value in options.items() if key != "mip_rel_gap"
            },
        )
        if bool(lp.success) and lp.fun is not None:
            lp_objective = float(lp.fun)
            denominator = max(abs(float(optimum)), 1e-12)
            lp_relative_gap = max(0.0, (float(optimum) - lp_objective) / denominator)
    exact_trivial = k > 20 and _exact_zero_or_one_certificate(
        table.controller_costs,
        evaluation.members,
        [
            (
                table.signatures,
                table.group_weights[group],
                _weighted_miss_allowance(
                    table.group_weights[group],
                    budget,
                    integer_weight_contract=table.integer_weight_contract,
                ),
            )
            for group, budget in budgets.items()
        ],
        allow_empty=bool(allow_empty),
    )
    optimality_certified = bool(enumeration_certified or exact_trivial)
    basis = (
        "bounded_exhaustive"
        if enumeration_certified
        else "exact_empty_or_minimum_singleton"
        if exact_trivial
        else "numerical_milp"
    )
    return _result(
        evaluation,
        status="optimal"
        if optimality_certified
        else "numerical_optimal"
        if numerical_closed
        else "feasible_uncertified",
        tie_break=tie_rule,
        optimality_certified=optimality_certified,
        mip_gap=gap,
        solver_message=str(primary.message),
        mip_node_count=node_count,
        mip_dual_bound=dual_bound,
        lp_relaxation_objective=lp_objective,
        lp_relative_gap=lp_relative_gap,
        solver_status_code=int(primary.status),
        presolve=bool(presolve),
        optimality_basis=basis,
    )


def solve_signature_lp_rounding_baseline(
    table: FailureSignatureTable,
    *,
    miss_budgets: Mapping[str, float],
    max_size: int | None = None,
    allow_empty: bool = True,
    threshold: float = 0.5,
    presolve: bool = True,
    time_limit: float | None = None,
) -> LPRoundingResult:
    """Round the controller coordinates of the LP and repair deterministically.

    Controller coordinates at least ``threshold`` are retained first, ordered
    by decreasing LP value, lower registered cost, and controller index.  If
    the rounded set violates a group budget, remaining controllers are added in
    the same order until the contract is feasible or ``max_size`` is reached.
    The method is a transparent comparator, not an approximation guarantee.
    """

    budgets = _budgets(table, miss_budgets)
    k = table.n_controllers
    s = table.n_signatures
    limit = k if max_size is None else int(max_size)
    if limit < 0 or limit > k:
        raise ValueError("max_size must lie between zero and controller count")
    cutoff = float(threshold)
    if not np.isfinite(cutoff) or not 0.0 < cutoff <= 1.0:
        raise ValueError("rounding threshold must lie in (0, 1]")
    constraints = _constraints(
        table, budgets, max_size=max_size, allow_empty=allow_empty
    )
    objective = np.concatenate([table.controller_costs, np.zeros(s, dtype=float)])
    options: dict[str, float | bool] = {"presolve": bool(presolve)}
    if time_limit is not None:
        if not np.isfinite(time_limit) or float(time_limit) <= 0:
            raise ValueError("time_limit must be finite and positive")
        options["time_limit"] = float(time_limit)
    lp = milp(
        c=objective,
        integrality=np.zeros(k + s, dtype=np.uint8),
        bounds=Bounds(np.zeros(k + s), np.ones(k + s)),
        constraints=constraints,
        options=options,
    )
    if not bool(lp.success) or lp.x is None or lp.fun is None:
        raise PortfolioInfeasible(
            f"signature LP is infeasible or unsolved: {lp.message}"
        )
    values = np.asarray(lp.x[:k], dtype=float)
    order = tuple(
        sorted(
            range(k),
            key=lambda index: (
                -float(values[index]),
                float(table.controller_costs[index]),
                index,
            ),
        )
    )
    selected = [index for index in order if values[index] >= cutoff - 1e-10][:limit]
    initial = tuple(sorted(selected))

    def feasible(members: tuple[int, ...]) -> bool:
        if not allow_empty and not members:
            return False
        evaluation = evaluate_signature_portfolio(table, members)
        return all(
            _weighted_miss_is_feasible(
                evaluation.group_weighted_misses[group],
                _weighted_miss_allowance(
                    table.group_weights[group],
                    budget,
                    integer_weight_contract=table.integer_weight_contract,
                ),
                integer_weight_contract=table.integer_weight_contract,
            )
            for group, budget in budgets.items()
        )

    repair: list[int] = []
    while len(selected) < limit and not feasible(tuple(sorted(selected))):
        controller = next(index for index in order if index not in selected)
        selected.append(controller)
        repair.append(controller)
    members = tuple(sorted(selected))
    evaluation = evaluate_signature_portfolio(table, members)
    is_feasible = feasible(members)
    return LPRoundingResult(
        members=members,
        initial_members=initial,
        repair_added=tuple(repair),
        objective=evaluation.cost,
        lp_objective=float(lp.fun),
        lp_controller_values=tuple(float(value) for value in values),
        threshold=cutoff,
        constraint_feasible=is_feasible,
        group_misses=evaluation.group_misses,
        group_weighted_misses=evaluation.group_weighted_misses,
        total_miss_count=evaluation.total_miss_count,
        status="feasible_lp_round_and_repair"
        if is_feasible
        else "infeasible_after_cap",
    )


def solve_signature_portfolio_by_enumeration(
    table: FailureSignatureTable,
    *,
    miss_budgets: Mapping[str, float],
    max_size: int | None = None,
    allow_empty: bool = True,
) -> SignaturePortfolioResult:
    """Complete deterministic enumeration for verification when K is small."""

    budgets = _budgets(table, miss_budgets)
    if table.n_controllers > 20:
        raise ValueError("complete enumeration is limited to at most 20 controllers")
    if not isinstance(allow_empty, (bool, np.bool_)):
        raise ValueError("allow_empty must be boolean")
    if max_size is not None and (
        isinstance(max_size, (bool, np.bool_))
        or int(max_size) != max_size
        or max_size < 0
        or max_size > table.n_controllers
    ):
        raise ValueError("max_size must lie between zero and controller count")
    limit = table.n_controllers if max_size is None else int(max_size)
    feasible: list[PortfolioEvaluation] = []
    for size in range(0 if allow_empty else 1, limit + 1):
        for members in combinations(range(table.n_controllers), size):
            evaluation = evaluate_signature_portfolio(table, members)
            if all(
                _weighted_miss_is_feasible(
                    evaluation.group_weighted_misses[group],
                    _weighted_miss_allowance(
                        table.group_weights[group],
                        budget,
                        integer_weight_contract=table.integer_weight_contract,
                    ),
                    integer_weight_contract=table.integer_weight_contract,
                )
                for group, budget in budgets.items()
            ):
                feasible.append(evaluation)
    if not feasible:
        raise PortfolioInfeasible("no enumerated portfolio satisfies all group budgets")
    best = min(
        feasible,
        key=lambda item: (
            sum(
                (
                    Fraction.from_float(float(table.controller_costs[i]))
                    for i in item.members
                ),
                Fraction(),
            ),
            sum(1 << index for index in item.members),
        ),
    )
    return _result(
        best,
        status="optimal_enumeration",
        tie_break="cost_then_bitmask",
        optimality_certified=True,
        mip_gap=0.0,
        solver_message="complete_enumeration",
        optimality_basis="bounded_exhaustive",
    )


def greedy_signature_portfolio(
    table: FailureSignatureTable,
    *,
    miss_budgets: Mapping[str, float],
    max_size: int | None = None,
    allow_empty: bool = True,
) -> SignaturePortfolioResult:
    """Deterministically reduce contract deficits per unit portable cost.

    Integer source weights use ``floor(beta_g W_g)`` permitted misses in both
    marginal gains and termination.  Noninteger importance weights retain the
    continuous ``beta_g W_g`` allowance.
    """

    budgets = _budgets(table, miss_budgets)
    if not isinstance(allow_empty, (bool, np.bool_)):
        raise ValueError("allow_empty must be boolean")
    if max_size is not None and (
        isinstance(max_size, (bool, np.bool_))
        or int(max_size) != max_size
        or max_size < 0
        or max_size > table.n_controllers
    ):
        raise ValueError("max_size must lie between zero and the controller count")
    covered = np.zeros(table.n_signatures, dtype=bool)
    selected: list[int] = []

    def deficits(candidate_covered: np.ndarray) -> dict[str, float]:
        values: dict[str, float] = {}
        for group, budget in budgets.items():
            weights = table.group_weights[group]
            weighted_miss = float(weights[~candidate_covered].sum())
            allowance = _weighted_miss_allowance(
                weights,
                budget,
                integer_weight_contract=table.integer_weight_contract,
            )
            values[group] = (
                0.0
                if _weighted_miss_is_feasible(
                    weighted_miss,
                    allowance,
                    integer_weight_contract=table.integer_weight_contract,
                )
                else weighted_miss - allowance
            )
        return values

    while any(value > 0.0 for value in deficits(covered).values()):
        if max_size is not None and len(selected) >= max_size:
            raise PortfolioInfeasible(
                "greedy solver reached max_size before satisfying group budgets"
            )
        before = deficits(covered)
        choices: list[tuple[float, float, int, np.ndarray]] = []
        for controller in range(table.n_controllers):
            if controller in selected:
                continue
            candidate = covered | table.signatures[:, controller]
            reduction = 0.0
            for group, deficit in before.items():
                if deficit <= 0:
                    continue
                weights = table.group_weights[group]
                newly_covered = ~covered & candidate
                reduction += min(deficit, float(weights[newly_covered].sum()))
            choices.append(
                (
                    reduction / float(table.controller_costs[controller]),
                    -float(table.controller_costs[controller]),
                    -controller,
                    candidate,
                )
            )
        if not choices:
            raise PortfolioInfeasible("greedy solver exhausted the controller library")
        best = max(choices, key=lambda item: item[:3])
        if best[0] <= 0:
            raise PortfolioInfeasible("greedy solver cannot reduce remaining deficits")
        selected.append(-int(best[2]))
        covered = best[3]

    if not selected and not allow_empty:
        if max_size == 0:
            raise PortfolioInfeasible(
                "nonempty policy conflicts with a zero portfolio-size cap"
            )
        controller = min(
            range(table.n_controllers),
            key=lambda index: (float(table.controller_costs[index]), index),
        )
        selected.append(controller)
        covered = table.signatures[:, controller].copy()

    evaluation = evaluate_signature_portfolio(table, tuple(sorted(selected)))
    _assert_feasible(table, evaluation, budgets)
    return _result(
        evaluation,
        status="feasible_greedy",
        tie_break="deficit_rescue_per_cost_then_cost_then_index",
        optimality_certified=False,
        mip_gap=None,
        solver_message="deterministic_greedy",
    )
