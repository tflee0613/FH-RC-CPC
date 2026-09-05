"""Finite algorithmic certificates for FH--RC--CPC.

This module turns four structural facts into executable checks: subset-zeta
frontier construction, exact projection of continuous miss auxiliaries,
the harmonic guarantee for integer-valued submodular cover, and robustness of
an optimum structure over a continuous box of additive controller costs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from operator import index

import numpy as np
import pandas as pd

from ..qualification.failure_hypergraph import Controller
from ..qualification.general_portfolio import (
    PortfolioInfeasible,
    _weighted_miss_allowance,
    _weighted_miss_is_feasible,
)
from ..qualification.signature_compression import (
    FailureSignatureTable,
    evaluate_signature_portfolio,
)
from ..qualification.signature_solver import (
    greedy_signature_portfolio,
    solve_signature_portfolio_by_enumeration,
)
from .exact_linear_program import ExactLPCertificationError, solve_exact_lp
from .portfolio_certificates import (
    _complete_frontier_controller_count,
    _group_mass_columns,
    _validated_frontier,
)


@dataclass(frozen=True)
class ContinuousAuxiliaryProjectionCertificate:
    """Exact fixed-portfolio projection of the continuous miss auxiliaries."""

    members: tuple[int, ...]
    minimum_auxiliaries: tuple[float, ...]
    group_weighted_misses: dict[str, float]
    group_allowances: dict[str, float]
    group_feasible: dict[str, bool]
    projected_feasible: bool
    boolean_feasible: bool
    projection_exact: bool


@dataclass(frozen=True)
class GreedySubmodularCoverCertificate:
    """Auditable conditions and factor for integer-valued submodular cover."""

    members: tuple[int, ...]
    greedy_cost: float
    target_value: int
    achieved_value: int
    maximum_singleton_gain: int
    harmonic_factor: float
    logarithmic_factor: float
    conditions_satisfied: bool
    exact_optimum_cost: float | None
    observed_ratio: float | None
    exhaustive_bound_verified: bool | None


@dataclass(frozen=True)
class CostBoxStructureCertificate:
    """LP certificate for optimum-structure invariance over a cost box."""

    target_masks: tuple[int, ...]
    feasible_masks: tuple[int, ...]
    inclusion_minimal_feasible_masks: tuple[int, ...]
    dominated_feasible_mask_count: int
    cost_lower: tuple[float, ...]
    cost_upper: tuple[float, ...]
    cost_atol: float
    all_optima_in_target_over_full_box: bool
    target_optimum_exists_over_full_box: bool
    checked_non_target_masks: int
    witness_mask: int | None
    witness_costs: tuple[float, ...] | None
    witness_optimal_masks: tuple[int, ...]
    witness_kind: str | None
    max_size: int | None = None
    witness_costs_exact: tuple[str, ...] | None = field(default=None, kw_only=True)
    verification_method: str = field(default="rational_primal_dual", kw_only=True)

@dataclass(frozen=True)
class SymmetricCostRobustnessCertificate:
    """Exact first-breakpoint certificate for independent relative costs.

    ``robustness_radius_exact`` records the supremum of radii for which every
    optimum in the relative box remains in ``target_masks``;
    ``robustness_radius`` is its downward-rounded float display. A non-target optimum at the
    breakpoint is returned as a cost-vector witness whenever one exists in the
    declared radius domain.
    """

    target_masks: tuple[int, ...]
    feasible_masks: tuple[int, ...]
    inclusion_minimal_feasible_masks: tuple[int, ...]
    nominal_costs: tuple[float, ...]
    radius_domain_upper: float
    cost_atol: float
    robustness_radius: float
    nominal_all_optima_in_target: bool
    radius_is_exact: bool
    radius_is_domain_limit: bool
    safe_for_every_strictly_smaller_radius: bool
    failure_at_radius: bool
    checked_non_target_minimal_masks: int
    witness_mask: int | None
    witness_costs: tuple[float, ...] | None
    witness_optimal_masks: tuple[int, ...]
    witness_kind: str | None
    max_size: int | None = None
    robustness_radius_exact: str | None = field(default=None, kw_only=True)
    witness_costs_exact: tuple[str, ...] | None = field(default=None, kw_only=True)
    verification_method: str = field(default="rational_primal_dual", kw_only=True)


def _validated_budgets(
    table: FailureSignatureTable,
    miss_budgets: Mapping[str, float],
) -> dict[str, float]:
    budgets = {str(group): float(value) for group, value in miss_budgets.items()}
    if set(budgets) != set(table.group_weights):
        raise ValueError("miss_budgets must contain every group exactly once")
    if any(
        not np.isfinite(value) or value < 0.0 or value > 1.0
        for value in budgets.values()
    ):
        raise ValueError("group miss budgets must lie in [0, 1]")
    return budgets


def _forward_subset_zeta(values: np.ndarray, controller_count: int) -> np.ndarray:
    """Return ``z[S] = sum_{A subseteq S} values[A]`` in place-safe form."""

    zeta = np.asarray(values).copy()
    for bit_index in range(controller_count):
        half = 1 << bit_index
        blocks = zeta.reshape(-1, 2 * half)
        blocks[:, half:] += blocks[:, :half]
    return zeta


def _signature_mask_index(table: FailureSignatureTable) -> np.ndarray:
    powers = np.left_shift(
        np.uint64(1), np.arange(table.n_controllers, dtype=np.uint64)
    )
    return np.sum(
        table.signatures.astype(np.uint64) * powers[None, :],
        axis=1,
        dtype=np.uint64,
    ).astype(np.int64)


def build_signature_zeta_frontier(
    table: FailureSignatureTable,
    *,
    include_empty: bool = False,
    max_controllers: int = 20,
) -> pd.DataFrame:
    """Build the complete portfolio frontier from signature masses.

    If a signature has epsilon-good mask ``A``, portfolio ``S`` misses it
    exactly when ``A`` is a subset of the complement of ``S``.  A forward
    subset-zeta transform therefore evaluates all ``2**K`` portfolios in
    ``O((|G| + 1) K 2**K)`` arithmetic operations.  Portfolio size and
    additive cost use a low-bit recurrence, adding only ``O(2**K)`` work.
    Counts remain ``int64`` and arbitrary positive task weights remain
    ``float64``; every registered group is carried through the transform.
    """

    limit = int(max_controllers)
    if limit <= 0 or table.n_controllers > limit:
        raise ValueError(
            f"zeta frontier construction is limited to at most {limit} controllers"
        )
    k = table.n_controllers
    state_count = 1 << k
    full = state_count - 1
    signature_masks = _signature_mask_index(table)

    def dense_mass(values: np.ndarray, *, integer: bool) -> np.ndarray:
        dtype = np.int64 if integer else np.float64
        dense = np.zeros(state_count, dtype=dtype)
        np.add.at(dense, signature_masks, np.asarray(values, dtype=dtype))
        return dense

    total_count_zeta = _forward_subset_zeta(
        dense_mass(table.signature_counts, integer=True), k
    )
    total_weight_zeta = _forward_subset_zeta(
        dense_mass(table.signature_weights, integer=False), k
    )
    group_count_zeta = {
        group: _forward_subset_zeta(dense_mass(values, integer=True), k)
        for group, values in sorted(table.group_counts.items())
    }
    group_weight_zeta = {
        group: _forward_subset_zeta(dense_mass(values, integer=False), k)
        for group, values in sorted(table.group_weights.items())
    }

    sizes = np.zeros(state_count, dtype=np.int64)
    costs = np.zeros(state_count, dtype=np.float64)
    for mask in range(1, state_count):
        low_bit = mask & -mask
        parent = mask ^ low_bit
        controller = low_bit.bit_length() - 1
        sizes[mask] = sizes[parent] + 1
        costs[mask] = costs[parent] + table.controller_costs[controller]

    start = 0 if include_empty else 1
    masks = np.arange(start, state_count, dtype=np.int64)
    complements = full ^ masks
    data: dict[str, object] = {
        "mask": masks,
        "size": sizes[masks],
        "cost": costs[masks],
        "total_miss_n": total_count_zeta[complements],
        "total_weighted_miss": total_weight_zeta[complements],
    }
    for group in sorted(table.group_counts):
        data[f"{group}_miss_n"] = group_count_zeta[group][complements]
        data[f"{group}_weighted_miss"] = group_weight_zeta[group][complements]
        data[f"{group}_n"] = np.full(
            len(masks), int(table.group_counts[group].sum()), dtype=np.int64
        )
        data[f"{group}_weight"] = np.full(
            len(masks), float(table.group_weights[group].sum()), dtype=float
        )
    return pd.DataFrame(data)


def certify_continuous_auxiliary_projection(
    table: FailureSignatureTable,
    *,
    members: Sequence[Controller],
    miss_budgets: Mapping[str, float],
) -> ContinuousAuxiliaryProjectionCertificate:
    """Certify exactness of the continuous-auxiliary MILP projection.

    For fixed binary controller coordinates, an uncovered signature forces
    ``u_q >= 1`` and the unit upper bound forces equality.  A covered signature
    admits ``u_q = 0``.  Since every group coefficient of ``u`` is
    nonnegative, this Boolean vector is the componentwise minimum auxiliary
    vector and existential LP feasibility is exactly Boolean miss feasibility.
    """

    budgets = _validated_budgets(table, miss_budgets)
    evaluation = evaluate_signature_portfolio(table, members)
    selected = evaluation.members
    covered = (
        table.signatures[:, list(selected)].any(axis=1)
        if selected
        else np.zeros(table.n_signatures, dtype=bool)
    )
    minimum_u = (~covered).astype(float)
    misses = {
        group: float(np.dot(table.group_weights[group], minimum_u))
        for group in sorted(budgets)
    }
    allowances = {
        group: _weighted_miss_allowance(
            table.group_weights[group],
            budgets[group],
            integer_weight_contract=table.integer_weight_contract,
        )
        for group in sorted(budgets)
    }
    group_feasible = {
        group: _weighted_miss_is_feasible(
            misses[group],
            allowances[group],
            integer_weight_contract=table.integer_weight_contract,
        )
        for group in sorted(budgets)
    }
    boolean_feasible = all(
        _weighted_miss_is_feasible(
            evaluation.group_weighted_misses[group],
            allowances[group],
            integer_weight_contract=table.integer_weight_contract,
        )
        for group in sorted(budgets)
    )
    projected_feasible = all(group_feasible.values())
    projection_exact = bool(
        projected_feasible == boolean_feasible
        and all(
            np.isclose(
                misses[group],
                evaluation.group_weighted_misses[group],
                rtol=0.0,
                atol=1.0e-10,
            )
            for group in budgets
        )
    )
    return ContinuousAuxiliaryProjectionCertificate(
        members=selected,
        minimum_auxiliaries=tuple(float(value) for value in minimum_u),
        group_weighted_misses=misses,
        group_allowances=allowances,
        group_feasible=group_feasible,
        projected_feasible=projected_feasible,
        boolean_feasible=boolean_feasible,
        projection_exact=projection_exact,
    )


def certify_greedy_submodular_cover(
    table: FailureSignatureTable,
    *,
    miss_budgets: Mapping[str, float],
    verify_optimum: bool = False,
) -> GreedySubmodularCoverCertificate:
    """Record Wolsey's safe ``H_D <= 1 + ln(D)`` greedy guarantee.

    For group ``g``, the progress term is weighted coverage truncated at
    ``W_g - floor(beta_g W_g)``.  With integer-valued group masses this is a
    normalized monotone integer-valued submodular function.  Their sum has
    target ``Q``.  Let ``D`` be the largest value contributed from the empty
    portfolio by any single controller.  Ratio greedy with arbitrary positive
    additive controller costs and no size cap is therefore an ``H_D``
    approximation whenever the full library reaches the target.  The looser
    ``H_Q`` argument is intentionally not reported as the theorem factor.
    ``verify_optimum`` additionally checks the inequality against complete
    enumeration for small libraries.
    """

    budgets = _validated_budgets(table, miss_budgets)
    if not np.all(np.isfinite(table.controller_costs)) or np.any(
        table.controller_costs <= 0.0
    ):
        raise ValueError("controller costs must be finite and positive")
    if not table.integer_weight_contract or any(
        not np.all(np.isclose(values, np.rint(values), rtol=0.0, atol=1.0e-10))
        for values in table.group_weights.values()
    ):
        raise ValueError(
            "the harmonic certificate requires integer-valued group masses"
        )

    total_mass = {
        group: int(round(float(table.group_weights[group].sum())))
        for group in sorted(budgets)
    }
    allowances = {
        group: int(
            round(
                _weighted_miss_allowance(
                    table.group_weights[group],
                    budgets[group],
                    integer_weight_contract=True,
                )
            )
        )
        for group in sorted(budgets)
    }
    target = sum(total_mass[group] - allowances[group] for group in budgets)
    full_coverage = table.signatures.any(axis=1)
    unreachable = {
        group: int(round(float(table.group_weights[group][~full_coverage].sum())))
        > allowances[group]
        for group in sorted(budgets)
    }
    if any(unreachable.values()):
        raise PortfolioInfeasible(
            "the full controller library cannot reach the submodular-cover target"
        )

    greedy = greedy_signature_portfolio(table, miss_budgets=budgets)
    remaining = sum(
        max(0, int(round(greedy.group_weighted_misses[group])) - allowances[group])
        for group in budgets
    )
    achieved = target - remaining
    if achieved != target:
        raise RuntimeError("uncapped greedy terminated before reaching the target")

    singleton_gains: list[int] = []
    for controller in range(table.n_controllers):
        covered = table.signatures[:, controller]
        gain = sum(
            min(
                total_mass[group] - allowances[group],
                int(round(float(table.group_weights[group][covered].sum()))),
            )
            for group in budgets
        )
        singleton_gains.append(gain)
    maximum_singleton_gain = max(singleton_gains, default=0)
    if target > 0 and maximum_singleton_gain <= 0:
        raise RuntimeError("a positive submodular-cover target has no positive gain")

    if maximum_singleton_gain <= 1:
        harmonic = 1.0
        logarithmic = 1.0
    else:
        harmonic = math.fsum(
            1.0 / value for value in range(1, maximum_singleton_gain + 1)
        )
        logarithmic = 1.0 + math.log(float(maximum_singleton_gain))

    optimum_cost: float | None = None
    observed_ratio: float | None = None
    bound_verified: bool | None = None
    if verify_optimum:
        optimum = solve_signature_portfolio_by_enumeration(table, miss_budgets=budgets)
        optimum_cost = float(optimum.objective)
        if optimum_cost == 0.0:
            observed_ratio = 1.0 if greedy.objective == 0.0 else float("inf")
        else:
            observed_ratio = float(greedy.objective / optimum_cost)
        bound_verified = bool(greedy.objective <= harmonic * optimum_cost + 1.0e-10)
        if not bound_verified:
            raise RuntimeError("exhaustive check contradicts the harmonic certificate")

    return GreedySubmodularCoverCertificate(
        members=greedy.members,
        greedy_cost=float(greedy.objective),
        target_value=int(target),
        achieved_value=int(achieved),
        maximum_singleton_gain=int(maximum_singleton_gain),
        harmonic_factor=float(harmonic),
        logarithmic_factor=float(logarithmic),
        conditions_satisfied=True,
        exact_optimum_cost=optimum_cost,
        observed_ratio=observed_ratio,
        exhaustive_bound_verified=bound_verified,
    )


def _mask_incidence(mask: int, controller_count: int) -> np.ndarray:
    return np.fromiter(
        (float(bool(mask & (1 << index))) for index in range(controller_count)),
        dtype=float,
        count=controller_count,
    )


def _registered_frontier_feasibility(
    frame: pd.DataFrame, budgets: Mapping[str, float]
) -> np.ndarray:
    """Use the registered mass inequality without an expanded rate tolerance.

    Weighted frontiers keep their supplied masses: an integer aggregate does
    not prove that individual task weights are integral.  For integral misses,
    comparison with the unfloored weighted allowance is already equivalent to
    its floor.  Count-only frontiers explicitly use that integer floor.
    """

    feasible = np.ones(len(frame), dtype=bool)
    for group, budget in budgets.items():
        miss_column, mass_column = _group_mass_columns(frame, group)
        integer_count_contract = miss_column == f"{group}_miss_n"
        misses = frame[miss_column].to_numpy(dtype=float)
        allowances = float(budget) * frame[mass_column].to_numpy(dtype=float)
        if integer_count_contract:
            allowances = np.floor(allowances)
        feasible &= np.fromiter(
            (
                _weighted_miss_is_feasible(
                    miss,
                    allowance,
                    integer_weight_contract=integer_count_contract,
                )
                for miss, allowance in zip(misses, allowances, strict=True)
            ),
            dtype=bool,
            count=len(frame),
        )
    return feasible


def _cost_certificate_family(frontier, miss_budgets, target_masks, max_size):
    frame, budgets = _validated_frontier(frontier, miss_budgets)
    count = _complete_frontier_controller_count(frame)
    limit = None
    if max_size is not None:
        if isinstance(max_size, (bool, np.bool_)):
            raise ValueError("max_size must be an integer")
        try:
            limit = index(max_size)
        except TypeError as error:
            raise ValueError("max_size must be an integer") from error
        if not 1 <= limit <= count:
            raise ValueError("max_size must lie between one and the controller count")
    registered = _registered_frontier_feasibility(frame, budgets)
    if limit is not None:
        registered &= frame["size"].to_numpy(dtype=int) <= limit
    feasible = tuple(sorted(frame.loc[registered, "mask"].astype(int)))
    if not feasible:
        raise PortfolioInfeasible("no frontier portfolio satisfies the miss budgets")
    try:
        if any(isinstance(mask, (bool, np.bool_)) for mask in target_masks):
            raise TypeError("boolean mask")
        targets = tuple(sorted({index(mask) for mask in target_masks}))
    except TypeError as error:
        raise ValueError("target_masks must contain integer masks") from error
    if not targets or any(mask not in feasible for mask in targets):
        raise ValueError("target_masks must be a nonempty subset of feasible masks")
    minimal = []
    for mask in sorted(feasible, key=lambda value: (value.bit_count(), value)):
        if not any((member & mask) == member for member in minimal):
            minimal.append(mask)
    return count, limit, feasible, targets, tuple(sorted(minimal))


def _exact_cost_input(values, count, name):
    represented = np.asarray(values, dtype=float)
    if represented.shape != (count,):
        raise ValueError(f"{name} must align with the controller library")
    if not np.all(np.isfinite(represented)) or np.any(represented <= 0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return tuple(Fraction.from_float(float(value)) for value in represented)


def _cost_tolerance_metadata(value):
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("cost_atol must be finite and nonnegative")
    # The numeric tolerance is recorded as metadata and cannot change truth.
    return tolerance


def _cost_comparison(candidate, other, normalized):
    return tuple(
        (int(bool(candidate & (1 << k))) - int(bool(other & (1 << k)))) * value
        for k, value in enumerate(normalized)
    )


def _exact_optimal_masks(feasible, costs):
    values = {
        mask: sum((cost for k, cost in enumerate(costs) if mask & (1 << k)), Fraction())
        for mask in feasible
    }
    minimum = min(values.values())
    return tuple(mask for mask in feasible if values[mask] == minimum)


def _display_exact_costs(costs):
    if costs is None:
        return None
    try:
        displayed = tuple(float(value) for value in costs)
    except OverflowError as error:
        raise ExactLPCertificationError(
            "exact witness exceeds finite float display range"
        ) from error
    if not all(math.isfinite(value) for value in displayed):
        raise ExactLPCertificationError(
            "exact witness exceeds finite float display range"
        )
    return displayed


def _certified_witness(feasible, candidate, costs, lower, upper):
    if any(
        not lo <= value <= hi for value, lo, hi in zip(costs, lower, upper, strict=True)
    ):
        raise ExactLPCertificationError("exact cost witness is outside its box")
    ties = _exact_optimal_masks(feasible, costs)
    if candidate not in ties:
        raise ExactLPCertificationError("exact cost witness is not globally optimal")
    return ties


def certify_cost_box_structure(
    frontier: pd.DataFrame,
    *,
    miss_budgets: Mapping[str, float],
    target_masks: Sequence[int],
    cost_lower: Sequence[float],
    cost_upper: Sequence[float],
    max_size: int | None = None,
    cost_atol: float = 1.0e-8,
) -> CostBoxStructureCertificate:
    """Prove strict optimum-family statements on a positive represented box.

    Costs mean exact binary-rational values after float conversion, not intended
    decimals. Relative-cost variables and common-scale normalized coefficients
    feed bounded rational LPs with exact primal/dual or Farkas posteriors.
    'cost_atol' is reported metadata only and never merges exact optima.
    Float witnesses are display approximations; 'witness_costs_exact' is the
    authoritative serializable rational witness. Unsupported work fails closed.
    """

    count, limit, feasible, targets, minimal = _cost_certificate_family(
        frontier, miss_budgets, target_masks, max_size
    )
    lower = _exact_cost_input(cost_lower, count, "cost_lower")
    upper = _exact_cost_input(cost_upper, count, "cost_upper")
    if any(lo > hi for lo, hi in zip(lower, upper, strict=True)):
        raise ValueError("cost bounds must be ordered")
    tolerance = _cost_tolerance_metadata(cost_atol)
    center = tuple((lo + hi) / 2 for lo, hi in zip(lower, upper, strict=True))
    scale = max(center)
    normalized = tuple(value / scale for value in center)
    relative_bounds = tuple(
        (lo / mid, hi / mid) for lo, hi, mid in zip(lower, upper, center, strict=True)
    )
    checked = 0
    first = None

    def finish(all_target, some_target, witness):
        candidate, costs, ties, kind = (
            witness if witness is not None else (None, None, (), None)
        )
        return CostBoxStructureCertificate(
            target_masks=targets,
            feasible_masks=feasible,
            inclusion_minimal_feasible_masks=minimal,
            dominated_feasible_mask_count=len(feasible) - len(minimal),
            cost_lower=tuple(map(float, lower)),
            cost_upper=tuple(map(float, upper)),
            cost_atol=tolerance,
            all_optima_in_target_over_full_box=all_target,
            target_optimum_exists_over_full_box=some_target,
            checked_non_target_masks=checked,
            witness_mask=candidate,
            witness_costs=_display_exact_costs(costs),
            witness_optimal_masks=ties,
            witness_kind=kind,
            max_size=limit,
            witness_costs_exact=None if costs is None else tuple(map(str, costs)),
        )

    for candidate in (mask for mask in minimal if mask not in targets):
        checked += 1
        comparisons = tuple(
            _cost_comparison(candidate, other, normalized) for other in minimal
        )
        possible = solve_exact_lp(
            (Fraction(),) * count,
            comparisons,
            (Fraction(),) * len(comparisons),
            relative_bounds,
        )
        if possible.status == "infeasible":
            continue
        if possible.status != "optimal":
            raise ExactLPCertificationError(
                "bounded cost-feasibility LP was not optimal"
            )
        costs = tuple(
            mid * value for mid, value in zip(center, possible.x, strict=True)
        )
        ties = _certified_witness(feasible, candidate, costs, lower, upper)
        if first is None:
            first = (candidate, costs, ties, "non_target_tie")
        # delta is a cost margin divided by the common positive cost scale.
        # Its sign, not a tolerance comparison, distinguishes the weak claim.
        margin_rows = [(*row, Fraction()) for row in comparisons]
        margin_rows.extend(
            (*_cost_comparison(candidate, target, normalized), Fraction(1))
            for target in targets
        )
        margin = solve_exact_lp(
            (*((Fraction(),) * count), Fraction(-1)),
            margin_rows,
            (Fraction(),) * len(margin_rows),
            (*relative_bounds, (Fraction(), None)),
        )
        if margin.status != "optimal":
            raise ExactLPCertificationError("bounded cost-margin LP was not optimal")
        if margin.objective < 0:
            costs = tuple(
                mid * value for mid, value in zip(center, margin.x[:-1], strict=True)
            )
            ties = _certified_witness(feasible, candidate, costs, lower, upper)
            if set(ties).intersection(targets):
                raise ExactLPCertificationError(
                    "positive exact margin retained a target optimum"
                )
            return finish(False, False, (candidate, costs, ties, "no_target_optimum"))
        if margin.objective != 0:
            raise ExactLPCertificationError("zero margin feasibility was contradicted")
    return finish(first is None, True, first)


def certify_symmetric_cost_robustness_radius(
    frontier: pd.DataFrame,
    *,
    miss_budgets: Mapping[str, float],
    target_masks: Sequence[int],
    nominal_costs: Sequence[float],
    radius_domain_upper: float = 1.0,
    max_size: int | None = None,
    cost_atol: float = 1.0e-8,
) -> SymmetricCostRobustnessCertificate:
    """Certify the exact first structural boundary in relative cost coordinates.

    The authoritative radius/witness are serialized rational strings in the
    '*_exact' fields. 'robustness_radius' is rounded DOWN for a conservative
    float display; its exactness flag concerns the rational certificate.
    'failure_at_radius' concerns that exact boundary, not its rounded display.
    At rho=1 all-zero costs also expose nonminimal feasible supersets.
    'cost_atol' is reported metadata and cannot change any statement.
    """

    count, limit, feasible, targets, minimal = _cost_certificate_family(
        frontier, miss_budgets, target_masks, max_size
    )
    nominal = _exact_cost_input(nominal_costs, count, "nominal_costs")
    domain_float = float(radius_domain_upper)
    if not math.isfinite(domain_float) or not 0 < domain_float <= 1:
        raise ValueError("radius_domain_upper must lie in (0, 1]")
    domain = Fraction.from_float(domain_float)
    tolerance = _cost_tolerance_metadata(cost_atol)
    nominal_optima = _exact_optimal_masks(feasible, nominal)
    nominal_valid = set(nominal_optima) <= set(targets)
    non_target = tuple(mask for mask in minimal if mask not in targets)

    def finish(radius, candidate, costs, checked, domain_limit):
        ties = _exact_optimal_masks(feasible, costs) if costs is not None else ()
        if candidate is not None and candidate not in ties:
            raise ExactLPCertificationError("exact radius witness is not optimal")
        lower = tuple(value * (1 - radius) for value in nominal)
        upper = tuple(value * (1 + radius) for value in nominal)
        if costs is not None:
            _certified_witness(feasible, candidate, costs, lower, upper)
        display_radius = float(radius)
        if Fraction.from_float(display_radius) > radius:
            display_radius = math.nextafter(display_radius, -math.inf)
        kind = None
        if candidate is not None:
            kind = (
                "non_target_tie"
                if set(ties).intersection(targets)
                else "no_target_optimum"
            )
        return SymmetricCostRobustnessCertificate(
            target_masks=targets,
            feasible_masks=feasible,
            inclusion_minimal_feasible_masks=minimal,
            nominal_costs=tuple(map(float, nominal)),
            radius_domain_upper=domain_float,
            cost_atol=tolerance,
            robustness_radius=display_radius,
            nominal_all_optima_in_target=nominal_valid,
            radius_is_exact=True,
            radius_is_domain_limit=domain_limit,
            safe_for_every_strictly_smaller_radius=True,
            failure_at_radius=candidate is not None,
            checked_non_target_minimal_masks=checked,
            witness_mask=candidate,
            witness_costs=_display_exact_costs(costs),
            witness_optimal_masks=ties,
            witness_kind=kind,
            max_size=limit,
            robustness_radius_exact=str(radius),
            witness_costs_exact=None if costs is None else tuple(map(str, costs)),
        )

    if not nominal_valid:
        candidate = min(mask for mask in nominal_optima if mask not in targets)
        return finish(Fraction(), candidate, nominal, 0, False)

    scale = max(nominal)
    normalized = tuple(value / scale for value in nominal)
    box_rows = []
    box_rhs = []
    for k in range(count):
        unit = tuple(Fraction(int(j == k)) for j in range(count))
        box_rows.extend([(*unit, Fraction(-1)), (*(-v for v in unit), Fraction(-1))])
        box_rhs.extend([Fraction(1), Fraction(-1)])
    bounds = (*((Fraction(), Fraction(2)),) * count, (Fraction(), domain))
    best = None
    for candidate in non_target:
        comparisons = [
            (*_cost_comparison(candidate, other, normalized), Fraction())
            for other in minimal
        ]
        result = solve_exact_lp(
            (*((Fraction(),) * count), Fraction(1)),
            [*comparisons, *box_rows],
            [*((Fraction(),) * len(comparisons)), *box_rhs],
            bounds,
        )
        if result.status == "infeasible":
            continue
        if result.status != "optimal":
            raise ExactLPCertificationError("bounded radius LP was not optimal")
        costs = tuple(
            value * factor for value, factor in zip(nominal, result.x[:-1], strict=True)
        )
        proposal = (result.objective, candidate, costs)
        if best is None or proposal[:2] < best[:2]:
            best = proposal
    if best is not None:
        radius, candidate, costs = best
        return finish(radius, candidate, costs, len(non_target), radius == domain)
    remaining = tuple(mask for mask in feasible if mask not in targets)
    if domain == 1 and remaining:
        return finish(
            domain, min(remaining), (Fraction(),) * count, len(non_target), True
        )
    return finish(domain, None, None, len(non_target), True)
