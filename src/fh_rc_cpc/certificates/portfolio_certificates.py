"""Certificates and computational audits for controller-library compression.

The functions in this module operate either on a complete, aggregate portfolio
frontier or on an in-memory :class:`FailureHypergraph`.  They never require task
identifiers, timestamps, trajectories, or plant labels, which makes their
outputs suitable for the privacy-screened Paper A evidence bundle.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, eye, hstack, vstack

from ..qualification.failure_hypergraph import FailureHypergraph
from ..qualification.general_portfolio import (
    PortfolioInfeasible,
    _uses_integer_weight_contract,
    _weighted_miss_allowance,
    _weighted_miss_is_feasible,
)
from ..qualification.signature_compression import (
    compress_failure_signatures,
    evaluate_signature_portfolio,
    evaluate_task_portfolio,
)
from ..qualification.signature_solver import solve_signature_portfolio

_BUDGET_ARITHMETIC_DOMAIN = "exact_costs_decimal_frontier_v1"


@dataclass(frozen=True)
class FrontierCertificate:
    """Auditable certificate for one selected portfolio on a complete frontier."""

    selected_mask: int
    selected_size: int
    minimum_size: int
    optimal_masks: tuple[int, ...]
    selected_in_optimal_class: bool
    group_miss_counts: dict[str, int]
    group_population_counts: dict[str, int]
    group_allowed_miss_counts: dict[str, int]
    group_slack_counts: dict[str, int]
    selected_uniform_tightening_factor: float
    singleton_uniform_relaxation_factor: float


@dataclass(frozen=True)
class RefinedFrontierCertificate:
    """Canonical optimum-class ranking paired with its finite certificate."""

    ranking: pd.DataFrame
    certificate: FrontierCertificate


@dataclass(frozen=True)
class EnumeratedOptimalClass:
    """Small-library frontier and exact or explicitly near-optimal cost class."""

    optimal_cost: float
    optimal_masks: tuple[int, ...]
    representative_mask: int
    frontier: pd.DataFrame


@dataclass(frozen=True)
class AdditiveCostTighteningCertificate:
    """Exact rational-frontier proof, not an original-graph IEEE certificate."""

    selected_mask: int
    selected_cost: float
    selected_required_multiplier: float
    selected_required_multiplier_exact: str
    reference_multiplier: float
    reference_multiplier_exact: str
    selected_feasible_at_reference: bool
    optimal_throughout_tightening_interval: bool
    competitor_masks_with_lower_cost: tuple[int, ...]
    arithmetic_domain: str = field(default=_BUDGET_ARITHMETIC_DOMAIN, kw_only=True)
    graph_ieee_equivalence_certified: bool = field(default=False, kw_only=True)


def enumerate_complete_portfolio_frontier(
    graph: FailureHypergraph,
    *,
    include_empty: bool = False,
    max_controllers: int = 20,
) -> pd.DataFrame:
    """Enumerate every portfolio and its exact task/group outcomes for small K.

    The output is ordered by integer mask and contains both count and weighted
    miss columns.  It is therefore a reusable finite certificate, rather than
    only the winning solution returned by a solver.
    """

    if max_controllers <= 0 or graph.n_controllers > int(max_controllers):
        raise ValueError(
            f"complete enumeration is limited to at most {int(max_controllers)} "
            "controllers"
        )
    start = 0 if include_empty else 1
    group_population = {
        group: int(membership.sum()) for group, membership in graph.groups.items()
    }
    group_weight = {
        group: float(graph.task_weights[membership].sum())
        for group, membership in graph.groups.items()
    }
    rows: list[dict[str, object]] = []
    for mask in range(start, 1 << graph.n_controllers):
        members = tuple(
            index for index in range(graph.n_controllers) if mask & (1 << index)
        )
        evaluation = evaluate_task_portfolio(graph, members)
        row: dict[str, object] = {
            "mask": mask,
            "size": len(members),
            "cost": evaluation.cost,
            "total_miss_n": evaluation.total_miss_count,
            "total_weighted_miss": evaluation.total_weighted_miss,
        }
        for group in sorted(graph.groups):
            row[f"{group}_miss_n"] = evaluation.group_misses[group]
            row[f"{group}_weighted_miss"] = evaluation.group_weighted_misses[group]
            row[f"{group}_n"] = group_population[group]
            row[f"{group}_weight"] = group_weight[group]
        rows.append(row)
    return pd.DataFrame(rows)


def enumerate_optimal_masks(
    graph: FailureHypergraph,
    *,
    miss_budgets: Mapping[str, float],
    max_size: int | None = None,
    cost_atol: float = 0.0,
    include_empty: bool = False,
) -> EnumeratedOptimalClass:
    """Return the exact registered-cost optimum class for a complete small menu.

    Additive costs are compared as exact represented binary rationals, not
    rounded frontier display values. An explicitly positive ``cost_atol``
    instead requests all feasible masks within that additive near-optimal
    band. The frontier and ``optimal_cost`` retain their float display format.
    """

    budgets = {str(group): float(value) for group, value in miss_budgets.items()}
    if set(budgets) != set(graph.groups):
        raise ValueError("miss budgets must contain every graph group exactly once")
    if any(
        not np.isfinite(value) or value < 0.0 or value > 1.0
        for value in budgets.values()
    ):
        raise ValueError("miss budgets must lie in [0, 1]")
    minimum_size = 0 if include_empty else 1
    if max_size is not None and (
        max_size < minimum_size or max_size > graph.n_controllers
    ):
        raise ValueError("max_size must lie within the controller library")
    if not np.isfinite(cost_atol) or cost_atol < 0:
        raise ValueError("cost_atol must be finite and nonnegative")
    frontier = enumerate_complete_portfolio_frontier(
        graph,
        include_empty=include_empty,
    )
    feasible = np.ones(len(frontier), dtype=bool)
    integer_contract = _uses_integer_weight_contract(graph.task_weights)
    for group, budget in budgets.items():
        allowance = _weighted_miss_allowance(
            graph.task_weights[graph.groups[group]],
            budget,
            integer_weight_contract=integer_contract,
        )
        feasible &= np.fromiter(
            (
                _weighted_miss_is_feasible(
                    miss,
                    allowance,
                    integer_weight_contract=integer_contract,
                )
                for miss in frontier[f"{group}_weighted_miss"].to_numpy(dtype=float)
            ),
            dtype=bool,
            count=len(frontier),
        )
    if max_size is not None:
        feasible &= frontier["size"].to_numpy(dtype=int) <= int(max_size)
    candidates = frontier.loc[feasible]
    if candidates.empty:
        raise PortfolioInfeasible("no enumerated portfolio satisfies all group budgets")
    controller_costs = tuple(
        Fraction.from_float(float(cost)) for cost in graph.controller_costs
    )
    exact_costs = {
        int(mask): sum(
            (
                cost
                for index, cost in enumerate(controller_costs)
                if mask & (1 << index)
            ),
            Fraction(0),
        )
        for mask in candidates["mask"].astype(int)
    }
    optimum_mask = min(exact_costs, key=lambda mask: (exact_costs[mask], mask))
    threshold = exact_costs[optimum_mask] + Fraction.from_float(float(cost_atol))
    masks = tuple(
        sorted(mask for mask, cost in exact_costs.items() if cost <= threshold)
    )
    optimum = float(candidates.loc[candidates["mask"].eq(optimum_mask), "cost"].iloc[0])
    return EnumeratedOptimalClass(
        optimal_cost=optimum,
        optimal_masks=masks,
        representative_mask=masks[0],
        frontier=frontier,
    )


def robust_tie_refinement(
    frontier: pd.DataFrame,
    *,
    miss_budgets: Mapping[str, float],
) -> pd.DataFrame:
    """Rank the primary optimum class by robust budget slack and miss counts.

    Primary optimality is minimum feasible cardinality.  The secondary order is
    the worst integer-budget utilization, pooled misses, total group misses,
    and finally the integer mask.  This order is deterministic and uses only
    development evidence. Any weighted columns must equal their count columns
    exactly; other weighted frontiers require a weighted graph API. Allowances
    use native IEEE multiplication followed by floor, without an added epsilon.
    """

    frame, budgets = _validated_unit_frontier(frontier, miss_budgets)
    feasible = np.ones(len(frame), dtype=bool)
    utilization = np.zeros(len(frame), dtype=float)
    total_misses = np.zeros(len(frame), dtype=int)
    for group, budget in budgets.items():
        misses = frame[f"{group}_miss_n"].to_numpy(dtype=int)
        allowed = _unit_frontier_allowances(frame, group, budget)
        feasible &= misses <= allowed
        ratio = np.divide(
            misses,
            allowed,
            out=np.where(misses == 0, 0.0, np.inf).astype(float),
            where=allowed > 0,
        )
        utilization = np.maximum(utilization, ratio)
        total_misses += misses
    candidates = frame.loc[feasible].copy()
    if candidates.empty:
        raise PortfolioInfeasible("no frontier portfolio satisfies the miss budgets")
    minimum_size = int(candidates["size"].min())
    candidates = candidates.loc[candidates["size"].eq(minimum_size)].copy()
    positions = candidates.index.map(frame.index.get_loc).to_numpy(dtype=int)
    candidates["worst_budget_utilization"] = utilization[positions]
    candidates["total_group_misses"] = total_misses[positions]
    pooled = "pooled" if "pooled" in budgets else next(iter(budgets))
    candidates["refinement_pooled_misses"] = candidates[f"{pooled}_miss_n"].astype(int)
    candidates = candidates.sort_values(
        [
            "worst_budget_utilization",
            "refinement_pooled_misses",
            "total_group_misses",
            "mask",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    candidates.insert(0, "robust_rank", np.arange(1, len(candidates) + 1))
    return candidates


def build_refined_frontier_certificate(
    frontier: pd.DataFrame,
    *,
    miss_budgets: Mapping[str, float],
) -> RefinedFrontierCertificate:
    """Enumerate, refine, and certify the primary optimum class.

    The complete frontier is filtered by the registered group budgets, locked
    to minimum cardinality, and ordered by the four-coordinate scientific rule
    used in the paper.  This wrapper deliberately differs from the generic
    MILP solver's minimum-mask reproducibility tie-break.
    """

    ranking = robust_tie_refinement(frontier, miss_budgets=miss_budgets)
    selected_mask = int(ranking.iloc[0]["mask"])
    certificate = build_frontier_certificate(
        frontier,
        selected_mask=selected_mask,
        miss_budgets=miss_budgets,
    )
    return RefinedFrontierCertificate(ranking=ranking, certificate=certificate)


def _frontier_columns(groups: Sequence[str]) -> list[str]:
    columns = ["mask", "size"]
    for group in groups:
        columns.extend((f"{group}_miss_n", f"{group}_n"))
    return columns


def _weighted_frontier_columns(group: str) -> tuple[str, str]:
    return f"{group}_weighted_miss", f"{group}_weight"


def _validated_frontier(
    frontier: pd.DataFrame,
    miss_budgets: Mapping[str, float],
    *,
    mass_atol: float = 1.0e-10,
) -> tuple[pd.DataFrame, dict[str, float]]:
    budgets = {str(group): float(value) for group, value in miss_budgets.items()}
    if not budgets:
        raise ValueError("at least one miss budget is required")
    if any(
        not np.isfinite(value) or value < 0 or value > 1 for value in budgets.values()
    ):
        raise ValueError("miss budgets must lie in [0, 1]")
    tolerance = float(mass_atol)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("mass_atol must be finite and nonnegative")
    base_columns = _frontier_columns(tuple(budgets))
    missing = sorted(set(base_columns) - set(frontier.columns))
    if missing:
        raise ValueError(f"frontier is missing required columns: {', '.join(missing)}")
    weighted_columns: list[str] = []
    for group in budgets:
        columns = _weighted_frontier_columns(group)
        present = tuple(column in frontier.columns for column in columns)
        if any(present) and not all(present):
            raise ValueError(
                f"weighted frontier for {group} requires both miss and total mass"
            )
        if all(present):
            weighted_columns.extend(columns)
    frame = (
        frontier.loc[:, base_columns + weighted_columns].copy().reset_index(drop=True)
    )
    if frame.empty:
        raise ValueError("frontier must contain nonempty portfolio rows")
    for column in base_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        values = numeric.to_numpy()
        if not np.all(np.isfinite(values)) or (
            not pd.api.types.is_integer_dtype(numeric.dtype)
            and not np.all(values == np.floor(values))
        ):
            raise ValueError("frontier counts, masks and sizes must be finite integers")
        frame[column] = numeric.astype(int)
    if frame["mask"].duplicated().any() or (frame["mask"] <= 0).any():
        raise ValueError("frontier must contain one row per nonempty portfolio mask")
    if not all(
        int(size) == int(mask).bit_count()
        for mask, size in zip(frame["mask"], frame["size"])
    ):
        raise ValueError("frontier sizes must equal the positive mask population count")
    for group in budgets:
        misses = pd.to_numeric(frame[f"{group}_miss_n"], errors="coerce")
        populations = pd.to_numeric(frame[f"{group}_n"], errors="coerce")
        if misses.isna().any() or populations.isna().any():
            raise ValueError("frontier miss and population counts must be numeric")
        if (
            (misses < 0).any()
            or (populations <= 0).any()
            or (misses > populations).any()
        ):
            raise ValueError("frontier miss counts must lie within group populations")
        if populations.nunique() != 1:
            raise ValueError(
                f"group population for {group} changes across the frontier"
            )
        frame[f"{group}_miss_n"] = misses.astype(int)
        frame[f"{group}_n"] = populations.astype(int)
        weighted_miss_column, weight_column = _weighted_frontier_columns(group)
        if weighted_miss_column not in frame:
            continue
        weighted_misses = pd.to_numeric(frame[weighted_miss_column], errors="coerce")
        weights = pd.to_numeric(frame[weight_column], errors="coerce")
        if weighted_misses.isna().any() or weights.isna().any():
            raise ValueError("frontier weighted miss and total mass must be numeric")
        miss_values = weighted_misses.to_numpy(dtype=float)
        weight_values = weights.to_numpy(dtype=float)
        if not np.all(np.isfinite(miss_values)) or not np.all(
            np.isfinite(weight_values)
        ):
            raise ValueError("frontier weighted miss and total mass must be finite")
        reference_weight = float(weight_values[0])
        scale = max(1.0, abs(reference_weight))
        if (
            reference_weight <= 0.0
            or np.max(np.abs(weight_values - reference_weight)) > tolerance * scale
        ):
            raise ValueError(f"group weighted mass for {group} changes across frontier")
        if np.any(miss_values < -tolerance) or np.any(
            miss_values > reference_weight + tolerance * scale
        ):
            raise ValueError("weighted misses must lie within group weighted mass")
        frame[weighted_miss_column] = miss_values
        frame[weight_column] = weight_values
    frame["mask"] = frame["mask"].astype(int)
    frame["size"] = frame["size"].astype(int)
    return frame, budgets


def _validated_unit_frontier(
    frontier: pd.DataFrame,
    miss_budgets: Mapping[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame, budgets = _validated_frontier(frontier, miss_budgets, mass_atol=0.0)
    for group in budgets:
        weighted_miss, total_weight = _weighted_frontier_columns(group)
        if weighted_miss in frame and (
            not np.array_equal(frame[weighted_miss], frame[f"{group}_miss_n"])
            or not np.array_equal(frame[total_weight], frame[f"{group}_n"])
        ):
            raise ValueError(
                "unit-mass frontier certificates require weighted masses "
                "to equal their counts exactly; use a weighted graph API"
            )
    return frame, budgets


def _unit_frontier_allowances(
    frame: pd.DataFrame,
    group: str,
    budget: float,
    multiplier: float = 1.0,
) -> np.ndarray:
    # Match the native count contract; fractions above one are vacuous.
    effective_budget = min(1.0, float(multiplier) * float(budget))
    return np.floor(effective_budget * frame[f"{group}_n"].to_numpy(dtype=float))


def _group_mass_columns(frame: pd.DataFrame, group: str) -> tuple[str, str]:
    weighted_miss, weight = _weighted_frontier_columns(group)
    if weighted_miss in frame.columns:
        return weighted_miss, weight
    return f"{group}_miss_n", f"{group}_n"


def _required_multiplier(
    frame: pd.DataFrame,
    budgets: Mapping[str, float],
) -> np.ndarray:
    required = np.zeros(len(frame), dtype=float)
    for group, budget in budgets.items():
        miss_column, mass_column = _group_mass_columns(frame, group)
        rate = frame[miss_column].to_numpy(dtype=float) / frame[mass_column].to_numpy(
            dtype=float
        )
        if budget == 0:
            ratio = np.where(rate == 0, 0.0, np.inf)
        else:
            ratio = rate / budget
        required = np.maximum(required, ratio)
    return required


def _complete_frontier_controller_count(frame: pd.DataFrame) -> int:
    masks = tuple(sorted(frame["mask"].astype(int)))
    if not masks or masks[0] != 1:
        raise ValueError("complete frontier must start at nonempty mask 1")
    full = masks[-1]
    controller_count = full.bit_length()
    if full != (1 << controller_count) - 1 or masks != tuple(range(1, full + 1)):
        raise ValueError("frontier must contain every nonempty portfolio mask")
    expected_sizes = np.fromiter(
        (int(mask).bit_count() for mask in frame["mask"]),
        dtype=int,
        count=len(frame),
    )
    if not np.array_equal(frame["size"].to_numpy(dtype=int), expected_sizes):
        raise ValueError("frontier sizes must equal the population count of each mask")
    return controller_count


def _require_exact_budget_tolerances(cost_atol: float, mass_atol: float) -> None:
    for name, value in (("cost_atol", cost_atol), ("mass_atol", mass_atol)):
        if not np.isfinite(value) or value != 0.0:
            raise ValueError(f"exact budget certificates require zero {name}")


def _exact_required_multipliers(
    frame: pd.DataFrame,
    budgets: Mapping[str, object],
    *,
    mass_atol: float = 0.0,
) -> tuple[Fraction | None, ...]:
    _require_exact_budget_tolerances(0.0, mass_atol)

    def mass_fraction(value: object) -> Fraction:
        if isinstance(value, (int, np.integer)):
            return Fraction(int(value))
        return Fraction(str(float(value)))

    exact_budgets: dict[str, Fraction] = {}
    for raw_group, value in budgets.items():
        group = str(raw_group)
        exact_budgets[group] = (
            value if isinstance(value, Fraction) else Fraction(str(value))
        )
    required: list[Fraction | None] = []
    for position in range(len(frame)):
        multiplier = Fraction(0)
        infinite = False
        for group, budget in exact_budgets.items():
            miss_column, mass_column = _group_mass_columns(frame, group)
            miss = mass_fraction(frame[miss_column].iloc[position])
            population = mass_fraction(frame[mass_column].iloc[position])
            if budget == 0:
                if miss > 0:
                    infinite = True
                    break
                ratio = Fraction(0)
            else:
                ratio = (miss / population) / budget
            multiplier = max(multiplier, ratio)
        required.append(None if infinite else multiplier)
    return tuple(required)


def _mask_costs(
    masks: np.ndarray,
    controller_costs: Sequence[float],
) -> tuple[Fraction, ...]:
    costs = np.asarray(controller_costs, dtype=float)
    if costs.ndim != 1 or not len(costs):
        raise ValueError("controller_costs must be a nonempty vector")
    if not np.all(np.isfinite(costs)) or np.any(costs <= 0):
        raise ValueError("controller_costs must be finite and positive")
    exact_costs = tuple(Fraction.from_float(float(cost)) for cost in costs)
    return tuple(
        sum(
            (
                cost
                for index, cost in enumerate(exact_costs)
                if int(mask) & (1 << index)
            ),
            Fraction(),
        )
        for mask in masks
    )


def additive_cost_budget_envelope(
    frontier: pd.DataFrame,
    *,
    controller_costs: Sequence[float],
    miss_budgets: Mapping[str, float | Fraction],
    cost_atol: float = 0.0,
    mass_atol: float = 0.0,
) -> pd.DataFrame:
    """Compute the exact lower envelope of the supplied rational frontier.

    Every nonempty mask must be present; this helper has no size-cap parameter.
    Weighted masses and float budgets use their decimal-string rational values,
    without near-integer snapping; explicit Fraction budgets are preserved.
    Controller costs use exact sums of their represented binary-rational values.
    Both tolerance parameters retain their positions but must be zero: a near
    band is not an exact optimum certificate. Float columns are displays only.
    Each row identifies its arithmetic domain and states that original-graph
    IEEE equivalence is NOT certified. This auxiliary mathematical frontier
    cannot replace the graph/evidence qualification gate.
    """

    _require_exact_budget_tolerances(cost_atol, mass_atol)
    frame, _ = _validated_frontier(
        frontier,
        miss_budgets,
        mass_atol=mass_atol,
    )
    controller_count = _complete_frontier_controller_count(frame)
    costs = np.asarray(controller_costs, dtype=float)
    if costs.shape != (controller_count,):
        raise ValueError("controller_costs must align with complete-frontier masks")
    portfolio_costs = _mask_costs(frame["mask"].to_numpy(dtype=int), costs)
    required = _exact_required_multipliers(
        frame,
        miss_budgets,
        mass_atol=mass_atol,
    )
    thresholds = tuple(sorted({value for value in required if value is not None}))
    rows: list[dict[str, object]] = []
    previous_signature: tuple[Fraction, tuple[int, ...]] | None = None
    for threshold in thresholds:
        feasible = np.fromiter(
            (value is not None and value <= threshold for value in required),
            dtype=bool,
            count=len(required),
        )
        if not bool(feasible.any()):
            continue
        optimum = min(
            cost for cost, admitted in zip(portfolio_costs, feasible) if admitted
        )
        optimal = tuple(
            sorted(
                int(mask)
                for mask, cost, admitted in zip(
                    frame["mask"], portfolio_costs, feasible
                )
                if admitted and cost == optimum
            )
        )
        signature = (optimum, optimal)
        if previous_signature == signature:
            continue
        rows.append(
            {
                "lambda_lower": float(threshold),
                "lambda_exact": str(threshold),
                "optimum_cost": float(optimum),
                "optimal_class_size": len(optimal),
                "optimal_masks": optimal,
                "representative_mask": optimal[0],
                "minimum_size_in_tie_class": int(
                    frame.loc[frame["mask"].isin(optimal), "size"].min()
                ),
                "maximum_size_in_tie_class": int(
                    frame.loc[frame["mask"].isin(optimal), "size"].max()
                ),
                "feasible_portfolios": int(feasible.sum()),
                "arithmetic_domain": _BUDGET_ARITHMETIC_DOMAIN,
                "graph_ieee_equivalence_certified": False,
            }
        )
        previous_signature = signature
    if not rows:
        raise PortfolioInfeasible("no finite budget multiplier admits a portfolio")
    for index, row in enumerate(rows):
        row["lambda_upper"] = (
            float(rows[index + 1]["lambda_lower"])
            if index + 1 < len(rows)
            else float("inf")
        )
        row["lambda_upper_exact"] = (
            str(rows[index + 1]["lambda_exact"])
            if index + 1 < len(rows)
            else "infinity"
        )
    columns = [
        "lambda_lower",
        "lambda_exact",
        "lambda_upper",
        "lambda_upper_exact",
        "optimum_cost",
        "optimal_class_size",
        "optimal_masks",
        "representative_mask",
        "minimum_size_in_tie_class",
        "maximum_size_in_tie_class",
        "feasible_portfolios",
        "arithmetic_domain",
        "graph_ieee_equivalence_certified",
    ]
    return pd.DataFrame(rows).loc[:, columns]


def certify_additive_cost_tightening(
    frontier: pd.DataFrame,
    *,
    selected_mask: int,
    controller_costs: Sequence[float],
    miss_budgets: Mapping[str, float | Fraction],
    reference_multiplier: float | Fraction = 1.0,
    cost_atol: float = 0.0,
    mass_atol: float = 0.0,
) -> AdditiveCostTighteningCertificate:
    """Verify rational-frontier optimality over [lambda_S, reference].

    Feasibility is monotone in the common multiplier.  Consequently, an
    optimum at ``reference`` remains optimal throughout the interval beginning
    at its own exact feasibility threshold.  The certificate verifies this
    fact against every nonempty mask, with exact binary-rational cost sums.
    Masses and float budgets/reference use decimal-string rationals; explicit
    Fraction budgets/reference retain their values. No mass is snapped, and
    both tolerance parameters must be zero. The returned domain explicitly
    excludes original-graph IEEE/evidence-gate equivalence. This helper has no
    size-cap parameter; graph qualification uses its own registered contract.
    """

    _require_exact_budget_tolerances(cost_atol, mass_atol)
    frame, _ = _validated_frontier(
        frontier,
        miss_budgets,
        mass_atol=mass_atol,
    )
    controller_count = _complete_frontier_controller_count(frame)
    costs = np.asarray(controller_costs, dtype=float)
    if costs.shape != (controller_count,):
        raise ValueError("controller_costs must align with complete-frontier masks")
    reference = float(reference_multiplier)
    if not np.isfinite(reference) or reference < 0:
        raise ValueError("reference_multiplier must be finite and nonnegative")
    selected_rows = np.flatnonzero(
        frame["mask"].to_numpy(dtype=int) == int(selected_mask)
    )
    if len(selected_rows) != 1:
        raise ValueError("selected mask must appear exactly once in the frontier")
    required = _exact_required_multipliers(
        frame,
        miss_budgets,
        mass_atol=mass_atol,
    )
    selected_required = required[int(selected_rows[0])]
    if selected_required is None:
        raise PortfolioInfeasible(
            "selected portfolio has no finite feasibility threshold"
        )
    reference_exact = (
        reference_multiplier
        if isinstance(reference_multiplier, Fraction)
        else Fraction(str(reference_multiplier))
    )
    portfolio_costs = _mask_costs(frame["mask"].to_numpy(dtype=int), costs)
    selected_cost = portfolio_costs[int(selected_rows[0])]
    lower_cost = np.fromiter(
        (cost < selected_cost for cost in portfolio_costs),
        dtype=bool,
        count=len(portfolio_costs),
    )
    feasible_at_reference = np.fromiter(
        (value is not None and value <= reference_exact for value in required),
        dtype=bool,
        count=len(required),
    )
    competitors = tuple(
        sorted(frame.loc[lower_cost & feasible_at_reference, "mask"].astype(int))
    )
    selected_feasible = selected_required <= reference_exact
    return AdditiveCostTighteningCertificate(
        selected_mask=int(selected_mask),
        selected_cost=float(selected_cost),
        selected_required_multiplier=float(selected_required),
        selected_required_multiplier_exact=str(selected_required),
        reference_multiplier=reference,
        reference_multiplier_exact=str(reference_exact),
        selected_feasible_at_reference=selected_feasible,
        optimal_throughout_tightening_interval=bool(
            selected_feasible and not competitors
        ),
        competitor_masks_with_lower_cost=competitors,
    )


def build_frontier_certificate(
    frontier: pd.DataFrame,
    *,
    selected_mask: int,
    miss_budgets: Mapping[str, float],
) -> FrontierCertificate:
    """Certify feasibility, primary optimality, and budget stability.

    Primary optimality uses unit controller-evaluation cost, so all feasible
    portfolios of minimum cardinality form the primary optimum class.  The
    displayed tightening/relaxation factors are analytic floating ratios, not
    proofs that a rounded endpoint passes the graph's IEEE gate. Feasibility
    and primary ties use native floored allowances directly, without epsilon.
    Any supplied weighted columns must equal count columns exactly.
    """

    frame, budgets = _validated_unit_frontier(frontier, miss_budgets)
    selected_rows = frame[frame["mask"].eq(int(selected_mask))]
    if len(selected_rows) != 1:
        raise ValueError("selected mask must appear exactly once in the frontier")
    required = _required_multiplier(frame, budgets)
    admitted = np.ones(len(frame), dtype=bool)
    for group, budget in budgets.items():
        admitted &= frame[f"{group}_miss_n"].to_numpy() <= _unit_frontier_allowances(
            frame, group, budget
        )
    feasible = frame.loc[admitted].copy()
    if feasible.empty:
        raise PortfolioInfeasible("no frontier portfolio satisfies the miss budgets")
    minimum_size = int(feasible["size"].min())
    optimal_masks = tuple(
        sorted(feasible.loc[feasible["size"].eq(minimum_size), "mask"].astype(int))
    )
    selected = selected_rows.iloc[0]
    selected_position = int(selected_rows.index[0])
    selected_required = float(required[frame.index.get_loc(selected_position)])

    group_misses: dict[str, int] = {}
    group_populations: dict[str, int] = {}
    group_allowed: dict[str, int] = {}
    group_slack: dict[str, int] = {}
    for group, budget in budgets.items():
        miss = int(selected[f"{group}_miss_n"])
        population = int(selected[f"{group}_n"])
        allowed = int(_unit_frontier_allowances(frame, group, budget)[0])
        group_misses[group] = miss
        group_populations[group] = population
        group_allowed[group] = allowed
        group_slack[group] = allowed - miss

    singletons = frame[frame["size"].eq(1)]
    singleton_relaxation = (
        float(np.min(required[frame["size"].to_numpy() == 1]))
        if not singletons.empty
        else float("inf")
    )
    return FrontierCertificate(
        selected_mask=int(selected_mask),
        selected_size=int(selected["size"]),
        minimum_size=minimum_size,
        optimal_masks=optimal_masks,
        selected_in_optimal_class=int(selected_mask) in optimal_masks,
        group_miss_counts=group_misses,
        group_population_counts=group_populations,
        group_allowed_miss_counts=group_allowed,
        group_slack_counts=group_slack,
        selected_uniform_tightening_factor=selected_required,
        singleton_uniform_relaxation_factor=singleton_relaxation,
    )


def uniform_budget_stability(
    frontier: pd.DataFrame,
    *,
    selected_mask: int,
    miss_budgets: Mapping[str, float],
    multipliers: Sequence[float],
) -> pd.DataFrame:
    """Re-solve a unit-mass/unit-cost frontier at specified IEEE multipliers.

    Scale each budget in IEEE arithmetic, then floor its population allowance;
    a scaled fraction above one is vacuous. Weighted columns must match counts.
    """

    frame, budgets = _validated_unit_frontier(frontier, miss_budgets)
    values = tuple(float(value) for value in multipliers)
    if not values or any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("budget multipliers must be finite and nonnegative")
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError("budget multipliers must be strictly increasing")
    rows: list[dict[str, object]] = []
    for multiplier in values:
        admitted = np.ones(len(frame), dtype=bool)
        for group, budget in budgets.items():
            admitted &= frame[
                f"{group}_miss_n"
            ].to_numpy() <= _unit_frontier_allowances(frame, group, budget, multiplier)
        feasible = frame.loc[admitted]
        if feasible.empty:
            rows.append(
                {
                    "budget_multiplier": multiplier,
                    "minimum_size": np.nan,
                    "optimal_class_size": 0,
                    "selected_in_optimal_class": False,
                    "feasible_portfolios": 0,
                    "optimal_masks": "",
                }
            )
            continue
        minimum_size = int(feasible["size"].min())
        optimal = tuple(
            sorted(feasible.loc[feasible["size"].eq(minimum_size), "mask"].astype(int))
        )
        rows.append(
            {
                "budget_multiplier": multiplier,
                "minimum_size": minimum_size,
                "optimal_class_size": len(optimal),
                "selected_in_optimal_class": int(selected_mask) in optimal,
                "feasible_portfolios": int(len(feasible)),
                "optimal_masks": ",".join(str(mask) for mask in optimal),
            }
        )
    return pd.DataFrame(rows)


def rescue_witness_certificate(
    pair_mechanisms: pd.DataFrame,
    *,
    pair_mask: int,
    group: str,
) -> dict[str, object]:
    """Return unique-rescue witnesses for one development-selected pair."""

    required = {
        "pair_mask",
        "group",
        "controller_i",
        "controller_j",
        "population_count",
        "failure_i_count",
        "failure_j_count",
        "common_failure_count",
        "i_rescues_j_count",
        "j_rescues_i_count",
    }
    missing = sorted(required - set(pair_mechanisms.columns))
    if missing:
        raise ValueError(f"pair mechanism table is missing: {', '.join(missing)}")
    rows = pair_mechanisms[
        pair_mechanisms["pair_mask"].astype(int).eq(int(pair_mask))
        & pair_mechanisms["group"].astype(str).eq(str(group))
    ]
    if len(rows) != 1:
        raise ValueError("pair and group must identify exactly one mechanism row")
    row = rows.iloc[0]
    return {
        "pair_mask": int(pair_mask),
        "group": str(group),
        "controller_i": str(row["controller_i"]),
        "controller_j": str(row["controller_j"]),
        "population_count": int(row["population_count"]),
        "member_i_unique_rescue_count": int(row["i_rescues_j_count"]),
        "member_j_unique_rescue_count": int(row["j_rescues_i_count"]),
        "misses_if_i_removed": int(row["failure_j_count"]),
        "misses_if_j_removed": int(row["failure_i_count"]),
        "pair_common_miss_count": int(row["common_failure_count"]),
    }


def _solve_sparse_task_milp(
    graph: FailureHypergraph,
    *,
    miss_budgets: Mapping[str, float],
    max_size: int | None,
) -> tuple[tuple[int, ...], float, int]:
    """Solve the task-indexed formulation with the public exact tie contract."""

    budgets = {str(group): float(value) for group, value in miss_budgets.items()}
    if set(budgets) != set(graph.groups):
        raise ValueError("miss budgets must contain every graph group exactly once")
    if any(
        not np.isfinite(value) or value < 0.0 or value > 1.0
        for value in budgets.values()
    ):
        raise ValueError("miss budgets must lie in [0, 1]")
    n, k = graph.n_tasks, graph.n_controllers
    integer_weight_contract = _uses_integer_weight_contract(graph.task_weights)
    cover = hstack(
        (csr_matrix(graph.good.astype(float)), eye(n, format="csr")),
        format="csr",
    )
    group_rows = []
    group_upper = []
    for group in sorted(budgets):
        membership = graph.groups[group].astype(float) * graph.task_weights
        group_rows.append(
            hstack((csr_matrix((1, k)), csr_matrix(membership.reshape(1, -1))))
        )
        group_upper.append(
            _weighted_miss_allowance(
                graph.task_weights[graph.groups[group]],
                budgets[group],
                integer_weight_contract=integer_weight_contract,
            )
        )
    group_matrix = vstack(group_rows, format="csr")
    constraints: list[LinearConstraint] = [
        LinearConstraint(cover, np.ones(n), np.full(n, np.inf)),
        LinearConstraint(
            group_matrix,
            np.full(len(group_upper), -np.inf),
            np.asarray(group_upper, dtype=float),
        ),
    ]
    if max_size is not None:
        if max_size <= 0 or max_size > k:
            raise ValueError("max_size must lie within the controller library")
        size = csr_matrix(
            (
                np.ones(k, dtype=float),
                (np.zeros(k, dtype=int), np.arange(k, dtype=int)),
            ),
            shape=(1, k + n),
        )
        constraints.append(
            LinearConstraint(size, np.array([-np.inf]), np.array([float(max_size)]))
        )
    integrality = np.concatenate(
        [np.ones(k, dtype=np.uint8), np.zeros(n, dtype=np.uint8)]
    )
    bounds = Bounds(np.zeros(k + n), np.ones(k + n))
    primary_objective = np.concatenate(
        [graph.controller_costs, np.zeros(n, dtype=float)]
    )
    options = {"presolve": True}
    start = perf_counter()
    primary = milp(
        c=primary_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options=options,
    )
    solver_calls = 1
    if not bool(primary.success) or primary.x is None:
        raise PortfolioInfeasible(f"sparse task MILP failed: {primary.message}")
    primary_members = tuple(
        int(index) for index in np.flatnonzero(primary.x[:k] >= 0.5)
    )
    optimum = (
        float(graph.controller_costs[list(primary_members)].sum())
        if primary_members
        else 0.0
    )
    cost_row = np.zeros((1, k + n), dtype=float)
    cost_row[0, :k] = graph.controller_costs
    tie_constraints = [
        *constraints,
        LinearConstraint(
            csr_matrix(cost_row),
            np.array([optimum], dtype=float),
            np.array([optimum], dtype=float),
        ),
    ]
    members = primary_members
    if k <= 20:
        tie_objective = np.concatenate(
            [
                np.exp2(np.arange(k, dtype=float)),
                np.full(n, 1.0 / (n + 1), dtype=float),
            ]
        )
        tie = milp(
            c=tie_objective,
            integrality=integrality,
            bounds=bounds,
            constraints=tie_constraints,
            options=options,
        )
        solver_calls += 1
        if bool(tie.success) and tie.x is not None:
            candidate = tuple(int(index) for index in np.flatnonzero(tie.x[:k] >= 0.5))
            candidate_cost = (
                float(graph.controller_costs[list(candidate)].sum())
                if candidate
                else 0.0
            )
            if np.isclose(candidate_cost, optimum, rtol=0.0, atol=1e-8):
                members = candidate
    else:
        locked_constraints = list(tie_constraints)
        locked_solution = None
        for controller in reversed(range(k)):
            lex_objective = np.zeros(k + n, dtype=float)
            lex_objective[controller] = 1.0
            lex = milp(
                c=lex_objective,
                integrality=integrality,
                bounds=bounds,
                constraints=locked_constraints,
                options=options,
            )
            solver_calls += 1
            if not bool(lex.success) or lex.x is None:
                locked_solution = None
                break
            locked_value = float(lex.x[controller] >= 0.5)
            lock_row = np.zeros((1, k + n), dtype=float)
            lock_row[0, controller] = 1.0
            locked_constraints.append(
                LinearConstraint(
                    csr_matrix(lock_row),
                    np.array([locked_value], dtype=float),
                    np.array([locked_value], dtype=float),
                )
            )
            locked_solution = lex.x
        if locked_solution is not None:
            candidate = tuple(
                int(index) for index in np.flatnonzero(locked_solution[:k] >= 0.5)
            )
            candidate_cost = (
                float(graph.controller_costs[list(candidate)].sum())
                if candidate
                else 0.0
            )
            if np.isclose(candidate_cost, optimum, rtol=0.0, atol=1e-8):
                members = candidate
    elapsed = perf_counter() - start
    return members, elapsed, solver_calls


def repeated_signature_ablation_instance(
    *, seed: int = 20_260_828
) -> FailureHypergraph:
    """Return the frozen 9,600-task/30-signature ablation instance.

    Thirty nonzero five-bit signatures are repeated 320 times, embedded in a
    12-controller library, and deterministically row-shuffled.  Five active
    controller columns are jointly necessary for zero-miss coverage; the
    remaining columns provide a stable unused-cost check.
    """

    signature_ids = np.arange(1, 31, dtype=np.uint16)
    signatures = np.zeros((30, 12), dtype=bool)
    signatures[:, :5] = (
        (signature_ids[:, None] >> np.arange(5, dtype=np.uint16)) & 1
    ).astype(bool)
    good = np.repeat(signatures, 320, axis=0)
    repeated_ids = np.repeat(signature_ids, 320)
    order = np.random.default_rng(int(seed)).permutation(len(good))
    good = good[order]
    repeated_ids = repeated_ids[order]
    return FailureHypergraph(
        good=good,
        controller_names=tuple(f"c{index:02d}" for index in range(12)),
        controller_costs=np.array([1.0] * 5 + [2.0] * 7, dtype=float),
        groups={
            "pooled": np.ones(len(good), dtype=bool),
            "even_signature": (repeated_ids % 2) == 0,
            "upper_signature": repeated_ids >= 16,
        },
        task_weights=np.ones(len(good), dtype=float),
    )


def benchmark_task_signature_formulations(
    graph: FailureHypergraph,
    *,
    miss_budgets: Mapping[str, float],
    max_size: int | None = None,
    repetitions: int = 3,
) -> pd.DataFrame:
    """Benchmark equivalent task and lossless-signature MILP formulations.

    Runtime records implementation timing in the current environment.  Model
    dimensions and exact solution equivalence are the portable comparisons.
    """

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    table = compress_failure_signatures(graph)
    task_times: list[float] = []
    task_solver_calls: list[int] = []
    signature_times: list[float] = []
    task_members: tuple[int, ...] | None = None
    signature_result = None
    run_audits: list[dict[str, object]] = []
    for repetition in range(int(repetitions)):
        task_members, elapsed, solver_calls = _solve_sparse_task_milp(
            graph, miss_budgets=miss_budgets, max_size=max_size
        )
        task_times.append(elapsed)
        task_solver_calls.append(solver_calls)
        start = perf_counter()
        signature_result = solve_signature_portfolio(
            table, miss_budgets=miss_budgets, max_size=max_size
        )
        signature_times.append(perf_counter() - start)
        task_run = evaluate_task_portfolio(graph, task_members)
        signature_run = evaluate_signature_portfolio(table, signature_result.members)
        audit = {
            "repetition": repetition,
            "task_members": list(task_run.members),
            "signature_members": list(signature_run.members),
            "members_equal": task_run.members == signature_run.members,
            "miss_vector_equal": bool(
                np.array_equal(
                    graph.common_failure(task_run.members),
                    graph.common_failure(signature_run.members),
                )
            ),
            "cost_equal": bool(
                np.isclose(task_run.cost, signature_run.cost, rtol=0.0, atol=1e-9)
            ),
            "group_misses_equal": (task_run.group_misses == signature_run.group_misses),
            "group_weighted_misses_equal": bool(
                task_run.group_weighted_misses.keys()
                == signature_run.group_weighted_misses.keys()
                and all(
                    np.isclose(
                        task_run.group_weighted_misses[group],
                        signature_run.group_weighted_misses[group],
                        rtol=0.0,
                        atol=1e-9,
                    )
                    for group in signature_run.group_weighted_misses
                )
            ),
        }
        audit["solution_equal"] = bool(
            audit["members_equal"]
            and audit["miss_vector_equal"]
            and audit["cost_equal"]
            and audit["group_misses_equal"]
            and audit["group_weighted_misses_equal"]
        )
        if not bool(audit["solution_equal"]):
            raise RuntimeError(
                "task/signature formulations diverged at repetition "
                f"{repetition}: {audit}"
            )
        run_audits.append(audit)
    assert task_members is not None and signature_result is not None
    task_eval = evaluate_task_portfolio(graph, task_members)
    signature_eval = evaluate_signature_portfolio(table, signature_result.members)
    task_miss_vector = graph.common_failure(task_members)
    signature_miss_vector = graph.common_failure(signature_result.members)
    members_equal = task_eval.members == signature_eval.members
    miss_vector_equal = bool(np.array_equal(task_miss_vector, signature_miss_vector))
    cost_equal = bool(np.isclose(task_eval.cost, signature_eval.cost))
    group_misses_equal = task_eval.group_misses == signature_eval.group_misses
    group_weighted_misses_equal = bool(
        task_eval.group_weighted_misses.keys()
        == signature_eval.group_weighted_misses.keys()
        and all(
            np.isclose(task_eval.group_weighted_misses[group], value)
            for group, value in signature_eval.group_weighted_misses.items()
        )
    )
    equivalent = bool(
        members_equal
        and miss_vector_equal
        and cost_equal
        and group_misses_equal
        and group_weighted_misses_equal
    )
    group_count = len(miss_budgets)
    size_constraints = 1 if max_size is not None else 0
    task_nnz = int(graph.good.sum()) + graph.n_tasks
    task_nnz += int(sum(graph.groups[group].sum() for group in miss_budgets))
    task_nnz += graph.n_controllers if max_size is not None else 0
    signature_nnz = int(table.signatures.sum()) + table.n_signatures
    signature_nnz += int(
        sum(np.count_nonzero(table.group_weights[group]) for group in miss_budgets)
    )
    signature_nnz += table.n_controllers if max_size is not None else 0
    common = {
        "n_tasks": graph.n_tasks,
        "n_signatures": table.n_signatures,
        "n_controllers": graph.n_controllers,
        "group_constraints": group_count,
        "members_equal": members_equal,
        "miss_vector_equal": miss_vector_equal,
        "cost_equal": cost_equal,
        "group_misses_equal": group_misses_equal,
        "group_weighted_misses_equal": group_weighted_misses_equal,
        "solution_equal": equivalent,
        "repetitions": int(repetitions),
        "all_repetitions_equal": bool(
            all(bool(audit["solution_equal"]) for audit in run_audits)
        ),
        "per_run_equivalence_json": json.dumps(
            run_audits, sort_keys=True, separators=(",", ":")
        ),
        "runtime_role": "environment-recorded implementation timing",
    }
    return pd.DataFrame(
        [
            {
                **common,
                "formulation": "task_sparse_milp",
                "controller_binaries": graph.n_controllers,
                "continuous_auxiliaries": graph.n_tasks,
                "total_variables": graph.n_controllers + graph.n_tasks,
                "coverage_constraints": graph.n_tasks,
                "total_constraints": graph.n_tasks + group_count + size_constraints,
                "constraint_nonzeros": task_nnz,
                "selected_size": len(task_eval.members),
                "objective": task_eval.cost,
                "median_seconds": float(np.median(task_times)),
                "minimum_seconds": float(np.min(task_times)),
                "solver_calls_per_repetition": int(task_solver_calls[-1]),
            },
            {
                **common,
                "formulation": "signature_milp",
                "controller_binaries": table.n_controllers,
                "continuous_auxiliaries": table.n_signatures,
                "total_variables": table.n_controllers + table.n_signatures,
                "coverage_constraints": table.n_signatures,
                "total_constraints": table.n_signatures
                + group_count
                + size_constraints,
                "constraint_nonzeros": signature_nnz,
                "selected_size": len(signature_eval.members),
                "objective": signature_eval.cost,
                "median_seconds": float(np.median(signature_times)),
                "minimum_seconds": float(np.min(signature_times)),
                "solver_calls_per_repetition": (
                    2 if table.n_controllers <= 20 else 1 + table.n_controllers
                ),
            },
        ]
    )
