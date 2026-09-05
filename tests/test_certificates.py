"""Focused checks for the exact certificate layer."""

from __future__ import annotations

import numpy as np

from fh_rc_cpc.certificates import (
    ScopeContract,
    build_signature_zeta_frontier,
    certify_continuous_auxiliary_projection,
    certify_cost_box_structure,
    certify_greedy_submodular_cover,
    certify_outcome_stability,
    compose_scopes,
    qualification_with_row_reserve,
    verify_row_change_witness,
)
from fh_rc_cpc.qualification import FailureHypergraph, compress_failure_signatures


def graph(good, *, costs=None) -> FailureHypergraph:
    values = np.asarray(good, dtype=bool)
    tasks, controllers = values.shape
    return FailureHypergraph(
        good=values,
        controller_names=tuple(f"c{index}" for index in range(controllers)),
        controller_costs=np.asarray(
            np.ones(controllers) if costs is None else costs, dtype=float
        ),
        groups={"pooled": np.ones(tasks, dtype=bool)},
        task_weights=np.ones(tasks),
    )


def test_projection_greedy_and_cost_box_certificates() -> None:
    example = graph([[1, 0, 1], [0, 1, 1]], costs=[1, 1, 3])
    table = compress_failure_signatures(example)
    budgets = {"pooled": 0.0}
    frontier = build_signature_zeta_frontier(table)
    projection = certify_continuous_auxiliary_projection(
        table, members=(0, 1), miss_budgets=budgets
    )
    greedy = certify_greedy_submodular_cover(
        table, miss_budgets=budgets, verify_optimum=True
    )
    cost_box = certify_cost_box_structure(
        frontier,
        miss_budgets=budgets,
        target_masks=(3,),
        cost_lower=(1.0, 1.0, 3.0),
        cost_upper=(1.0, 1.0, 4.0),
    )
    assert projection.projection_exact and projection.projected_feasible
    assert greedy.conditions_satisfied and greedy.exhaustive_bound_verified
    assert cost_box.all_optima_in_target_over_full_box


def test_whole_row_outcome_distance_has_an_attaining_witness() -> None:
    example = graph([[1, 0, 0]] * 5 + [[0, 1, 0]] * 3 + [[0, 0, 1]] * 3)
    budgets = {"pooled": 4 / 11}
    certificate = certify_outcome_stability(
        example, incumbent=(0, 1), miss_budgets=budgets, max_size=2
    )
    assert certificate.distance == 2
    assert certificate.radius == 1
    assert certificate.witness is not None
    assert verify_row_change_witness(
        example, certificate.witness, miss_budgets=budgets
    )


def test_scope_composition_preserves_each_population_obligation() -> None:
    old = ScopeContract("old", graph([[1, 0, 1]]), {"pooled": 0.0})
    new = ScopeContract("new", graph([[0, 1, 0]]), {"pooled": 0.0})
    result = compose_scopes((old, new))
    assert result.minimum_cost == 2
    assert result.optimal_masks == (3, 6)
    assert all(
        all(item.passed for item in optimum.scope_certificates)
        for optimum in result.optimal_portfolios
    )


def test_row_reserve_changes_the_qualification_query() -> None:
    example = graph([[1, 0], [1, 0], [1, 0]], costs=[1, 2])
    result = qualification_with_row_reserve(
        example,
        row_reserve=1,
        miss_budgets={"pooled": 1 / 3},
        allow_empty=False,
    )
    assert result.status == "optimal"
    assert result.minimum_cost == 1
    assert tuple(item.members for item in result.optima) == ((0,),)
    assert result.optima[0].destruction.distance == 2
