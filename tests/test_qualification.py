"""Focused checks for the qualification, reduction and selection layer."""

from __future__ import annotations

import numpy as np

from fh_rc_cpc.qualification import (
    FailureHypergraph,
    compress_failure_signatures,
    evaluate_signature_portfolio,
    evaluate_task_portfolio,
    greedy_signature_portfolio,
    quotient_controller_behavior,
    solve_signature_portfolio,
    solve_signature_portfolio_by_enumeration,
)


def small_registry() -> FailureHypergraph:
    return FailureHypergraph(
        good=np.array(
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=bool,
        ),
        controller_names=("a", "b", "c", "d"),
        controller_costs=np.array([1.0, 1.0, 3.0, 4.0]),
        groups={
            "pooled": np.ones(6, dtype=bool),
            "priority": np.array([1, 1, 1, 1, 0, 0], dtype=bool),
        },
        task_weights=np.ones(6),
    )


def test_task_signature_and_enumeration_results_agree() -> None:
    graph = small_registry()
    table = compress_failure_signatures(graph)
    budgets = {"pooled": 1 / 3, "priority": 0.0}
    exact = solve_signature_portfolio(table, miss_budgets=budgets)
    enumerated = solve_signature_portfolio_by_enumeration(
        table, miss_budgets=budgets
    )
    greedy = greedy_signature_portfolio(table, miss_budgets=budgets)
    assert exact.members == enumerated.members == (0, 1)
    assert exact.objective == enumerated.objective == greedy.objective == 2.0
    assert evaluate_task_portfolio(graph, exact.members) == (
        evaluate_signature_portfolio(table, exact.members)
    )


def test_behavior_quotient_preserves_coverage_and_minimum_cost_lifts() -> None:
    graph = FailureHypergraph(
        good=np.array([[1, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=bool),
        controller_names=("a", "a_alias", "b"),
        controller_costs=np.array([1.0, 2.0, 1.0]),
        groups={"pooled": np.ones(3, dtype=bool)},
        task_weights=np.ones(3),
    )
    quotient = quotient_controller_behavior(graph)
    assert quotient.classes == ((0, 1), (2,))
    assert quotient.representatives == (0, 2)
    assert quotient.map_members(("a_alias", "b")) == (0, 1)
    assert quotient.lift_members((0,), mode="minimum_cost") == ((0,),)
    assert quotient.lift_members((0,), mode="all_equivalent") == ((0,), (1,))
    for original in ((0,), (1,), (2,), (0, 2), (1, 2)):
        reduced = quotient.map_members(original)
        assert np.array_equal(
            graph.common_failure(original), quotient.graph.common_failure(reduced)
        )
