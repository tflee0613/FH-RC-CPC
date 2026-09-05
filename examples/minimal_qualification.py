#!/usr/bin/env python3
"""Solve a small minimum-cost policy-retention problem."""

from __future__ import annotations

import json

import numpy as np

from fh_rc_cpc.qualification import (
    FailureHypergraph,
    compress_failure_signatures,
    evaluate_signature_portfolio,
    evaluate_task_portfolio,
    greedy_signature_portfolio,
    solve_signature_portfolio,
)


def main() -> None:
    graph = FailureHypergraph(
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
    table = compress_failure_signatures(graph)
    budgets = {"pooled": 1 / 3, "priority": 0.0}
    exact = solve_signature_portfolio(table, miss_budgets=budgets)
    greedy = greedy_signature_portfolio(table, miss_budgets=budgets)
    task_evaluation = evaluate_task_portfolio(graph, exact.members)
    signature_evaluation = evaluate_signature_portfolio(table, exact.members)
    if task_evaluation != signature_evaluation:
        raise RuntimeError("task and signature evaluations disagree")
    print(
        json.dumps(
            {
                "status": "pass",
                "tasks": graph.n_tasks,
                "signatures": table.n_signatures,
                "exact_members": [
                    graph.controller_names[index] for index in exact.members
                ],
                "exact_cost": exact.objective,
                "greedy_cost": greedy.objective,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
