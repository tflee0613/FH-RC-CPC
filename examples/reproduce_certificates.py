#!/usr/bin/env python3
"""Reproduce representative cost-box and whole-row outcome certificates."""

from __future__ import annotations

import json

import numpy as np

from fh_rc_cpc.certificates import (
    build_signature_zeta_frontier,
    certify_cost_box_structure,
    certify_outcome_stability,
    verify_row_change_witness,
)
from fh_rc_cpc.qualification import FailureHypergraph, compress_failure_signatures


def make_graph(good, costs) -> FailureHypergraph:
    values = np.asarray(good, dtype=bool)
    tasks, controllers = values.shape
    return FailureHypergraph(
        good=values,
        controller_names=tuple(f"c{index}" for index in range(controllers)),
        controller_costs=np.asarray(costs, dtype=float),
        groups={"pooled": np.ones(tasks, dtype=bool)},
        task_weights=np.ones(tasks),
    )


def main() -> None:
    cost_graph = make_graph([[1, 0, 1], [0, 1, 1]], [1, 1, 3])
    frontier = build_signature_zeta_frontier(
        compress_failure_signatures(cost_graph)
    )
    cost = certify_cost_box_structure(
        frontier,
        miss_budgets={"pooled": 0.0},
        target_masks=(3,),
        cost_lower=(1.0, 1.0, 3.0),
        cost_upper=(1.0, 1.0, 4.0),
    )

    outcome_graph = make_graph(
        [[1, 0, 0]] * 5 + [[0, 1, 0]] * 3 + [[0, 0, 1]] * 3,
        [1, 1, 1],
    )
    budgets = {"pooled": 4 / 11}
    outcome = certify_outcome_stability(
        outcome_graph,
        incumbent=(0, 1),
        miss_budgets=budgets,
        max_size=2,
    )
    if outcome.witness is None or not verify_row_change_witness(
        outcome_graph, outcome.witness, miss_budgets=budgets
    ):
        raise RuntimeError("outcome-change witness did not attain its boundary")

    print(
        json.dumps(
            {
                "status": "pass",
                "cost_box_preserves_target": (
                    cost.all_optima_in_target_over_full_box
                ),
                "outcome_change_distance": outcome.distance,
                "outcome_stability_radius": outcome.radius,
                "witness_rows": list(outcome.witness.rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
