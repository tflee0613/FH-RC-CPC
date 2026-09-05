"""Deterministic nonintegral stress family for controller-library compression."""

from __future__ import annotations

import hashlib
import time

import numpy as np
from scipy.optimize import linprog

from ..qualification.behavior_quotient import quotient_controller_behavior
from ..qualification.failure_hypergraph import FailureHypergraph
from ..qualification.general_portfolio import solve_group_constrained_portfolio
from ..qualification.signature_compression import compress_failure_signatures
from ..qualification.signature_solver import (
    greedy_signature_portfolio,
    solve_signature_portfolio,
)


def fractional_triangle_instance(
    cycle_count: int,
    repetitions: int,
    *,
    duplicate_factor: int = 2,
) -> FailureHypergraph:
    """Build disjoint triangle-cover blocks with duplicate nominal controllers.

    Each triangle is a vertex-cover instance: the integer optimum selects two
    of three behavior classes, whereas the LP assigns one half to every class.
    Repeated edges exercise task-signature compression, and duplicate controller
    columns exercise the exact behavior quotient without changing the optimum.
    """

    cycles = int(cycle_count)
    repeats = int(repetitions)
    duplicates = int(duplicate_factor)
    if cycles < 2 or repeats <= 0 or duplicates <= 0:
        raise ValueError(
            "cycle_count >= 2 and positive repetitions/duplicates required"
        )
    behavior_count = 3 * cycles
    controller_count = behavior_count * duplicates
    edge_count = 3 * cycles
    task_count = edge_count * repeats
    good = np.zeros((task_count, controller_count), dtype=bool)
    block_ids = np.zeros(task_count, dtype=int)
    row = 0
    for block in range(cycles):
        vertices = (3 * block, 3 * block + 1, 3 * block + 2)
        for left, right in (
            (vertices[0], vertices[1]),
            (vertices[1], vertices[2]),
            (vertices[2], vertices[0]),
        ):
            columns = [
                vertex * duplicates + copy
                for vertex in (left, right)
                for copy in range(duplicates)
            ]
            for _ in range(repeats):
                good[row, columns] = True
                block_ids[row] = block
                row += 1
    names = tuple(
        f"b{behavior // 3:03d}_v{behavior % 3}_d{copy}"
        for behavior in range(behavior_count)
        for copy in range(duplicates)
    )
    return FailureHypergraph(
        good=good,
        controller_names=names,
        controller_costs=np.ones(controller_count, dtype=float),
        groups={
            "pooled": np.ones(task_count, dtype=bool),
            "even_blocks": (block_ids % 2) == 0,
            "odd_blocks": (block_ids % 2) == 1,
        },
        task_weights=np.ones(task_count, dtype=float),
    )


def graph_sha256(graph: FailureHypergraph) -> str:
    digest = hashlib.sha256()
    digest.update(graph.good.tobytes(order="C"))
    digest.update(graph.controller_costs.astype("<f8").tobytes(order="C"))
    for group in sorted(graph.groups):
        digest.update(group.encode("utf-8"))
        digest.update(graph.groups[group].tobytes(order="C"))
    return digest.hexdigest()


def benchmark_fractional_triangle_instance(
    cycle_count: int,
    repetitions: int,
) -> dict[str, object]:
    """Compare task, signature, quotient, LP, rounding, and greedy solutions."""

    graph = fractional_triangle_instance(cycle_count, repetitions)
    budgets = {group: 0.0 for group in graph.groups}
    analytic_optimum = 2 * int(cycle_count)

    start = time.perf_counter()
    task_exact = solve_group_constrained_portfolio(
        graph.good,
        costs=graph.controller_costs,
        groups=graph.groups,
        miss_budgets=budgets,
        weights=graph.task_weights,
    )
    task_seconds = time.perf_counter() - start

    start = time.perf_counter()
    task_signatures = compress_failure_signatures(graph)
    signature_exact = solve_signature_portfolio(
        task_signatures,
        miss_budgets=budgets,
        deterministic_tie_break=False,
    )
    signature_seconds = time.perf_counter() - start

    start = time.perf_counter()
    quotient = quotient_controller_behavior(graph)
    quotient_signatures = compress_failure_signatures(quotient.graph)
    quotient_exact = solve_signature_portfolio(
        quotient_signatures,
        miss_budgets=budgets,
        compute_lp_relaxation=True,
        deterministic_tie_break=False,
    )
    quotient_seconds = time.perf_counter() - start

    lp = linprog(
        c=quotient.graph.controller_costs,
        A_ub=-quotient.graph.good.astype(float),
        b_ub=-np.ones(quotient.graph.n_tasks, dtype=float),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not lp.success or lp.x is None or lp.fun is None:
        raise RuntimeError(f"fractional stress LP failed: {lp.message}")
    rounded = tuple(int(index) for index in np.flatnonzero(lp.x >= 0.5 - 1e-9))
    rounded_feasible = bool(quotient.graph.good[:, list(rounded)].any(axis=1).all())
    greedy = greedy_signature_portfolio(quotient_signatures, miss_budgets=budgets)

    objectives = (
        float(task_exact.objective),
        float(signature_exact.objective),
        float(quotient_exact.objective),
    )
    if any(not np.isclose(value, analytic_optimum) for value in objectives):
        raise RuntimeError("exact formulations disagree with the analytic optimum")
    expected_lp = 1.5 * int(cycle_count)
    if not np.isclose(float(lp.fun), expected_lp, atol=1e-8, rtol=0.0):
        raise RuntimeError("LP relaxation disagrees with the analytic triangle bound")
    if not rounded_feasible:
        raise RuntimeError("registered half-threshold rounding is infeasible")

    return {
        "version": "paper-a-fractional-triangle-stress-v1",
        "cycle_count": int(cycle_count),
        "repetitions": int(repetitions),
        "task_count": graph.n_tasks,
        "nominal_controller_count": graph.n_controllers,
        "behavior_class_count": quotient.graph.n_controllers,
        "task_signature_count": task_signatures.n_signatures,
        "quotient_signature_count": quotient_signatures.n_signatures,
        "analytic_integer_optimum": analytic_optimum,
        "task_milp_objective": float(task_exact.objective),
        "signature_milp_objective": float(signature_exact.objective),
        "quotient_milp_objective": float(quotient_exact.objective),
        "lp_objective": float(lp.fun),
        "lp_relative_gap": float((analytic_optimum - lp.fun) / analytic_optimum),
        "lp_fractional_variable_count": int(
            np.count_nonzero((lp.x > 1e-9) & (lp.x < 1 - 1e-9))
        ),
        "half_threshold_rounded_size": len(rounded),
        "half_threshold_rounding_feasible": rounded_feasible,
        "half_threshold_rounding_relative_gap": float(
            (len(rounded) - analytic_optimum) / analytic_optimum
        ),
        "greedy_objective": float(greedy.objective),
        "greedy_relative_gap": float(
            (greedy.objective - analytic_optimum) / analytic_optimum
        ),
        "quotient_mip_node_count": quotient_exact.mip_node_count,
        "graph_sha256": graph_sha256(graph),
        "task_milp_seconds": task_seconds,
        "signature_milp_seconds": signature_seconds,
        "quotient_milp_seconds": quotient_seconds,
        "runtime_role": "implementation diagnostic only",
    }
