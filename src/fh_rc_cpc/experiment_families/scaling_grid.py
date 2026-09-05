"""Large deterministic scaling instances with solver diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from typing import Any

import numpy as np
import scipy

from ..qualification.failure_hypergraph import FailureHypergraph
from ..qualification.general_portfolio import PortfolioInfeasible
from ..qualification.signature_compression import compress_failure_signatures
from ..qualification.signature_solver import (
    greedy_signature_portfolio,
    solve_signature_portfolio,
)

REGISTERED_SCALING_CONFIG: dict[str, Any] = {
    "version": "paper-a-scaling-diagnostics-v2",
    "controller_counts": (60, 100, 200, 500),
    "signature_targets": {"60": 1000, "100": 2000, "200": 5000, "500": 10000},
    "seeds": (7, 19, 43),
    "group_miss_budget": 0.0,
    "per_solve_time_limit_seconds": 900.0,
}


def solver_environment_record() -> dict[str, Any]:
    """Return machine-readable runtime and bundled-solver identity."""

    try:
        from scipy.optimize._highspy import _core as highs_core

        highs_version = (
            f"{highs_core.HIGHS_VERSION_MAJOR}."
            f"{highs_core.HIGHS_VERSION_MINOR}."
            f"{highs_core.HIGHS_VERSION_PATCH}"
        )
    except (ImportError, AttributeError):  # pragma: no cover - SciPy fallback
        highs_version = f"bundled-with-scipy-{scipy.__version__}"
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "highs_version": highs_version,
        "solver_interface": "scipy.optimize.milp",
        "solver_threads": "not explicitly set; HiGHS default",
    }


def diagnostic_instance(
    controller_count: int,
    signature_target: int,
    seed: int,
) -> FailureHypergraph:
    """Create one fixed, non-regenerated overlapping-group instance."""

    k = int(controller_count)
    q = int(signature_target)
    if k < 20 or q < 100 or q <= k:
        raise ValueError("diagnostic instance dimensions are too small")
    rng = np.random.default_rng(np.random.SeedSequence([k, q, int(seed), 20260826]))
    mode_count = 20
    modes = np.arange(q, dtype=int) % mode_count
    rng.shuffle(modes)
    severity = rng.random(q)
    good = np.zeros((q, k), dtype=bool)
    capability_count = 4
    for controller in range(k):
        capabilities = rng.choice(mode_count, size=capability_count, replace=False)
        base = np.isin(modes, capabilities)
        exploration = rng.random(q) < 0.04
        dropout = rng.random(q) < 0.08
        good[:, controller] = (base & ~dropout) | exploration
    core_count = min(12, max(6, k // 20))
    bit_count = int(np.ceil(np.log2(q - core_count + 1)))
    codes = np.arange(1, q - core_count + 1, dtype=np.uint64)
    bits = ((codes[:, None] >> np.arange(bit_count, dtype=np.uint64)) & 1).astype(bool)
    good[core_count:, k - bit_count :] = bits
    good[core_count:, :core_count] |= np.eye(core_count, dtype=bool)[
        np.arange(q - core_count) % core_count
    ]
    good[:core_count, :] = False
    good[np.arange(core_count), np.arange(core_count)] = True
    uncovered = np.flatnonzero(~good.any(axis=1))
    if len(uncovered):
        raise RuntimeError("constructed diagnostic library does not cover every task")
    costs = 0.9 + 0.2 * rng.random(k)
    groups = {
        "pooled": np.ones(q, dtype=bool),
        "mode_overlap_a": (modes % 3) != 0,
        "mode_overlap_b": (modes % 3) != 1,
        "mode_overlap_c": (modes % 3) != 2,
        "high_severity": severity >= 0.6,
    }
    return FailureHypergraph(
        good=good,
        controller_names=tuple(f"c{index:03d}" for index in range(k)),
        controller_costs=costs,
        groups=groups,
        task_weights=np.ones(q, dtype=float),
    )


def graph_sha256(graph: FailureHypergraph) -> str:
    digest = hashlib.sha256()
    digest.update(graph.good.tobytes(order="C"))
    digest.update(graph.controller_costs.astype("<f8").tobytes(order="C"))
    for group in sorted(graph.groups):
        digest.update(group.encode("utf-8"))
        digest.update(graph.groups[group].tobytes(order="C"))
    return digest.hexdigest()


def benchmark_diagnostic_instance(
    controller_count: int,
    seed: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if config is None:
        config = REGISTERED_SCALING_CONFIG
    target = int(config["signature_targets"][str(int(controller_count))])
    graph = diagnostic_instance(controller_count, target, seed)
    start = time.perf_counter()
    table = compress_failure_signatures(graph)
    compression_seconds = time.perf_counter() - start
    if table.n_signatures != target:
        raise RuntimeError("diagnostic signatures are unexpectedly duplicated")
    budget = float(config["group_miss_budget"])
    budgets = {name: budget for name in graph.groups}
    limit = float(config["per_solve_time_limit_seconds"])
    start = time.perf_counter()
    exact = solve_signature_portfolio(
        table,
        miss_budgets=budgets,
        presolve=True,
        time_limit=limit,
        compute_lp_relaxation=False,
        deterministic_tie_break=False,
    )
    presolve_on_seconds = time.perf_counter() - start
    with_lp = solve_signature_portfolio(
        table,
        miss_budgets=budgets,
        presolve=True,
        time_limit=limit,
        compute_lp_relaxation=True,
        deterministic_tie_break=False,
    )
    if not np.isclose(with_lp.objective, exact.objective, rtol=0.0, atol=1e-8):
        raise RuntimeError("LP diagnostic rerun changed the exact objective")
    off_status = "not_run"
    off_seconds = float("nan")
    off_nodes: int | None = None
    off_objective = float("nan")
    try:
        start = time.perf_counter()
        without_presolve = solve_signature_portfolio(
            table,
            miss_budgets=budgets,
            presolve=False,
            time_limit=limit,
            compute_lp_relaxation=False,
            deterministic_tie_break=False,
        )
        off_seconds = time.perf_counter() - start
        off_status = without_presolve.status
        off_nodes = without_presolve.mip_node_count
        off_objective = float(without_presolve.objective)
    except PortfolioInfeasible as error:
        off_seconds = time.perf_counter() - start
        off_status = f"unsolved:{str(error)[:120]}"
    start = time.perf_counter()
    greedy = greedy_signature_portfolio(table, miss_budgets=budgets)
    greedy_seconds = time.perf_counter() - start
    greedy_gap = (float(greedy.objective) - float(exact.objective)) / max(
        abs(float(exact.objective)), 1e-12
    )
    nonzero_cover = int(graph.good.sum()) + table.n_signatures
    solver_options = {
        "primary": {
            "presolve": True,
            "mip_rel_gap": 0.0,
            "time_limit_seconds": limit,
            "deterministic_tie_break": False,
        },
        "presolve_comparator": {
            "presolve": False,
            "mip_rel_gap": 0.0,
            "time_limit_seconds": limit,
            "deterministic_tie_break": False,
        },
        "lp_relaxation": {
            "presolve": True,
            "time_limit_seconds": limit,
        },
    }
    return {
        "version": str(config["version"]),
        "environment": solver_environment_record(),
        "solver_options": solver_options,
        "controller_count": graph.n_controllers,
        "seed": int(seed),
        "task_count": graph.n_tasks,
        "signature_count": table.n_signatures,
        "group_count": len(graph.groups),
        "milp_variable_count": graph.n_controllers + table.n_signatures,
        "milp_constraint_count": table.n_signatures + len(graph.groups),
        "constraint_nonzero_count": nonzero_cover
        + sum(
            int(values.astype(bool).sum()) for values in table.group_weights.values()
        ),
        "good_density": float(graph.good.mean()),
        "graph_sha256": graph_sha256(graph),
        "fixed_instance_no_regeneration": True,
        "exact_objective": float(exact.objective),
        "exact_size": len(exact.members),
        "exact_members_json": json.dumps(list(exact.members), separators=(",", ":")),
        "exact_status": exact.status,
        "exact_mip_gap": exact.mip_gap,
        "exact_mip_node_count": exact.mip_node_count,
        "exact_mip_dual_bound": exact.mip_dual_bound,
        "lp_relaxation_objective": with_lp.lp_relaxation_objective,
        "lp_relative_gap": with_lp.lp_relative_gap,
        "presolve_on_seconds": presolve_on_seconds,
        "presolve_off_status": off_status,
        "presolve_off_objective": off_objective,
        "presolve_off_mip_node_count": off_nodes,
        "presolve_off_seconds": off_seconds,
        "presolve_objective_equal": bool(
            np.isfinite(off_objective)
            and np.isclose(off_objective, exact.objective, rtol=0.0, atol=1e-8)
        ),
        "greedy_objective": float(greedy.objective),
        "greedy_size": len(greedy.members),
        "greedy_relative_gap": greedy_gap,
        "greedy_seconds": greedy_seconds,
        "compression_seconds": compression_seconds,
        "runtime_role": "environment-recorded implementation timing",
    }


def benchmark_registered_diagnostic_grid(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every registered controller-count/seed cell without filtering."""

    if config is None:
        config = REGISTERED_SCALING_CONFIG

    cells = [
        benchmark_diagnostic_instance(int(controller_count), int(seed), config)
        for controller_count in config["controller_counts"]
        for seed in config["seeds"]
    ]
    return {
        "version": str(config["version"]),
        "cell_count": len(cells),
        "environment": solver_environment_record(),
        "cells": cells,
    }
