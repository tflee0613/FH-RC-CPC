"""Code-only, fixed-input synthetic primary-MILP/LP/greedy diagnostics.

This is a benchmark of explicit solver stages, not the complete-certificate
wrapper. No observed result, governed row, or external input file is required.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from numbers import Integral, Real
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, eye, hstack, vstack

from ..qualification.failure_hypergraph import FailureHypergraph
from .scaling_grid import solver_environment_record

_CELLS = tuple((k, seed) for k in (24, 48) for seed in (7, 19, 43))
_GROUP_BETAS = (("pooled", "0.05"), ("first_half", "0.02"), ("alternating", "0.02"))
_VERSION = "paper-a-branching-stress-v1"
_TOLERANCE = 1e-7


def branching_stress_registration() -> dict[str, Any]:
    """Return a fresh JSON-safe copy of the fixed six-case scientific recipe."""
    return {
        "version": _VERSION,
        "registered_cells": [list(cell) for cell in _CELLS],
        "rng": "numpy.default_rng(numpy.random.SeedSequence([K, seed, 20260831]))",
        "tasks_per_controller": 4,
        "good_probability": 0.12,
        "zero_row_repair": "Ascending zero-row indices; one uniform column per row",
        "unit_costs_and_task_weights": True,
        "groups": [
            {"name": name, "beta": beta, "selector": selector}
            for (name, beta), selector in zip(
                _GROUP_BETAS,
                ("all rows", "row index < N/2", "zero-based even row index"),
            )
        ],
        "allowance": "floor(exact decimal beta * group size)",
        "max_size": "K",
        "primary_options": {"presolve": True, "time_limit": 5.0, "mip_rel_gap": 0.0},
        "lp_options": {"presolve": True, "time_limit": 1.0},
        "deterministic_tie_break": False,
        "greedy": "integer deficit reduction per unit cost; lower index breaks ties",
        "reporting": "all six cases, including unsolved or uncertified results",
        "scope": "synthetic primary/LP/greedy stages, not full-certificate timing",
    }


def _build_instance(controller_count: int, seed: int) -> tuple[FailureHypergraph, int]:
    if isinstance(controller_count, bool) or not isinstance(controller_count, Integral):
        raise ValueError("controller_count must be an integer in [1, 512]")
    if not 1 <= int(controller_count) <= 512:
        raise ValueError("controller_count must be an integer in [1, 512]")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    k, seed = int(controller_count), int(seed)
    n = 4 * k
    rng = np.random.default_rng(np.random.SeedSequence([k, seed, 20260831]))
    good = rng.random((n, k)) < 0.12
    repaired = np.flatnonzero(~good.any(axis=1))
    good[repaired, rng.integers(k, size=len(repaired))] = True
    graph = FailureHypergraph(
        good=good,
        controller_names=tuple(f"c{index:03d}" for index in range(k)),
        controller_costs=np.ones(k),
        groups={
            "pooled": np.ones(n, bool),
            "first_half": np.arange(n) < n // 2,
            "alternating": np.arange(n) % 2 == 0,
        },
        task_weights=np.ones(n),
    )
    return graph, int(len(repaired))


def branching_stress_instance(controller_count: int, seed: int) -> FailureHypergraph:
    """Generate immutable rows without solving or touching global RNG state.

    The registered study uses K=24/48 and seeds 7/19/43. Other integer menu sizes
    in [1,512] are supported for caller diagnostics/tests, not registered results.
    """
    return _build_instance(controller_count, seed)[0]


def _groups_and_allowances(graph: FailureHypergraph) -> tuple[np.ndarray, np.ndarray]:
    if graph.n_tasks != 4 * graph.n_controllers or set(graph.groups) != {
        g for g, _ in _GROUP_BETAS
    }:
        raise ValueError("input must have the branching-study dimensions and groups")
    if not (np.all(graph.controller_costs == 1) and np.all(graph.task_weights == 1)):
        raise ValueError("branching-study input hashes require unit costs and weights")
    groups = np.array([graph.groups[name] for name, _ in _GROUP_BETAS])
    allowances = np.array(
        [
            int(Fraction(beta) * int(mask.sum()))
            for (_, beta), mask in zip(_GROUP_BETAS, groups)
        ],
        dtype="<i8",
    )
    return groups, allowances


def branching_stress_input_sha256(graph: FailureHypergraph) -> str:
    """Hash row-major Boolean G, declared-order group masks, and little-endian B.

    Group order is pooled/first_half/alternating, not the graph mapping order.
    This study-specific convention assumes and checks unit costs/row weights.
    It is not a general FailureHypergraph identity hash.
    """
    groups, allowances = _groups_and_allowances(graph)
    return hashlib.sha256(
        graph.good.tobytes(order="C")
        + groups.tobytes(order="C")
        + allowances.tobytes(order="C")
    ).hexdigest()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def benchmark_branching_stress_instance(
    controller_count: int, seed: int
) -> dict[str, Any]:
    """Run primary MILP, LP and greedy once; independently check original rows.

    Non-optimal statuses and missing gap/dual evidence remain explicit. Timers
    exclude generation, signature grouping and initial sparse model assembly.
    No secondary tie solve or exhaustive verification is timed or invoked.
    """
    graph, repaired_count = _build_instance(controller_count, seed)
    k, n = graph.n_controllers, graph.n_tasks
    groups, budgets = _groups_and_allowances(graph)
    names = [name for name, _ in _GROUP_BETAS]
    good = graph.good
    signatures, inverse = np.unique(good, axis=0, return_inverse=True)
    q = len(signatures)
    masses = np.array([np.bincount(inverse, weights=g, minlength=q) for g in groups])
    cover = hstack(
        [csr_matrix(-signatures.astype(float)), -eye(q, format="csr")], format="csr"
    )
    group_rows = hstack([csr_matrix((3, k)), csr_matrix(masses)], format="csr")
    size_row = csr_matrix(np.r_[np.ones(k), np.zeros(q)].reshape(1, -1))
    matrix = vstack([cover, group_rows, size_row], format="csc")
    upper = np.r_[-np.ones(q), budgets, k]
    objective = np.r_[np.ones(k), np.zeros(q)]
    config = branching_stress_registration()

    def evaluate(members: list[int]) -> np.ndarray:
        counts = groups[:, ~good[:, members].any(axis=1)].sum(axis=1)
        compressed = masses[:, ~signatures[:, members].any(axis=1)].sum(axis=1)
        if not np.array_equal(counts, compressed):
            raise RuntimeError("original-row and signature group counts differ")
        return counts

    start = perf_counter()
    primary = milp(
        c=objective,
        integrality=np.r_[np.ones(k), np.zeros(q)],
        bounds=Bounds(np.zeros(k + q), np.ones(k + q)),
        constraints=LinearConstraint(matrix, -np.inf, upper),
        options=config["primary_options"],
    )
    primary_seconds = perf_counter() - start
    status = getattr(primary, "status", None)
    selected, counts, cost, validation_error = None, None, None, None
    reported_objective = _finite(getattr(primary, "fun", None))
    if getattr(primary, "x", None) is not None:
        x = np.asarray(primary.x, dtype=float)
        if x.shape != (k + q,) or not np.isfinite(x).all():
            validation_error = "invalid solution shape or nonfinite coordinates"
        elif (
            np.any(x < -_TOLERANCE)
            or np.any(x > 1 + _TOLERANCE)
            or np.max(matrix @ x - upper) > _TOLERANCE
            or np.max(np.abs(x[:k] - np.rint(x[:k]))) > _TOLERANCE
        ):
            validation_error = (
                "returned coordinates violate bounds, integrality or constraints"
            )
        else:
            candidate = np.flatnonzero(x[:k] >= 0.5).tolist()
            candidate_counts = evaluate(candidate)
            if np.any(candidate_counts > budgets):
                validation_error = "returned portfolio violates original-row allowances"
            elif (
                reported_objective is None
                or abs(reported_objective - len(candidate)) > _TOLERANCE
            ):
                validation_error = (
                    "reported objective disagrees with selected unit cost"
                )
            else:
                selected, counts, cost = candidate, candidate_counts, len(candidate)
    gap = _finite(getattr(primary, "mip_gap", None))
    dual = _finite(getattr(primary, "mip_dual_bound", None))
    optimal = bool(
        status == 0
        and getattr(primary, "success", False)
        and cost is not None
        and gap == 0.0
        and dual is not None
        and abs(cost - dual) <= _TOLERANCE
    )

    start = perf_counter()
    lp = linprog(
        objective,
        A_ub=matrix,
        b_ub=upper,
        bounds=(0.0, 1.0),
        method="highs",
        options=config["lp_options"],
    )
    lp_seconds = perf_counter() - start
    lp_value, lp_validation_error = None, None
    if getattr(lp, "status", None) == 0 and getattr(lp, "success", False):
        lp_x = np.asarray(getattr(lp, "x", None), dtype=float)
        value = _finite(getattr(lp, "fun", None))
        if lp_x.shape != (k + q,) or not np.isfinite(lp_x).all() or value is None:
            lp_validation_error = "invalid LP coordinates or objective"
        elif (
            np.any(lp_x < -_TOLERANCE)
            or np.any(lp_x > 1 + _TOLERANCE)
            or np.max(matrix @ lp_x - upper) > _TOLERANCE
            or abs(float(objective @ lp_x) - value) > _TOLERANCE
            or (cost is not None and value > cost + _TOLERANCE)
        ):
            lp_validation_error = "LP result fails primal/objective consistency checks"
        else:
            lp_value = value

    start = perf_counter()
    greedy: list[int] = []
    while True:
        misses = ~good[:, greedy].any(axis=1)
        deficits = np.maximum(groups[:, misses].sum(axis=1) - budgets, 0)
        if not deficits.any():
            break
        gains = [
            sum(
                min(int(d), int((g & misses & good[:, c]).sum()))
                for d, g in zip(deficits, groups)
            )
            if c not in greedy
            else -1
            for c in range(k)
        ]
        chosen = int(np.argmax(gains))
        if gains[chosen] <= 0:
            raise RuntimeError("greedy stalled despite a fully covering menu")
        greedy.append(chosen)
    greedy_seconds = perf_counter() - start
    greedy_counts = evaluate(greedy)
    if np.any(greedy_counts > budgets):
        raise RuntimeError("greedy violates original-row allowances")
    node_value = _finite(getattr(primary, "mip_node_count", None))
    return {
        "version": _VERSION,
        "K": k,
        "N": n,
        "Q": q,
        "seed": int(seed),
        "registered_case": (k, int(seed)) in _CELLS,
        "graph_sha256": branching_stress_input_sha256(graph),
        "environment": solver_environment_record(),
        "solver_options": {
            "primary": config["primary_options"],
            "lp": config["lp_options"],
        },
        "deterministic_tie_break": False,
        "initially_zero_rows_repaired": repaired_count,
        "good_density": float(good.mean()),
        "group_sizes": dict(zip(names, groups.sum(axis=1).tolist())),
        "allowances": dict(zip(names, budgets.tolist())),
        "controller_binaries": k,
        "auxiliaries": q,
        "constraints": matrix.shape[0],
        "nonzeros": matrix.nnz,
        "primary_status": None if status is None else int(status),
        "primary_message": str(getattr(primary, "message", "")),
        "optimal": optimal,
        "incumbent_verified": cost is not None,
        "validation_error": validation_error,
        "reported_objective": reported_objective,
        "incumbent_cost": cost,
        "selected_members": selected,
        "original_and_compressed_group_misses": None
        if counts is None
        else dict(zip(names, counts.tolist())),
        "mip_gap": gap,
        "dual_bound": dual,
        "nodes": None if node_value is None else int(node_value),
        "primary_seconds": primary_seconds,
        "lp_status": int(lp.status),
        "lp_message": str(lp.message),
        "lp_objective": lp_value,
        "lp_validation_error": lp_validation_error,
        "lp_seconds": lp_seconds,
        "integer_normalized_lp_gap": None
        if not optimal or lp_value is None
        else max(0.0, (cost - lp_value) / cost),
        "greedy_cost": len(greedy),
        "greedy_members": greedy,
        "greedy_seconds": greedy_seconds,
        "greedy_original_and_compressed_group_misses": dict(
            zip(names, greedy_counts.tolist())
        ),
        "greedy_to_optimum": None if not optimal else len(greedy) / cost,
        "runtime_role": "current-environment primary/LP/greedy stage timing only",
        "complete_optimum_classes_or_stability_benchmarked": False,
    }


def run_branching_stress_suite() -> dict[str, Any]:
    """Return every registered case without replacing timeouts or failures."""
    cells: list[dict[str, Any]] = []
    for k, seed in _CELLS:
        try:
            cell = benchmark_branching_stress_instance(k, seed)
        except Exception as error:
            cell = {
                "K": k,
                "seed": seed,
                "optimal": False,
                "status": "runner_error",
                "error": f"{type(error).__name__}: {error}",
            }
        cells.append(cell)
    return {
        "version": _VERSION,
        "registration": branching_stress_registration(),
        "environment": solver_environment_record(),
        "case_count": len(cells),
        "all_optimal": all(cell["optimal"] for cell in cells),
        "cells": cells,
        "synthetic_only": True,
        "contains_row_arrays": False,
        "peak_rss_measured": False,
    }
