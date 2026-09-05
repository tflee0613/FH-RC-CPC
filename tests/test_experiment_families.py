"""Checks for the analytic and seeded synthetic families reported in the paper."""

from __future__ import annotations

import numpy as np

from fh_rc_cpc.experiment_families.branching_cases import (
    benchmark_branching_stress_instance,
    branching_stress_instance,
    branching_stress_registration,
)
from fh_rc_cpc.experiment_families.fractional_triangle import (
    benchmark_fractional_triangle_instance,
    fractional_triangle_instance,
    graph_sha256,
)
from fh_rc_cpc.experiment_families.scaling_grid import (
    REGISTERED_SCALING_CONFIG,
    diagnostic_instance,
)
from fh_rc_cpc.qualification import compress_failure_signatures


def test_fractional_triangle_identities_and_determinism() -> None:
    first = fractional_triangle_instance(4, 3)
    second = fractional_triangle_instance(4, 3)
    assert graph_sha256(first) == graph_sha256(second)
    assert first.good.shape == (36, 24)
    assert np.all(first.good.sum(axis=1) == 4)
    result = benchmark_fractional_triangle_instance(4, 3)
    assert result["behavior_class_count"] == 12
    assert result["task_signature_count"] == 12
    assert result["analytic_integer_optimum"] == 8
    assert result["task_milp_objective"] == 8
    assert result["signature_milp_objective"] == 8
    assert result["quotient_milp_objective"] == 8
    assert result["lp_objective"] == 6
    assert result["half_threshold_rounded_size"] == 12
    assert result["greedy_objective"] == 8


def test_registered_scaling_grid_has_twelve_deterministic_cells() -> None:
    config = REGISTERED_SCALING_CONFIG
    assert len(config["controller_counts"]) * len(config["seeds"]) == 12
    first = diagnostic_instance(60, 1000, 7)
    second = diagnostic_instance(60, 1000, 7)
    np.testing.assert_array_equal(first.good, second.good)
    np.testing.assert_array_equal(first.controller_costs, second.controller_costs)
    assert first.good.any(axis=1).all()
    assert compress_failure_signatures(first).n_signatures == 1000


def test_branching_registration_and_small_solver_check() -> None:
    registration = branching_stress_registration()
    assert registration["registered_cells"] == [
        [size, seed] for size in (24, 48) for seed in (7, 19, 43)
    ]
    first = branching_stress_instance(4, 7)
    second = branching_stress_instance(4, 7)
    np.testing.assert_array_equal(first.good, second.good)
    result = benchmark_branching_stress_instance(4, 7)
    assert result["optimal"]
    assert result["mip_gap"] == 0.0
    assert result["lp_objective"] <= result["incumbent_cost"]
    assert result["greedy_cost"] >= result["incumbent_cost"]
