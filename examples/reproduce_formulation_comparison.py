#!/usr/bin/env python3
"""Reproduce the reported analytic, scaling and branching experiment families."""

from __future__ import annotations

import json

from fh_rc_cpc.experiment_families import (
    benchmark_fractional_triangle_instance,
    benchmark_registered_diagnostic_grid,
    run_branching_stress_suite,
)


def main() -> None:
    triangles = [
        benchmark_fractional_triangle_instance(cycles, 10)
        for cycles in (10, 20, 40, 80)
    ]
    for row in triangles:
        cycles = int(row["cycle_count"])
        if not (
            row["task_count"] == 30 * cycles
            and row["nominal_controller_count"] == 6 * cycles
            and row["behavior_class_count"] == 3 * cycles
            and row["task_milp_objective"] == 2 * cycles
            and row["lp_objective"] == 1.5 * cycles
            and row["half_threshold_rounded_size"] == 3 * cycles
        ):
            raise RuntimeError("fractional-triangle identity changed")

    scaling = benchmark_registered_diagnostic_grid()
    if scaling["cell_count"] != 12:
        raise RuntimeError("registered scaling grid is incomplete")
    if not all(cell["exact_mip_gap"] == 0.0 for cell in scaling["cells"]):
        raise RuntimeError("a scaling case did not close at zero gap")

    branching = run_branching_stress_suite()
    if branching["case_count"] != 6 or not branching["all_optimal"]:
        raise RuntimeError("a registered branching case was not certified")

    print(
        json.dumps(
            {
                "status": "pass",
                "triangle_cases": len(triangles),
                "scaling_cases": scaling["cell_count"],
                "branching_cases": branching["case_count"],
                "branching_objectives": [
                    cell["incumbent_cost"] for cell in branching["cells"]
                ],
                "branching_nodes": [cell["nodes"] for cell in branching["cells"]],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
