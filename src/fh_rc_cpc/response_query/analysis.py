#!/usr/bin/env python3
"""Reconstruct the public pulse/recovery retention result using stdlib only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from . import qualification
from .metrics import phase_metrics

RECIPES = ["A32", "A64", "M16", "M64", "R0", "R1"]
FIELDS = [
    "task_id",
    "plant",
    "scope",
    "supported",
    "constraint_factor",
    "recipe",
    "status",
    "accepted",
    "good",
    "pulse_error",
    "recovery_error",
    "dynamic_error",
    "pulse_start",
    "pulse_stop",
    "recovery_start",
    "recovery_stop",
    "dynamic_reference",
    "dynamic_regret",
    "recipe_index",
]
CURVE_FIELDS = [
    "task_id",
    "recipe",
    "step",
    "h3",
    "h4",
    "reference_h3",
    "reference_h4",
    "v1",
    "v2",
    "pulse_window",
    "recovery_window",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: float, expected: float, label: str) -> None:
    require(
        math.isclose(actual, expected, rel_tol=2e-13, abs_tol=2e-13),
        f"{label} mismatch: {actual!r} != {expected!r}",
    )


def number(text: str) -> float | None:
    if text == "":
        return None
    value = float(text)
    require(math.isfinite(value), "numeric CSV fields must be finite or empty")
    return value


def bit(text: str) -> bool:
    require(text in ("0", "1"), "Boolean CSV fields must be 0 or 1")
    return text == "1"


def read_response_metrics(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == FIELDS, "response-metric schema mismatch")
        raw_rows = list(reader)
    require(len(raw_rows) == 360, "response-metric row count mismatch")
    rows = []
    numeric = {
        "constraint_factor",
        "pulse_error",
        "recovery_error",
        "dynamic_error",
        "pulse_start",
        "pulse_stop",
        "recovery_start",
        "recovery_stop",
        "dynamic_reference",
        "dynamic_regret",
    }
    for raw in raw_rows:
        require(
            None not in raw.values() and set(raw) == set(FIELDS),
            "malformed response row",
        )
        row = dict(raw)
        for field in numeric:
            row[field] = number(row[field])
        for field in ("supported", "accepted", "good"):
            row[field] = bit(row[field])
        row["recipe_index"] = int(row["recipe_index"])
        require(row["recipe"] in RECIPES, "unknown recipe")
        require(
            row["recipe_index"] == RECIPES.index(row["recipe"]), "recipe index mismatch"
        )
        rows.append(row)
    keys = [(row["task_id"], row["recipe"]) for row in rows]
    require(len(set(keys)) == len(keys), "duplicate response key")
    return rows


def audit_response_rows(rows: list[dict], qualification_rows: list[dict]) -> dict:
    base = {row["task_id"]: row for row in qualification_rows}
    expected_tasks = sorted(
        row["task_id"]
        for row in qualification_rows
        if row["scope"] == "disturbance_shift"
        and row["process"] in ("crystallization", "four_tank", "multistage_extraction")
    )
    tasks = sorted({row["task_id"] for row in rows})
    require(tasks == expected_tasks and len(tasks) == 60, "response cohort mismatch")
    by_task = {task_id: [] for task_id in tasks}
    for row in rows:
        require(row["task_id"] in base, "response task outside qualification data")
        source = base[row["task_id"]]
        require(
            row["scope"] == source["scope"] == "disturbance_shift", "scope mismatch"
        )
        require(row["plant"] == source["process"], "process mismatch")
        require(row["supported"] == source["supported"], "support mismatch")
        close(
            row["constraint_factor"], source["constraint_factor"], "constraint factor"
        )
        cell = source["cells"][row["recipe_index"]]
        require(row["status"] == cell["status"], "terminal status mismatch")
        require(row["accepted"] == cell["accepted"], "acceptance mismatch")
        require(row["good"] == cell["good"], "original-good mismatch")
        if row["dynamic_error"] is not None:
            require(
                row["pulse_error"] is not None and row["recovery_error"] is not None,
                "finite dynamic error requires both phase metrics",
            )
            require(
                row["pulse_error"] >= 0 and row["recovery_error"] >= 0,
                "normalized phase errors must be nonnegative",
            )
            close(
                row["dynamic_error"],
                max(row["pulse_error"], row["recovery_error"]),
                row["task_id"] + " phase maximum",
            )
        by_task[row["task_id"]].append(row)
    for task_id, cells in by_task.items():
        cells.sort(key=lambda item: item["recipe_index"])
        require(
            [cell["recipe"] for cell in cells] == RECIPES,
            task_id + " recipe grid mismatch",
        )
        eligible = [
            cell
            for cell in cells
            if cell["accepted"] and cell["dynamic_error"] is not None
        ]
        if not eligible:
            require(
                all(
                    cell["dynamic_reference"] is None and cell["dynamic_regret"] is None
                    for cell in cells
                ),
                task_id + " unsupported reference mismatch",
            )
            continue
        reference = min(cell["dynamic_error"] for cell in eligible)
        for cell in cells:
            close(cell["dynamic_reference"], reference, task_id + " dynamic reference")
            if cell["dynamic_error"] is not None:
                regret = cell["dynamic_error"] - reference
                if abs(regret) < 1e-14:
                    regret = 0.0
                close(cell["dynamic_regret"], regret, task_id + " dynamic regret")
            else:
                require(
                    cell["dynamic_regret"] is None,
                    task_id + " ineligible regret must be empty",
                )
    return {
        "tasks": len(tasks),
        "supported_tasks": sum(base[task]["supported"] for task in tasks),
        "cells": len(rows),
        "by_task": by_task,
    }


def dynamic_groups(by_task: dict[str, list[dict]]) -> list[dict]:
    task_rows = [cells[0] for _, cells in sorted(by_task.items())]
    supported = [row for row in task_rows if row["supported"]]
    memberships = [("dynamic::pooled", supported)]
    for plant in sorted({row["plant"] for row in supported}):
        memberships.append(
            (
                "dynamic::process:" + plant,
                [row for row in supported if row["plant"] == plant],
            )
        )
    memberships.append(
        (
            "dynamic::constraint_factor",
            [row for row in supported if row["constraint_factor"] >= 0.8],
        )
    )
    return [
        {
            "name": name,
            "task_ids": [row["task_id"] for row in members],
            "population": len(members),
            "allowance": len(members) // 5,
        }
        for name, members in memberships
    ]


def solve_at(
    theta: float,
    qualification_rows: list[dict],
    qualification,
    by_task,
    groups: list[dict],
) -> list[dict]:
    """Enumerate all 63 portfolios under original and added response obligations."""
    require(math.isfinite(theta) and theta >= 0, "theta must be finite and nonnegative")
    original = qualification.make_groups(qualification_rows, qualification.SCOPES)
    result = []
    for mask in range(1, 64):
        original_misses = {
            "original::" + group["name"]: sum(
                not (qualification_rows[index]["good_mask"] & mask)
                for index in group["rows"]
            )
            for group in original
        }
        response_misses = {}
        for group in groups:
            misses = 0
            for task_id in group["task_ids"]:
                covered = any(
                    mask & (1 << cell["recipe_index"])
                    and cell["good"]
                    and cell["dynamic_regret"] is not None
                    and cell["dynamic_regret"] <= theta
                    for cell in by_task[task_id]
                )
                misses += int(not covered)
            response_misses[group["name"]] = misses
        feasible = all(
            original_misses["original::" + group["name"]] <= group["allowance"]
            for group in original
        ) and all(
            response_misses[group["name"]] <= group["allowance"] for group in groups
        )
        result.append(
            {
                "mask": mask,
                "size": mask.bit_count(),
                "feasible": feasible,
                "dynamic_misses": response_misses,
                "original_misses": original_misses,
            }
        )
    return result


def frontier(
    metrics: list[dict],
    qualification_rows: list[dict],
    qualification,
    by_task,
    groups: list[dict],
) -> tuple[list[dict], list[tuple]]:
    breakpoints = sorted(
        {
            0.0,
            *(
                cell["dynamic_regret"]
                for cell in metrics
                if cell["good"] and cell["dynamic_regret"] is not None
            ),
        }
    )
    states = []
    for theta in breakpoints:
        solutions = solve_at(theta, qualification_rows, qualification, by_task, groups)
        feasible = [row for row in solutions if row["feasible"]]
        require(feasible, "response contract unexpectedly infeasible")
        minimum = min(row["size"] for row in feasible)
        optima = [row["mask"] for row in feasible if row["size"] == minimum]
        states.append((theta, minimum, optima, solutions))
    intervals = []
    start = states[0]
    for state in states[1:]:
        if state[1:3] != start[1:3]:
            intervals.append(
                {
                    "lower": start[0],
                    "upper": state[0],
                    "upper_inclusive": False,
                    "minimum_size": start[1],
                    "optimum_masks": start[2],
                }
            )
            start = state
    intervals.append(
        {
            "lower": start[0],
            "upper": None,
            "upper_inclusive": True,
            "minimum_size": start[1],
            "optimum_masks": start[2],
        }
    )
    return intervals, states


def verify_curves(metrics: list[dict], response_dir: Path) -> int:
    path = response_dir / "representative_response_P0010.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(
            reader.fieldnames == CURVE_FIELDS, "representative-curve schema mismatch"
        )
        rows = list(reader)
    require(len(rows) == 600, "representative-curve row count mismatch")
    response = {(row["task_id"], row["recipe"]): row for row in metrics}
    references = None
    for recipe in RECIPES:
        recipe_rows = [row for row in rows if row["recipe"] == recipe]
        require(len(recipe_rows) == 100, recipe + " curve length mismatch")
        require(
            [int(row["step"]) for row in recipe_rows] == list(range(100)),
            recipe + " step order mismatch",
        )
        require(
            all(row["task_id"] == "P0010" for row in recipe_rows),
            "representative task mismatch",
        )
        current_references = [
            (float(row["reference_h3"]), float(row["reference_h4"]))
            for row in recipe_rows
        ]
        if references is None:
            references = current_references
        else:
            require(references == current_references, "candidate references disagree")
        outputs = []
        for row in recipe_rows:
            require(
                row["pulse_window"] in ("0", "1")
                and row["recovery_window"] in ("0", "1"),
                "invalid curve-window flag",
            )
            step = int(row["step"])
            require(
                int(row["pulse_window"]) == int(27 <= step < 50), "pulse flag mismatch"
            )
            require(
                int(row["recovery_window"]) == int(50 <= step < 73),
                "recovery flag mismatch",
            )
            values = [
                float(row[field])
                for field in ("h3", "h4", "reference_h3", "reference_h4", "v1", "v2")
            ]
            require(
                all(math.isfinite(value) for value in values), "nonfinite curve value"
            )
            outputs.append(values[:2])
        phases = phase_metrics(
            outputs, current_references, [0.6, 0.6], (27, 50), (50, 73)
        )
        pulse, recovery = phases["pulse_error"], phases["recovery_error"]
        metric = response[("P0010", recipe)]
        close(pulse, metric["pulse_error"], recipe + " representative pulse error")
        close(
            recovery,
            metric["recovery_error"],
            recipe + " representative recovery error",
        )
        close(
            max(pulse, recovery),
            metric["dynamic_error"],
            recipe + " representative dynamic error",
        )
    return len(rows)


def compare_expected(actual: dict, expected: dict) -> None:
    for key in (
        "tasks",
        "supported_tasks",
        "cells",
        "breakpoints",
        "portfolio_checks",
        "strict_minimum_size",
        "strict_optimum_masks",
        "relaxed_minimum_size",
        "relaxed_optimum_masks",
        "strict_four_tank_misses_mask47",
        "strict_four_tank_budget",
        "strict_unique_R0_tasks",
        "representative_task",
        "representative_curve_rows",
    ):
        require(actual[key] == expected[key], key + " mismatch")
    close(actual["boundary"], expected["boundary"], "frontier boundary")
    require(
        actual["dynamic_groups"] == expected["dynamic_groups"],
        "dynamic groups mismatch",
    )
    require(
        len(actual["frontier"]) == len(expected["frontier"]), "frontier length mismatch"
    )
    for actual_row, expected_row in zip(actual["frontier"], expected["frontier"]):
        for key in ("upper_inclusive", "minimum_size", "optimum_masks"):
            require(
                actual_row[key] == expected_row[key], "frontier " + key + " mismatch"
            )
        close(actual_row["lower"], expected_row["lower"], "frontier lower")
        if expected_row["upper"] is None:
            require(actual_row["upper"] is None, "frontier upper mismatch")
        else:
            close(actual_row["upper"], expected_row["upper"], "frontier upper")


def load_inputs(data_dir: Path) -> tuple:
    """Read the published cohort without importing executable code from data_dir."""
    data_dir = Path(data_dir)
    contract_path = data_dir / "response_retention" / "response_contract.json"
    require(
        hashlib.sha256(contract_path.read_bytes()).hexdigest()
        == "849d034b6fd63e3bcb13c987578cb434cc66067ecc75de9d6780be8dd1bfb615",
        "response contract differs from the published query specification",
    )
    qualification_dir = data_dir / "qualification"
    qualification.check_manifest(qualification_dir)
    qualification_rows = qualification.load_data(qualification_dir)
    metrics = read_response_metrics(
        data_dir / "response_retention" / "response_metrics.csv"
    )
    audit = audit_response_rows(metrics, qualification_rows)
    groups = dynamic_groups(audit["by_task"])
    return qualification_rows, metrics, audit, groups


def query(data_dir: Path, theta: float) -> dict:
    """Return the complete optimum class and per-group misses for a chosen query."""
    require(math.isfinite(theta) and theta >= 0, "theta must be finite and nonnegative")
    rows, _, audit, groups = load_inputs(data_dir)
    solutions = solve_at(theta, rows, qualification, audit["by_task"], groups)
    feasible = [row for row in solutions if row["feasible"]]
    minimum = min((row["size"] for row in feasible), default=None)
    return {
        "theta": theta,
        "minimum_size": minimum,
        "optimum_masks": [row["mask"] for row in feasible if row["size"] == minimum],
        "portfolios": solutions,
    }


def verify(data_dir: Path) -> dict:
    """Reconstruct the full published frontier and compare every expected result."""
    response_dir = Path(data_dir) / "response_retention"
    qualification_rows, metrics, audit, groups = load_inputs(data_dir)
    qualification.verify(Path(data_dir) / "qualification", check_hashes=True)
    intervals, states = frontier(
        metrics, qualification_rows, qualification, audit["by_task"], groups
    )
    expected = json.loads(
        (response_dir / "EXPECTED_RESPONSE_RESULTS.json").read_text(encoding="utf-8")
    )
    boundary = intervals[0]["upper"]
    strict_state = next(state for state in reversed(states) if state[0] < boundary)
    relaxed_state = next(state for state in states if state[0] == boundary)
    strict_mask47 = next(row for row in strict_state[3] if row["mask"] == 47)
    unique_r0_tasks = sorted(
        task_id
        for task_id, cells in audit["by_task"].items()
        if any(
            cell["recipe"] == "R0"
            and cell["good"]
            and cell["dynamic_regret"] is not None
            and cell["dynamic_regret"] <= strict_state[0]
            for cell in cells
        )
        and not any(
            cell["recipe"] != "R0"
            and cell["good"]
            and cell["dynamic_regret"] is not None
            and cell["dynamic_regret"] <= strict_state[0]
            for cell in cells
        )
    )
    curve_rows = verify_curves(metrics, response_dir)
    actual = {
        "tasks": audit["tasks"],
        "supported_tasks": audit["supported_tasks"],
        "cells": audit["cells"],
        "breakpoints": len(states),
        "portfolio_checks": len(states) * 63,
        "dynamic_groups": [
            {
                "name": group["name"],
                "population": group["population"],
                "allowance": group["allowance"],
            }
            for group in groups
        ],
        "frontier": intervals,
        "boundary": boundary,
        "strict_minimum_size": strict_state[1],
        "strict_optimum_masks": strict_state[2],
        "relaxed_minimum_size": relaxed_state[1],
        "relaxed_optimum_masks": relaxed_state[2],
        "strict_four_tank_misses_mask47": strict_mask47["dynamic_misses"][
            "dynamic::process:four_tank"
        ],
        "strict_four_tank_budget": next(
            group["allowance"]
            for group in groups
            if group["name"] == "dynamic::process:four_tank"
        ),
        "strict_unique_R0_tasks": unique_r0_tasks,
        "representative_task": "P0010",
        "representative_curve_rows": curve_rows,
    }
    compare_expected(actual, expected)
    return actual
