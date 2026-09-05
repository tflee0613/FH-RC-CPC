#!/usr/bin/env python3
"""Rebuild the public qualification matrix and exact optima using stdlib only."""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

RECIPES = ["A32", "A64", "M16", "M64", "R0", "R1"]
SCOPES = [
    "development",
    "core_id",
    "parameter_shift",
    "independent_id",
    "disturbance_shift",
    "new_ib",
    "selection_isolated_tep",
]
CONTRACT = {
    "schema": "fh-rc-cpc-public-results-v1",
    "recipes": RECIPES,
    "scopes": SCOPES,
    "recipe_costs": [1] * 6,
    "epsilon": 1.0,
    "maximum_violation_rate": 0.01,
    "budget_numerator": 1,
    "budget_denominator": 5,
    "constraint_factor_threshold": 0.8,
    "registered_tasks": 432,
    "outcome_cells": 2592,
    "unsupported_rows_in_quota_denominator": False,
    "terminal_fidelity": "nmpc_iters=500; maximum rollout_steps per task; replicate=0",
}
TASK_FIELDS = ["task_id", "scope", "process", "constraint_factor", "supported"]
CELL_FIELDS = [
    "task_id",
    "recipe",
    "status",
    "score",
    "violation_rate",
    "accepted",
    "good",
]


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError("%s mismatch: %r != %r" % (label, actual, expected))


def number(text):
    if text == "":
        return None
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("use an empty CSV field for a missing value; no inf/nan")
    return value


def bit(text):
    if text not in ("0", "1"):
        raise ValueError("Boolean CSV fields must be 0 or 1")
    return text == "1"


def add_unique(mapping, key, value):
    if key in mapping:
        raise ValueError("duplicate key: %r" % (key,))
    mapping[key] = value


def require_grid(tasks, cells):
    expected = {(t, r) for t in tasks for r in RECIPES}
    if set(cells) != expected:
        raise ValueError("outcome grid contains missing or unexpected task/recipe keys")


def read_csv(path, fields):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require_equal(reader.fieldnames, fields, path.name + " schema")
        rows = list(reader)
    if any(set(row) != set(fields) or None in row.values() for row in rows):
        raise ValueError("malformed CSV row")
    return rows


def qualify(cells, epsilon=1.0):
    accepted = [
        c["status"] == "complete"
        and c["score"] is not None
        and c["violation_rate"] is not None
        and c["violation_rate"] <= 0.01
        for c in cells
    ]
    values = [c["score"] for c, a in zip(cells, accepted) if a]
    ref = min(values) if values else None
    good = [bool(a and c["score"] <= ref + epsilon) for c, a in zip(cells, accepted)]
    return accepted, ref, good


def load_data(folder):
    require_equal(
        json.loads((folder / "contract.json").read_text()), CONTRACT, "frozen contract"
    )
    tasks, cells = {}, {}
    for raw in read_csv(folder / "tasks.csv", TASK_FIELDS):
        task = dict(raw)
        if (
            not re.fullmatch(r"P[0-9]{4}", task["task_id"])
            or task["scope"] not in SCOPES
        ):
            raise ValueError("invalid public task ID or scope")
        if task["process"] not in (
            "crystallization",
            "four_tank",
            "ib",
            "multistage_extraction",
            "tep",
        ):
            raise ValueError("process outside the public six-recipe cohort")
        task["constraint_factor"] = number(task["constraint_factor"])
        task["supported"] = bit(task["supported"])
        add_unique(tasks, task["task_id"], task)
    for raw in read_csv(folder / "outcomes.csv", CELL_FIELDS):
        cell = dict(raw)
        if cell["status"] not in ("complete", "failed"):
            raise ValueError("nonterminal status")
        cell["score"] = number(cell["score"])
        cell["violation_rate"] = number(cell["violation_rate"])
        if cell["violation_rate"] is not None and not 0 <= cell["violation_rate"] <= 1:
            raise ValueError("violation rate outside [0, 1]")
        cell["accepted"], cell["good"] = bit(cell["accepted"]), bit(cell["good"])
        add_unique(cells, (cell["task_id"], cell["recipe"]), cell)
    require_equal(len(tasks), CONTRACT["registered_tasks"], "task count")
    require_equal(len(cells), CONTRACT["outcome_cells"], "cell count")
    require_grid(tasks, cells)
    ordered = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        row = [cells[task_id, recipe] for recipe in RECIPES]
        accepted, reference, good = qualify(row)
        require_equal(accepted, [c["accepted"] for c in row], task_id + " acceptance")
        require_equal(good, [c["good"] for c in row], task_id + " qualification")
        require_equal(any(accepted), task["supported"], task_id + " support")
        task.update(
            cells=row,
            reference=reference,
            good_mask=sum(1 << k for k, value in enumerate(good) if value),
        )
        ordered.append(task)
    return ordered


def quota(n):
    return n // 5


def make_groups(rows, scopes, pooled=False):
    groups = []
    blocks = [("pooled_scopes", scopes)] if pooled else [(s, [s]) for s in scopes]
    for label, selected_scopes in blocks:
        indices = {
            i
            for i, r in enumerate(rows)
            if r["supported"] and r["scope"] in selected_scopes
        }
        memberships = [("pooled", indices)]
        for plant in sorted({rows[i]["process"] for i in indices}):
            memberships.append(
                (
                    "process:" + plant,
                    {i for i in indices if rows[i]["process"] == plant},
                )
            )
        memberships.append(
            (
                "constraint_factor",
                {
                    i
                    for i in indices
                    if rows[i]["constraint_factor"] is not None
                    and rows[i]["constraint_factor"] >= 0.8
                },
            )
        )
        for kind, member in memberships:
            groups.append(
                {
                    "name": label + "::" + kind,
                    "rows": member,
                    "population": len(member),
                    "allowance": quota(len(member)),
                }
            )
    return groups


def solve(good_masks, groups, reserve=0):
    feasible, counts = [], []
    for mask in range(1, 64):
        misses = [sum(not (good_masks[i] & mask) for i in g["rows"]) for g in groups]
        counts.append(
            {"mask": mask, "size": bin(mask).count("1"), "group_misses": misses}
        )
        if all(
            m <= g["allowance"] - (reserve if g["population"] else 0)
            for m, g in zip(misses, groups)
        ):
            feasible.append(mask)
    minimum = min((bin(m).count("1") for m in feasible), default=None)
    return {
        "status": "GLOBAL_MINIMUM_CERTIFIED"
        if feasible
        else "GLOBAL_INFEASIBLE_BY_FULL_ENUMERATION",
        "minimum_size": minimum,
        "optimal_masks": [m for m in feasible if bin(m).count("1") == minimum],
        "feasible_masks": feasible,
        "all_mask_counts": counts,
    }


def compact(solution):
    return {k: v for k, v in solution.items() if k != "all_mask_counts"}


def calculate(rows):
    good_masks = [r["good_mask"] for r in rows]
    groups = make_groups(rows, SCOPES)
    result = {
        "counts": {
            "tasks": len(rows),
            "cells": len(rows) * 6,
            "complete": sum(
                c["status"] == "complete" for r in rows for c in r["cells"]
            ),
            "accepted": sum(c["accepted"] for r in rows for c in r["cells"]),
            "good": sum(c["good"] for r in rows for c in r["cells"]),
            "supported": sum(r["supported"] for r in rows),
        },
        "group_catalog": [{k: v for k, v in g.items() if k != "rows"} for g in groups],
        "all63_mask_group_counts": solve(good_masks, groups)["all_mask_counts"],
        "all127_scope_queries": [],
    }
    for scope_mask in range(1, 128):
        scopes = [s for i, s in enumerate(SCOPES) if scope_mask & (1 << i)]
        result["all127_scope_queries"].append(
            dict(
                scope_mask=scope_mask,
                scopes=scopes,
                **compact(solve(good_masks, make_groups(rows, scopes))),
            )
        )
    result["pooled_all_scopes"] = compact(
        solve(good_masks, make_groups(rows, SCOPES, pooled=True))
    )
    result["table5"] = []
    selected_names = [
        "development::pooled",
        "development::process:crystallization",
        "new_ib::pooled",
    ]
    selected_groups = [
        next(g for g in groups if g["name"] == n) for n in selected_names
    ]
    for mask in (55, 15, 47, 63):
        result["table5"].append(
            {
                "mask": mask,
                "count": bin(mask).count("1"),
                "groups": [
                    {
                        "name": g["name"],
                        "misses": sum(not (good_masks[i] & mask) for i in g["rows"]),
                        "population": g["population"],
                        "allowance": g["allowance"],
                    }
                    for g in selected_groups
                ],
            }
        )
    result["table6"] = []
    registry_rows = [dict(r, supported=True) for r in rows]
    for scope in SCOPES:
        scoped = [r for r in rows if r["scope"] == scope]
        supported = [r for r in scoped if r["supported"]]
        hits = sum(bool(r["good_mask"] & 53) for r in supported)
        opt = compact(solve(good_masks, make_groups(rows, [scope])))
        registry_opt = compact(solve(good_masks, make_groups(registry_rows, [scope])))
        expanded = compact(
            solve(
                good_masks,
                make_groups(rows, list(dict.fromkeys(["development", scope]))),
            )
        )
        result["table6"].append(
            {
                "scope": scope,
                "registered": len(scoped),
                "supported": len(supported),
                "mask53_hits": hits,
                "scope_optimum": opt,
                "registry_scope_optimum": registry_opt,
                "frozen53_supported_feasible": 53 in opt["feasible_masks"],
                "frozen53_registry_feasible": 53 in registry_opt["feasible_masks"],
                "expanded_optimum": expanded,
            }
        )
    result["epsilon_sensitivity"] = []
    for epsilon in (0, 0.25, 0.5, 1, 2, 5, 10):
        changed = [
            sum(1 << j for j, b in enumerate(qualify(r["cells"], epsilon)[2]) if b)
            for r in rows
        ]
        result["epsilon_sensitivity"].append(
            {
                "epsilon": epsilon,
                "development": compact(
                    solve(changed, make_groups(rows, ["development"]))
                ),
                "separate_all_scopes": compact(solve(changed, groups)),
            }
        )
    result["reserve_sensitivity"] = [
        {
            "reserve": ell,
            "development": compact(
                solve(good_masks, make_groups(rows, ["development"]), ell)
            ),
            "separate_all_scopes": compact(solve(good_masks, groups, ell)),
        }
        for ell in range(4)
    ]
    return result


def check_manifest(folder):
    listed = set()
    for line in (folder / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if (
            Path(name).name != name
            or name in (".", "..", "MANIFEST.sha256")
            or name in listed
        ):
            raise ValueError("invalid manifest member")
        listed.add(name)
        require_equal(
            hashlib.sha256((folder / name).read_bytes()).hexdigest(),
            digest,
            name + " SHA256",
        )
    required = {
        "tasks.csv",
        "outcomes.csv",
        "contract.json",
        "EXPECTED_RESULTS.json",
        "verify_results.py",
        "test_verify_results.py",
        "README.md",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "quota_sensitivity.py",
        "test_quota_sensitivity.py",
        "QUOTA_SENSITIVITY.json",
        "TASK_CONSTRUCTION.json",
        "verify_construction.py",
        "test_verify_construction.py",
    }
    require_equal(listed, required, "manifest file set")


def verify(folder, check_hashes=True):
    if check_hashes:
        check_manifest(folder)
    actual = calculate(load_data(folder))
    expected = json.loads((folder / "EXPECTED_RESULTS.json").read_text())
    require_equal(actual, expected, "recomputed results")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print all recomputed counts and solution classes",
    )
    args = parser.parse_args()
    result = verify(args.data_dir)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "PASS: CSV -> acceptance/reference/G -> 63 portfolios -> 127 scope combinations"
        )
        print(json.dumps(result["counts"], sort_keys=True))
        print(
            "Table 5: mask | count | development / crystallization / new-IB (misses/quota)"
        )
        for row in result["table5"]:
            print(
                "%d | %d | %s"
                % (
                    row["mask"],
                    row["count"],
                    " / ".join(
                        "%d:%d" % (g["misses"], g["allowance"]) for g in row["groups"]
                    ),
                )
            )
        print(
            "Table 6: seven registered/supported populations and frozen-53 coverage reproduced."
        )
        print("Development optima:", result["all127_scope_queries"][0]["optimal_masks"])
        print("All-scope optima:", result["all127_scope_queries"][-1]["optimal_masks"])


if __name__ == "__main__":
    main()
