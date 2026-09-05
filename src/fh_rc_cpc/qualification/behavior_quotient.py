"""Lossless controller-column quotient for static library compression."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Literal, Sequence

import numpy as np

from .failure_hypergraph import Controller, FailureHypergraph

LiftMode = Literal["minimum_cost", "all_equivalent"]


@dataclass(frozen=True)
class BehaviorQuotient:
    """Controller equivalence classes and their minimum-cost representatives."""

    graph: FailureHypergraph
    classes: tuple[tuple[int, ...], ...]
    representatives: tuple[int, ...]
    original_to_class: tuple[int, ...]
    original_names: tuple[str, ...]
    original_costs: tuple[float, ...]
    registered_behavior: bool = False

    def map_members(self, members: Sequence[Controller]) -> tuple[int, ...]:
        original = tuple(self._original_index(member) for member in members)
        return tuple(sorted({self.original_to_class[index] for index in original}))

    def _original_index(self, member: Controller) -> int:
        if isinstance(member, str):
            try:
                return self.original_names.index(member)
            except ValueError as error:
                raise KeyError(f"unknown controller: {member}") from error
        index = int(member)
        if index < 0 or index >= len(self.original_to_class):
            raise IndexError(f"controller index out of range: {index}")
        return index

    def _class_index(self, member: Controller) -> int:
        if isinstance(member, str):
            try:
                return self.graph.controller_names.index(member)
            except ValueError as error:
                raise KeyError(f"unknown quotient controller: {member}") from error
        index = int(member)
        if index < 0 or index >= len(self.classes):
            raise IndexError(f"behavior-class index out of range: {index}")
        return index

    def lift_members(
        self,
        members: Sequence[Controller],
        *,
        mode: LiftMode = "minimum_cost",
        cost_atol: float = 0.0,
    ) -> tuple[tuple[int, ...], ...]:
        """Lift one quotient portfolio to nominal controller choices.

        The default ``minimum_cost`` mode takes the Cartesian product of all
        class members whose nominal cost equals that class's minimum cost.
        Consequently, lifting a class-level additive-cost optimum preserves
        both behavior and objective value.  A nonzero ``cost_atol`` is one
        total excess-cost budget across all selected classes, not a budget
        per class.  Such near-minimum lifts concern the supplied class
        portfolio only; other near-optimal class portfolios are not searched.
        ``all_equivalent`` is an explicit behavior-only mode that also returns
        higher-cost nominal members; its output is an equivalence expansion,
        not an optimum-class lift.
        """

        if mode not in {"minimum_cost", "all_equivalent"}:
            raise ValueError("mode must be 'minimum_cost' or 'all_equivalent'")
        tolerance = float(cost_atol)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("cost_atol must be finite and nonnegative")
        classes = tuple(sorted(self._class_index(member) for member in members))
        if len(set(classes)) != len(classes):
            raise ValueError("quotient portfolio members must be unique")
        if not classes:
            return ((),)
        choices: list[tuple[int, ...]] = []
        for class_index in classes:
            class_members = self.classes[class_index]
            if mode == "all_equivalent":
                choices.append(class_members)
                continue
            minimum_cost = float(self.graph.controller_costs[class_index])
            choices.append(
                tuple(
                    index
                    for index in class_members
                    if abs(self.original_costs[index] - minimum_cost) <= tolerance
                )
            )
        lifted: set[tuple[int, ...]] = set()
        for choice in product(*choices):
            if mode == "minimum_cost":
                excess = math.fsum(
                    self.original_costs[index]
                    - float(self.graph.controller_costs[class_index])
                    for index, class_index in zip(choice, classes, strict=True)
                )
                if excess > tolerance:
                    continue
            lifted.add(tuple(sorted(int(value) for value in choice)))
        return tuple(sorted(lifted))

    def lift_masks(
        self,
        class_masks: Sequence[int],
        *,
        mode: LiftMode = "minimum_cost",
        cost_atol: float = 0.0,
    ) -> tuple[int, ...]:
        """Lift class masks using the declared cost-preserving/equivalent mode."""

        full = (1 << len(self.classes)) - 1
        nominal: set[int] = set()
        for raw_mask in class_masks:
            mask = int(raw_mask)
            if mask <= 0 or mask > full:
                raise ValueError("class mask is outside the quotient library")
            members = tuple(
                index for index in range(len(self.classes)) if mask & (1 << index)
            )
            for choice in self.lift_members(
                members,
                mode=mode,
                cost_atol=cost_atol,
            ):
                nominal.add(sum(1 << index for index in choice))
        return tuple(sorted(nominal))


def _column_key(values: np.ndarray) -> tuple[object, ...]:
    return tuple(
        value.item() if isinstance(value, np.generic) else value for value in values
    )


def quotient_controller_behavior(
    graph: FailureHypergraph,
    *,
    feasible_matrix: np.ndarray | None = None,
    path_matrix: np.ndarray | None = None,
    registered_behavior: bool = False,
) -> BehaviorQuotient:
    """Collapse globally equivalent controller columns without changing coverage.

    Equivalence always requires identical epsilon-good columns.  The strict
    ``registered_behavior`` mode additionally requires aligned feasibility and
    path/action matrices, implementing the registered ``G + feasibility +
    path`` equivalence contract. Within a class, the minimum-cost controller is
    retained; ties use the original controller order.
    """

    if registered_behavior and (path_matrix is None or feasible_matrix is None):
        raise ValueError(
            "registered_behavior requires both feasibility and path matrices"
        )

    matrices: list[np.ndarray] = [graph.good]
    for label, values in (
        ("path_matrix", path_matrix),
        ("feasible_matrix", feasible_matrix),
    ):
        if values is None:
            continue
        array = np.asarray(values)
        if array.shape != graph.good.shape:
            raise ValueError(f"{label} must align with the task-controller matrix")
        matrices.append(array)

    key_to_members: dict[tuple[tuple[object, ...], ...], list[int]] = {}
    for controller in range(graph.n_controllers):
        key = tuple(_column_key(matrix[:, controller]) for matrix in matrices)
        key_to_members.setdefault(key, []).append(controller)
    classes = tuple(tuple(values) for values in key_to_members.values())
    representatives = tuple(
        min(
            values,
            key=lambda index: (float(graph.controller_costs[index]), int(index)),
        )
        for values in classes
    )
    original_to_class = [-1] * graph.n_controllers
    for class_index, values in enumerate(classes):
        for controller in values:
            original_to_class[controller] = class_index
    quotient_graph = FailureHypergraph(
        good=graph.good[:, list(representatives)],
        controller_names=tuple(
            graph.controller_names[index] for index in representatives
        ),
        controller_costs=graph.controller_costs[list(representatives)],
        groups=graph.groups,
        task_weights=graph.task_weights,
    )
    return BehaviorQuotient(
        graph=quotient_graph,
        classes=classes,
        representatives=representatives,
        original_to_class=tuple(original_to_class),
        original_names=graph.controller_names,
        original_costs=tuple(float(value) for value in graph.controller_costs),
        registered_behavior=bool(registered_behavior),
    )
