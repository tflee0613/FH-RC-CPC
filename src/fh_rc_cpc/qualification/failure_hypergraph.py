"""Permutation-invariant failure-hypergraph representation for FH-RC-CPC."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

Controller = int | str


@dataclass(frozen=True)
class FailureHypergraph:
    """Finite task/controller outcomes with overlapping reliability groups.

    ``good[t, k]`` is true exactly when controller ``k`` is epsilon-good on
    eligible task ``t``.  Equivalently, each controller defines a failure
    hyperedge containing the rows on which its column is false.
    """

    good: np.ndarray
    controller_names: tuple[str, ...]
    controller_costs: np.ndarray
    groups: Mapping[str, np.ndarray]
    task_weights: np.ndarray

    def __post_init__(self) -> None:
        good = np.asarray(self.good)
        if good.dtype != np.bool_:
            raise ValueError("good must be a boolean task-by-controller matrix")
        if good.ndim != 2 or min(good.shape) <= 0:
            raise ValueError("good must be a nonempty task-by-controller matrix")
        n_tasks, n_controllers = good.shape

        names = tuple(self.controller_names)
        if (
            len(names) != n_controllers
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("controller_names must be nonempty, unique, and aligned")

        costs = np.asarray(self.controller_costs, dtype=float)
        if costs.shape != (n_controllers,):
            raise ValueError("controller_costs must align with controller columns")
        if not np.all(np.isfinite(costs)) or np.any(costs <= 0):
            raise ValueError("controller_costs must be finite and positive")

        weights = np.asarray(self.task_weights, dtype=float)
        if weights.shape != (n_tasks,):
            raise ValueError("task_weights must align with task rows")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("task_weights must be finite and positive")

        if not self.groups:
            raise ValueError("at least one reliability group is required")
        groups: dict[str, np.ndarray] = {}
        for name, values in sorted(self.groups.items()):
            if not isinstance(name, str) or not name:
                raise ValueError("group names must be nonempty strings")
            membership = np.asarray(values)
            if membership.dtype != np.bool_ or membership.shape != (n_tasks,):
                raise ValueError("group masks must be boolean and align with tasks")
            if not bool(membership.any()):
                raise ValueError("reliability groups cannot be empty")
            membership = membership.astype(bool, copy=True)
            membership.setflags(write=False)
            groups[name] = membership

        good = good.astype(bool, copy=True)
        costs = costs.astype(float, copy=True)
        weights = weights.astype(float, copy=True)
        good.setflags(write=False)
        costs.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "good", good)
        object.__setattr__(self, "controller_names", names)
        object.__setattr__(self, "controller_costs", costs)
        object.__setattr__(self, "groups", MappingProxyType(groups))
        object.__setattr__(self, "task_weights", weights)

    @property
    def n_tasks(self) -> int:
        return int(self.good.shape[0])

    @property
    def n_controllers(self) -> int:
        return int(self.good.shape[1])

    def controller_index(self, controller: Controller) -> int:
        if isinstance(controller, str):
            try:
                return self.controller_names.index(controller)
            except ValueError as error:
                raise KeyError(f"unknown controller: {controller}") from error
        index = int(controller)
        if index < 0 or index >= self.n_controllers:
            raise IndexError(f"controller index out of range: {index}")
        return index

    def _member_indices(self, members: Sequence[Controller]) -> tuple[int, ...]:
        indices = tuple(self.controller_index(member) for member in members)
        if len(set(indices)) != len(indices):
            raise ValueError("portfolio members must be unique")
        return indices

    def group_mask(self, group: str | None) -> np.ndarray:
        if group is None:
            return np.ones(self.n_tasks, dtype=bool)
        try:
            return self.groups[group]
        except KeyError as error:
            raise KeyError(f"unknown group: {group}") from error

    def common_failure(
        self,
        members: Sequence[Controller],
        *,
        group: str | None = None,
    ) -> np.ndarray:
        """Return tasks missed by every selected controller."""

        indices = self._member_indices(members)
        covered = (
            self.good[:, list(indices)].any(axis=1)
            if indices
            else np.zeros(self.n_tasks, dtype=bool)
        )
        failure = self.group_mask(group) & ~covered
        failure.setflags(write=False)
        return failure

    def pair_mechanism(
        self,
        controller_i: Controller,
        controller_j: Controller,
        *,
        group: str | None = None,
    ) -> dict[str, float | int | str]:
        """Summarize common failure and bidirectional rescue for a pair."""

        i = self.controller_index(controller_i)
        j = self.controller_index(controller_j)
        if i == j:
            raise ValueError("pair members must be distinct")
        domain = self.group_mask(group)
        weights = self.task_weights
        population_weight = float(weights[domain].sum())
        failure_i = domain & ~self.good[:, i]
        failure_j = domain & ~self.good[:, j]
        common = failure_i & failure_j
        i_rescues_j = domain & self.good[:, i] & ~self.good[:, j]
        j_rescues_i = domain & self.good[:, j] & ~self.good[:, i]

        failure_i_weight = float(weights[failure_i].sum())
        failure_j_weight = float(weights[failure_j].sum())
        common_weight = float(weights[common].sum())
        i_rescue_weight = float(weights[i_rescues_j].sum())
        j_rescue_weight = float(weights[j_rescues_i].sum())
        failure_i_rate = failure_i_weight / population_weight
        failure_j_rate = failure_j_weight / population_weight
        common_rate = common_weight / population_weight
        independence = failure_i_rate * failure_j_rate
        return {
            "controller_i": self.controller_names[i],
            "controller_j": self.controller_names[j],
            "group": "all" if group is None else group,
            "population_count": int(domain.sum()),
            "population_weight": population_weight,
            "failure_i_count": int(failure_i.sum()),
            "failure_j_count": int(failure_j.sum()),
            "common_failure_count": int(common.sum()),
            "i_rescues_j_count": int(i_rescues_j.sum()),
            "j_rescues_i_count": int(j_rescues_i.sum()),
            "failure_i_weight": failure_i_weight,
            "failure_j_weight": failure_j_weight,
            "common_failure_weight": common_weight,
            "i_rescues_j_weight": i_rescue_weight,
            "j_rescues_i_weight": j_rescue_weight,
            "failure_i_rate": failure_i_rate,
            "failure_j_rate": failure_j_rate,
            "common_failure_rate": common_rate,
            "independence_reference": independence,
            "independence_residual": common_rate - independence,
            "p_i_good_given_j_failure": (
                i_rescue_weight / failure_j_weight
                if failure_j_weight > 0
                else float("nan")
            ),
            "p_j_good_given_i_failure": (
                j_rescue_weight / failure_i_weight
                if failure_i_weight > 0
                else float("nan")
            ),
            "pair_cost": float(self.controller_costs[[i, j]].sum()),
        }

    def permute_controllers(self, order: Sequence[int]) -> FailureHypergraph:
        """Return an equivalent hypergraph under a controller-column permutation."""

        indices = tuple(int(index) for index in order)
        if sorted(indices) != list(range(self.n_controllers)):
            raise ValueError("order must be a permutation of all controller indices")
        return FailureHypergraph(
            good=self.good[:, list(indices)],
            controller_names=tuple(self.controller_names[index] for index in indices),
            controller_costs=self.controller_costs[list(indices)],
            groups=self.groups,
            task_weights=self.task_weights,
        )
