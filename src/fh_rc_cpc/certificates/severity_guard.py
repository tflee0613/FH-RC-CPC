"""An exact maximum-loss guard composed with unchanged frequency contracts.

This adapter converts accepted full-menu scores into a separate zero-miss
scope. It does not replace epsilon-good outcomes or any original allowance.
The guard concerns retained opportunities on fixed evidence, not a runtime
selector, an absolute physical limit, or a row-reserve certificate.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from numbers import Integral, Real
from operator import index

import numpy as np

from ..qualification.failure_hypergraph import FailureHypergraph
from .scope_composition import ScopeContract

RelativeLosses = tuple[tuple[Fraction | None, ...], ...]


def _binary64_exact(value: Real, name: str) -> Fraction:
    """Reject nonfinite or lossy conversion instead of changing the evidence."""
    try:
        converted = float(value)
        if not np.isfinite(converted):
            raise ValueError("nonfinite binary64 value")
        result = Fraction.from_float(converted)
        if isinstance(value, Integral):
            original = Fraction(int(value))
        else:
            original = Fraction(*value.as_integer_ratio())
    except (OverflowError, TypeError, ValueError, AttributeError) as error:
        raise ValueError(
            f"{name} must be exactly representable as finite binary64"
        ) from error
    if original != result:
        raise ValueError(f"{name} must be exactly representable as finite binary64")
    return result


def _exact_score_rows(
    accepted: np.ndarray,
    scores: np.ndarray,
) -> RelativeLosses:
    """Snapshot accepted finite scores as exact binary-rational values."""
    acceptance = np.asarray(accepted).copy()
    values = np.asarray(scores).copy()
    if (
        acceptance.dtype != np.dtype(bool)
        or acceptance.ndim != 2
        or min(acceptance.shape, default=0) <= 0
    ):
        raise ValueError("accepted must be a nonempty two-dimensional Boolean matrix")
    if values.shape != acceptance.shape or values.dtype.kind not in "iuf":
        raise ValueError("scores must be an aligned real numeric matrix")
    if not acceptance.any(axis=1).all():
        raise ValueError("every task must pass the full-menu acceptance support gate")
    return tuple(
        tuple(
            _binary64_exact(value, "accepted score") if admitted else None
            for admitted, value in zip(row_accepted, row_scores, strict=True)
        )
        for row_accepted, row_scores in zip(acceptance, values, strict=True)
    )


def relative_score_losses(
    accepted: np.ndarray,
    scores: np.ndarray,
    *,
    absolute_tolerance_floor: Fraction | float = 0,
    baseline_good: np.ndarray | None = None,
) -> RelativeLosses:
    """Return immutable exact accepted-range losses; ``None`` means rejected.

    Every row must have an accepted candidate with a finite score. The existing
    evidence/support gates must be applied by the caller before this conversion.
    Finite binary64 scores are interpreted as their exact represented rationals;
    other numeric dtypes must convert losslessly. There is no numerical tolerance.
    Equal-score accepted rows have zero loss.
    Rejected scores may be nonfinite and never influence the reference or range.
    Optional floors set the loss to zero: either a nonnegative absolute score
    difference floor, or authoritative registered ``baseline_good`` entries.
    The latter must be an aligned Boolean subset of accepted entries and keeps
    the original comparison's rounding. With either floor, these are tail losses,
    not raw normalized regret. Both options default to the raw-loss query.
    """
    floor = _nonnegative_real(absolute_tolerance_floor, "absolute_tolerance_floor")
    rows = _exact_score_rows(accepted, scores)
    baseline = np.zeros((len(rows), len(rows[0])), dtype=bool)
    if baseline_good is not None:
        baseline = np.asarray(baseline_good).copy()
        actual_accepted = np.array([[v is not None for v in row] for row in rows])
        if (
            baseline.dtype != np.dtype(bool)
            or baseline.shape != actual_accepted.shape
            or np.any(baseline & ~actual_accepted)
        ):
            raise ValueError(
                "baseline_good must be an aligned Boolean subset of accepted outcomes"
            )
    result = []
    for exact, base_row in zip(rows, baseline, strict=True):
        available = tuple(value for value in exact if value is not None)
        low, high = min(available), max(available)
        span = high - low
        result.append(
            tuple(
                None
                if value is None
                else Fraction(0)
                if registered_good or value - low <= floor or not span
                else (value - low) / span
                for value, registered_good in zip(exact, base_row, strict=True)
            )
        )
    return tuple(result)


def _nonnegative_real(value: Fraction | float, name: str) -> Fraction:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not Boolean")
    if isinstance(value, Fraction):
        threshold = value
    elif isinstance(value, Real):
        threshold = _binary64_exact(value, name)
    else:
        raise ValueError(f"{name} must be a finite real or Fraction")
    if threshold < 0:
        raise ValueError(f"{name} must be nonnegative")
    return threshold


def score_range_guard_scope(
    *,
    name: str,
    accepted: np.ndarray,
    scores: np.ndarray,
    controller_names: Sequence[str],
    controller_costs: np.ndarray,
    maximum_relative_loss: Fraction | float,
    absolute_tolerance_floor: Fraction | float = 0,
    baseline_good: np.ndarray | None = None,
) -> ScopeContract:
    """Build a zero-miss scope for a supplied maximum relative opportunity loss.

    Compose the returned scope with **every original** scope using
    ``compose_scopes``. The engine checks their candidate-ID/cost map and returns
    all minimum-cost portfolios. The original frequency budgets stay unchanged.
    Caller registration must bind the score/evaluator version, full candidate
    menu and threshold. This object snapshots the derived binary evidence; it is
    not evidence that a physical loss limit or future policy behavior is met.

    Guard rows describe another predicate on existing tasks. They are not new
    independent observations and must not be counted as additional physical rows
    when requesting a shared outcome-row reserve.

    A nonnegative absolute tolerance floor uses the threshold
    ``max(floor, eta * accepted_score_span)`` for excess above the best score.
    It keeps the relative guard from tightening negligible in-tolerance losses.
    Pass the authoritative original Boolean ``baseline_good`` matrix to retain
    its entries with their registered rounding. This explicitly unions those
    entries with the new exact tail predicate; it is not a tolerance added to
    the latter. Nonaccepted entries cannot be promoted by this floor.
    ``maximum_portfolio_loss`` accepts the same options for matching queries.
    """
    eta = _nonnegative_real(maximum_relative_loss, "maximum_relative_loss")
    if eta > 1:
        raise ValueError("maximum_relative_loss must lie in [0, 1]")
    losses = relative_score_losses(
        accepted,
        scores,
        absolute_tolerance_floor=absolute_tolerance_floor,
        baseline_good=baseline_good,
    )
    good = np.array(
        [[value is not None and value <= eta for value in row] for row in losses],
        dtype=bool,
    )
    graph = FailureHypergraph(
        good,
        tuple(controller_names),
        controller_costs,
        {"maximum_relative_loss": np.ones(len(losses), dtype=bool)},
        np.ones(len(losses)),
    )
    return ScopeContract(name, graph, {"maximum_relative_loss": 0.0})


def maximum_portfolio_loss(
    accepted: np.ndarray,
    scores: np.ndarray,
    members: Sequence[int],
    *,
    absolute_tolerance_floor: Fraction | float = 0,
    baseline_good: np.ndarray | None = None,
) -> Fraction | None:
    """Maximum of retained per-task minimum losses; ``None`` denotes infinity.

    Members are distinct, zero-based column indices. No retained accepted member
    on any task, including an empty portfolio, returns ``None``. This is different
    from a finite loss of one (retaining only the worst accepted score).
    Floor options have exactly the semantics of ``relative_score_losses`` and
    ``score_range_guard_scope``. A floored query reports maximum tail loss.
    """
    losses = relative_score_losses(
        accepted,
        scores,
        absolute_tolerance_floor=absolute_tolerance_floor,
        baseline_good=baseline_good,
    )
    if isinstance(members, (str, bytes)):
        raise ValueError("members must be a sequence of distinct integer indices")
    try:
        raw = tuple(members)
        if any(isinstance(member, (bool, np.bool_)) for member in raw):
            raise ValueError("member indices must not be Boolean")
        chosen = tuple(index(member) for member in raw)
    except TypeError as error:
        raise ValueError(
            "members must be a sequence of distinct integer indices"
        ) from error
    if len(chosen) != len(set(chosen)) or any(
        member < 0 or member >= len(losses[0]) for member in chosen
    ):
        raise ValueError("members must be distinct registered column indices")
    worst = Fraction(0)
    for row in losses:
        available = tuple(row[member] for member in chosen if row[member] is not None)
        if not available:
            return None
        worst = max(worst, min(available))
    return worst
