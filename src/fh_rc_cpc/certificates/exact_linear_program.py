"""Bounded rational LPs with independently checked primal/dual certificates.

This deliberately small two-phase simplex is for certificate LPs, not the
primary portfolio MILP. All arithmetic is Fraction arithmetic. Bland's rule
handles degeneracy; work/bit guards raise rather than manufacture a status.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence


class ExactLPCertificationError(RuntimeError):
    """A finite exact certificate could not be established within the guards."""


@dataclass(frozen=True)
class ExactLPResult:
    status: str
    x: tuple[Fraction, ...] | None
    objective: Fraction | None
    multipliers: tuple[Fraction, ...] = ()
    ray: tuple[Fraction, ...] | None = None
    pivots: int = 0


def _dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction())


def _verify(c, a, b, lower, original_a, original_b, upper, result):
    """Check the ORIGINAL inequalities, independently of the pivot tableau."""

    n = len(c)
    if result.status == "infeasible":
        dual = result.multipliers
        valid = (
            len(dual) == len(a)
            and all(value <= 0 for value in dual)
            and all(_dot(dual, [row[j] for row in a]) <= 0 for j in range(n))
            and _dot(dual, b) > 0
        )
    else:
        x = result.x
        valid = x is not None and len(x) == n
        if valid:
            valid = (
                all(x[j] >= lower[j] for j in range(n))
                and all(upper[j] is None or x[j] <= upper[j] for j in range(n))
                and all(
                    _dot(row, x) <= rhs
                    for row, rhs in zip(original_a, original_b, strict=True)
                )
            )
        if valid and result.status == "optimal":
            dual = result.multipliers
            y = tuple(x[j] - lower[j] for j in range(n))
            valid = (
                len(dual) == len(a)
                and all(value <= 0 for value in dual)
                and all(_dot(dual, [row[j] for row in a]) <= c[j] for j in range(n))
                and _dot(dual, b) == _dot(c, y)
                and result.objective == _dot(c, x)
            )
        elif valid and result.status == "unbounded":
            ray = result.ray
            valid = (
                ray is not None
                and len(ray) == n
                and all(value >= 0 for value in ray)
                and all(_dot(row, ray) <= 0 for row in a)
                and _dot(c, ray) < 0
            )
        elif result.status not in {"optimal", "unbounded"}:
            valid = False
    if not valid:
        raise ExactLPCertificationError(
            "rational LP failed independent primal/dual verification"
        )


def solve_exact_lp(
    objective: Sequence[Fraction],
    a_ub: Sequence[Sequence[Fraction]],
    b_ub: Sequence[Fraction],
    bounds: Sequence[tuple[Fraction, Fraction | None]],
    *,
    max_pivots: int = 10_000,
    max_bits: int = 32_768,
) -> ExactLPResult:
    """Minimize c*x with A*x<=b and finite lower / optional upper bounds.

    Results are returned only after exact original-inequality checks. An
    optimum carries nonpositive inequality multipliers with matching objective;
    infeasibility carries lambda<=0, A^T*lambda<=0, b^T*lambda>0 on the
    lower-shifted inequalities (including upper bounds). An unbounded result
    carries an original feasible point and an improving feasible ray.
    """

    c = tuple(map(Fraction, objective))
    n = len(c)
    original_a = tuple(tuple(map(Fraction, row)) for row in a_ub)
    original_b = tuple(map(Fraction, b_ub))
    if not n or n > 128 or len(bounds) != n:
        raise ExactLPCertificationError(
            "exact LP needs 1..128 variables and aligned bounds"
        )
    if len(original_a) != len(original_b) or any(len(row) != n for row in original_a):
        raise ValueError("LP inequalities must align")
    if max_pivots < 0 or max_bits < 1:
        raise ValueError("exact LP guards must be nonnegative pivots / positive bits")
    lower = tuple(Fraction(pair[0]) for pair in bounds)
    upper = tuple(None if pair[1] is None else Fraction(pair[1]) for pair in bounds)
    if any(hi is not None and lo > hi for lo, hi in zip(lower, upper, strict=True)):
        raise ValueError("LP bounds must be ordered")
    a = list(original_a)
    b = [rhs - _dot(row, lower) for row, rhs in zip(a, original_b, strict=True)]
    for j, hi in enumerate(upper):
        if hi is not None:
            a.append(tuple(Fraction(int(i == j)) for i in range(n)))
            b.append(hi - lower[j])
    m = len(a)
    if m > 512:
        raise ExactLPCertificationError("exact LP exceeds 512 augmented inequalities")
    signs = [Fraction(1 if value >= 0 else -1) for value in b]
    negative = [i for i, rhs in enumerate(b) if rhs < 0]
    real_count = n + m
    width = real_count + len(negative)
    artificial = {row: real_count + i for i, row in enumerate(negative)}
    rows = []
    rhs = []
    basis = []
    transform = []
    for i in range(m):
        row = [signs[i] * value for value in a[i]] + [Fraction()] * (width - n)
        row[n + i] = signs[i]
        basic = n + i
        if i in artificial:
            basic = artificial[i]
            row[basic] = Fraction(1)
        rows.append(row)
        rhs.append(signs[i] * b[i])
        basis.append(basic)
        transform.append([Fraction(int(i == j)) for j in range(m)])
    pivots = 0

    def guard():
        for array in (*rows, *transform, rhs, *a, b, c, lower):
            if any(
                max(value.numerator.bit_length(), value.denominator.bit_length())
                > max_bits
                for value in array
            ):
                raise ExactLPCertificationError(
                    "rational LP exceeds its exact-arithmetic bit guard"
                )

    def pivot(leaving, entering):
        nonlocal pivots
        if pivots >= max_pivots:
            raise ExactLPCertificationError("rational LP exceeds its pivot guard")
        pivots += 1
        factor = rows[leaving][entering]
        rows[leaving] = [value / factor for value in rows[leaving]]
        rhs[leaving] /= factor
        transform[leaving] = [value / factor for value in transform[leaving]]
        for i in range(len(rows)):
            if i == leaving or not rows[i][entering]:
                continue
            amount = rows[i][entering]
            rows[i] = [
                x - amount * y for x, y in zip(rows[i], rows[leaving], strict=True)
            ]
            rhs[i] -= amount * rhs[leaving]
            transform[i] = [
                x - amount * y
                for x, y in zip(transform[i], transform[leaving], strict=True)
            ]
        basis[leaving] = entering
        guard()

    def simplex(cost):
        while True:
            basic_cost = [cost[j] for j in basis]
            reduced = [
                cost[j] - _dot(basic_cost, [row[j] for row in rows])
                for j in range(len(cost))
            ]
            entering = next(
                (j for j, value in enumerate(reduced) if j not in basis and value < 0),
                None,
            )
            if entering is None:
                return None
            eligible = [i for i, row in enumerate(rows) if row[entering] > 0]
            if not eligible:
                return entering
            leaving = min(
                eligible, key=lambda i: (rhs[i] / rows[i][entering], basis[i])
            )
            pivot(leaving, entering)

    def dual(cost):
        return tuple(
            signs[j]
            * sum(
                (cost[basis[i]] * transform[i][j] for i in range(len(rows))), Fraction()
            )
            for j in range(m)
        )

    def finish(status, cost, ray=None):
        x = None
        value = None
        if status != "infeasible":
            solution = [Fraction()] * len(cost)
            for i, basic in enumerate(basis):
                solution[basic] = rhs[i]
            x = tuple(solution[j] + lower[j] for j in range(n))
            value = _dot(c, x) if status == "optimal" else None
        result = ExactLPResult(status, x, value, dual(cost), ray, pivots)
        _verify(c, a, b, lower, original_a, original_b, upper, result)
        return result

    guard()
    phase_one = [Fraction(int(j >= real_count)) for j in range(width)]
    if simplex(phase_one) is not None:
        raise ExactLPCertificationError(
            "nonnegative phase-I objective unexpectedly unbounded"
        )
    phase_one_value = sum(
        (phase_one[basic] * rhs[i] for i, basic in enumerate(basis)), Fraction()
    )
    if phase_one_value > 0:
        return finish("infeasible", phase_one)
    if phase_one_value < 0:
        raise ExactLPCertificationError("negative phase-I artificial mass")

    # Zero basic artificials are pivoted out. A row with no real nonbasic
    # coefficient and zero RHS is redundant and may be removed exactly.
    for i in reversed(range(len(rows))):
        if basis[i] < real_count:
            continue
        entering = next(
            (j for j in range(real_count) if j not in basis and rows[i][j]), None
        )
        if entering is not None:
            pivot(i, entering)
        else:
            if rhs[i]:
                raise ExactLPCertificationError("nonzero redundant phase-I row")
            del rows[i], rhs[i], basis[i], transform[i]
    rows[:] = [row[:real_count] for row in rows]
    phase_two = list(c) + [Fraction()] * m
    entering = simplex(phase_two)
    if entering is not None:
        direction = [Fraction()] * real_count
        direction[entering] = Fraction(1)
        for i, basic in enumerate(basis):
            direction[basic] = -rows[i][entering]
        return finish("unbounded", phase_two, tuple(direction[:n]))
    return finish("optimal", phase_two)
