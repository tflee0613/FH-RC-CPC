"""Dependency-free response metrics; windows are half-open sample intervals."""

import math
from collections.abc import Sequence


def pulse_windows(
    length: int, onset: float, duration: float
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return floor/ceil pulse bounds and an equally long available recovery window."""
    if isinstance(length, bool) or not isinstance(length, int) or length < 2:
        raise ValueError("length must be an integer of at least two samples")
    if not all(math.isfinite(x) for x in (onset, duration)) or not (
        0 <= onset < 1 and 0 < duration <= 1
    ):
        raise ValueError("onset must be in [0, 1); duration must be in (0, 1]")
    start = math.floor(length * onset)
    stop = min(length, max(start + 1, math.ceil(length * (onset + duration))))
    recovery_stop = min(length, stop + stop - start)
    if stop == recovery_stop:
        raise ValueError("pulse leaves no recovery samples")
    return (start, stop), (stop, recovery_stop)


def phase_metrics(
    outputs: Sequence[Sequence[float]],
    references: Sequence[Sequence[float]],
    spans: Sequence[float],
    pulse: tuple[int, int],
    recovery: tuple[int, int],
) -> dict[str, float]:
    """Mean worst-channel normalized error per phase; dynamic error is their maximum."""
    if not len(outputs) or len(outputs) != len(references) or not len(spans):
        raise ValueError("nonempty outputs and references must have the same length")
    if any(not math.isfinite(x) or x <= 0 for x in spans):
        raise ValueError("output spans must be finite and positive")
    for start, stop in (pulse, recovery):
        if any(
            isinstance(x, bool) or not isinstance(x, int) for x in (start, stop)
        ) or not 0 <= start < stop <= len(outputs):
            raise ValueError(
                "windows must be nonempty integer intervals within the trajectory"
            )
    if pulse[1] > recovery[0]:
        raise ValueError("recovery must follow the pulse without overlap")
    errors = []
    for output, reference in zip(outputs, references):
        if len(output) != len(spans) or len(reference) != len(spans):
            raise ValueError("channel counts must match output spans")
        if any(not math.isfinite(x) for row in (output, reference) for x in row):
            raise ValueError("outputs and references must be finite")
        errors.append(
            max(abs(y - r) / scale for y, r, scale in zip(output, reference, spans))
        )
    pulse_error = sum(errors[pulse[0] : pulse[1]]) / (pulse[1] - pulse[0])
    recovery_error = sum(errors[recovery[0] : recovery[1]]) / (
        recovery[1] - recovery[0]
    )
    if not all(math.isfinite(value) for value in (pulse_error, recovery_error)):
        raise ValueError("response metric arithmetic overflow")
    return {
        "pulse_error": pulse_error,
        "recovery_error": recovery_error,
        "dynamic_error": max(pulse_error, recovery_error),
    }
