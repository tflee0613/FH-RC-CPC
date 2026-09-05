"""Checks for the public pulse/recovery retention query."""

from __future__ import annotations

import math
import os
from pathlib import Path

from fh_rc_cpc.response_query import query, verify
from fh_rc_cpc.response_query.metrics import phase_metrics, pulse_windows


def data_directory() -> Path:
    configured = os.environ.get("FH_RC_CPC_DATA")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1].parent / "Public_Result_Data"


def test_public_response_archive_reproduces_registered_frontier() -> None:
    result = verify(data_directory())
    assert result["tasks"] == 60
    assert result["supported_tasks"] == 59
    assert result["cells"] == 360
    assert result["representative_curve_rows"] == 600
    assert result["breakpoints"] == 246
    assert result["portfolio_checks"] == 15_498
    assert result["strict_optimum_masks"] == [63]
    assert result["relaxed_optimum_masks"] == [47]


def test_response_boundary_is_inclusive_for_the_reduced_portfolio() -> None:
    boundary = verify(data_directory())["boundary"]
    below = query(data_directory(), math.nextafter(boundary, 0.0))
    at = query(data_directory(), boundary)
    assert below["minimum_size"] == 6
    assert below["optimum_masks"] == [63]
    assert at["minimum_size"] == 5
    assert at["optimum_masks"] == [47]


def test_registered_phase_metric_uses_the_worst_output_channel() -> None:
    assert pulse_windows(100, 0.27, 0.23) == ((27, 50), (50, 73))
    result = phase_metrics(
        [[1, 4], [3, 2], [0, 0], [2, 2]],
        [[0, 0]] * 4,
        [1, 2],
        (0, 2),
        (2, 4),
    )
    assert result == {
        "pulse_error": 2.5,
        "recovery_error": 1.0,
        "dynamic_error": 2.5,
    }
