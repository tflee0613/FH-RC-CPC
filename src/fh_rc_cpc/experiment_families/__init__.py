"""Analytic and seeded synthetic experiment families reported in the paper."""

from .branching_cases import run_branching_stress_suite
from .fractional_triangle import benchmark_fractional_triangle_instance
from .scaling_grid import benchmark_registered_diagnostic_grid

__all__ = [
    "benchmark_fractional_triangle_instance",
    "benchmark_registered_diagnostic_grid",
    "run_branching_stress_suite",
]
