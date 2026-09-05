"""Failure-hypergraph construction, lossless reductions and exact selection."""

from .behavior_quotient import BehaviorQuotient, quotient_controller_behavior
from .failure_hypergraph import FailureHypergraph
from .general_portfolio import (
    GeneralPortfolioResult,
    PortfolioInfeasible,
    greedy_group_constrained_portfolio,
    solve_group_constrained_portfolio,
)
from .signature_compression import (
    FailureSignatureTable,
    compress_failure_signatures,
    evaluate_signature_portfolio,
    evaluate_task_portfolio,
)
from .signature_solver import (
    greedy_signature_portfolio,
    solve_signature_portfolio,
    solve_signature_portfolio_by_enumeration,
)

__all__ = [
    "BehaviorQuotient",
    "FailureHypergraph",
    "FailureSignatureTable",
    "GeneralPortfolioResult",
    "PortfolioInfeasible",
    "compress_failure_signatures",
    "evaluate_signature_portfolio",
    "evaluate_task_portfolio",
    "greedy_group_constrained_portfolio",
    "greedy_signature_portfolio",
    "quotient_controller_behavior",
    "solve_group_constrained_portfolio",
    "solve_signature_portfolio",
    "solve_signature_portfolio_by_enumeration",
]
