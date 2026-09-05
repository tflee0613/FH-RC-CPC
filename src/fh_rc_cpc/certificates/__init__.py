"""Exact cost, outcome, scope-composition and reserve certificates."""

from .algorithm_certificates import (
    build_signature_zeta_frontier,
    certify_continuous_auxiliary_projection,
    certify_cost_box_structure,
    certify_greedy_submodular_cover,
    certify_symmetric_cost_robustness_radius,
)
from .outcome_stability import (
    apply_row_change_witness,
    certify_outcome_stability,
    portfolio_failure_distance,
    portfolio_repair_distance,
    qualification_with_row_reserve,
    verify_row_change_witness,
)
from .scope_composition import ScopeContract, compose_scopes, evaluate_scope
from .severity_guard import (
    maximum_portfolio_loss,
    relative_score_losses,
    score_range_guard_scope,
)

__all__ = [
    "ScopeContract",
    "apply_row_change_witness",
    "build_signature_zeta_frontier",
    "certify_continuous_auxiliary_projection",
    "certify_cost_box_structure",
    "certify_greedy_submodular_cover",
    "certify_outcome_stability",
    "certify_symmetric_cost_robustness_radius",
    "compose_scopes",
    "evaluate_scope",
    "maximum_portfolio_loss",
    "portfolio_failure_distance",
    "portfolio_repair_distance",
    "qualification_with_row_reserve",
    "relative_score_losses",
    "score_range_guard_scope",
    "verify_row_change_witness",
]
