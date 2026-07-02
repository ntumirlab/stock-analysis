"""Minimal JSON-serializable container for per-trial metadata in Bayesian optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from tests.oscar_tw_strategy.utils.custom_report_metrics import (
    compute_annual_return_from_creturn,
)


@dataclass
class TrialResult:
    """Stores all metadata produced by a single Optuna trial as JSON-safe primitives."""

    trial_status: str  # "ok" | "invalid_objective_nan" | "exception"
    failure_reason: str | None
    objective_name: str | None
    objective_value: float | None
    sharpe_ratio: float | None
    annual_return: float | None  # anchored to creturn.index[0] (first transaction)
    max_drawdown: float | None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_report(
        cls,
        report,
        objective_result,
        end_date: str | None = None,
    ) -> TrialResult:
        """Build a TrialResult from a finlab report and an ObjectiveResult."""
        creturn = getattr(report, "creturn", None)
        start_date = creturn.index[0] if creturn is not None and len(creturn) > 0 else None

        annual_return = compute_annual_return_from_creturn(
            creturn=creturn,
            start_date=start_date,
            end_date=end_date,
        )

        metrics = report.get_metrics()
        sharpe_ratio = _safe_float(metrics.get("ratio", {}).get("sharpeRatio"))
        max_drawdown = _safe_float(metrics.get("risk", {}).get("maxDrawdown"))

        objective_value = _safe_float(objective_result.value)
        if objective_value is None:
            return cls(
                trial_status="invalid_objective_nan",
                failure_reason="objective_value is NaN/inf/None",
                objective_name=objective_result.name,
                objective_value=None,
                sharpe_ratio=sharpe_ratio,
                annual_return=annual_return,
                max_drawdown=max_drawdown,
            )

        return cls(
            trial_status="ok",
            failure_reason=None,
            objective_name=objective_result.name,
            objective_value=objective_value,
            sharpe_ratio=sharpe_ratio,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
        )

    @classmethod
    def from_exception(cls, exc: Exception) -> TrialResult:
        return cls(
            trial_status="exception",
            failure_reason=f"{type(exc).__name__}: {exc}",
            objective_name=None,
            objective_value=None,
            sharpe_ratio=None,
            annual_return=None,
            max_drawdown=None,
        )

    # ------------------------------------------------------------------
    # Optuna integration
    # ------------------------------------------------------------------

    def apply_to_trial(self, trial) -> None:
        """Write all fields to the trial as individual user attrs."""
        for key, value in asdict(self).items():
            trial.set_user_attr(key, value)

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_float(value, default=None) -> float | None:
    """Convert value to float, returning default on failure or non-finite values."""
    import math
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default
