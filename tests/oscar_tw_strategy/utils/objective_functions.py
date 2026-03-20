"""Objective functions for Bayesian optimization based on finlab report objects."""

from __future__ import annotations

from dataclasses import dataclass

from tests.oscar_tw_strategy.utils.custom_report_metrics import (
    compute_total_reward_amount_from_creturn,
    get_metrics_with_fixed_annual_return,
)


@dataclass(frozen=True)
class ObjectiveResult:
    name: str
    value: float | None


class BaseReportObjective:
    """Base class for objective functions that score a finlab report."""

    name: str = "base"

    def evaluate(self, report, start_date: str | None = None, end_date: str | None = None) -> ObjectiveResult:
        raise NotImplementedError


class SharpeObjective(BaseReportObjective):
    name = "train_sharpe"

    def evaluate(self, report, start_date: str | None = None, end_date: str | None = None) -> ObjectiveResult:
        metrics = report.get_metrics()
        value = metrics.get("ratio", {}).get("sharpeRatio")
        try:
            value = float(value) if value is not None else None
        except (TypeError, ValueError):
            value = None
        return ObjectiveResult(name=self.name, value=value)


class AnnualReturnObjective(BaseReportObjective):
    name = "train_annual_return"

    def evaluate(self, report, start_date: str | None = None, end_date: str | None = None) -> ObjectiveResult:
        metrics = get_metrics_with_fixed_annual_return(report, start_date=start_date, end_date=end_date)
        value = metrics.get("profitability", {}).get("annualReturn")
        try:
            value = float(value) if value is not None else None
        except (TypeError, ValueError):
            value = None
        return ObjectiveResult(name=self.name, value=value)


class TotalRewardObjective(BaseReportObjective):
    name = "train_total_reward"

    def __init__(self, initial_capital: float):
        self.initial_capital = float(initial_capital)

    def evaluate(self, report, start_date: str | None = None, end_date: str | None = None) -> ObjectiveResult:
        value = compute_total_reward_amount_from_creturn(
            creturn=getattr(report, "creturn", None),
            initial_capital=self.initial_capital,
            start_date=start_date,
            end_date=end_date,
        )
        return ObjectiveResult(name=self.name, value=value)


class CalmarObjective(BaseReportObjective):
    name = "train_calmar"

    def evaluate(self, report, start_date: str | None = None, end_date: str | None = None) -> ObjectiveResult:
        metrics = get_metrics_with_fixed_annual_return(report, start_date=start_date, end_date=end_date)
        annual_return = metrics.get("profitability", {}).get("annualReturn")
        max_drawdown = metrics.get("risk", {}).get("maxDrawdown")
        try:
            annual_return = float(annual_return)
            max_drawdown = float(max_drawdown)
        except (TypeError, ValueError):
            return ObjectiveResult(name=self.name, value=None)

        if max_drawdown >= 0:
            return ObjectiveResult(name=self.name, value=None)

        return ObjectiveResult(name=self.name, value=annual_return / abs(max_drawdown))


def build_objective(name: str, initial_capital: float) -> BaseReportObjective:
    key = (name or "").strip().lower()

    if key == SharpeObjective.name:
        return SharpeObjective()
    if key == AnnualReturnObjective.name:
        return AnnualReturnObjective()
    if key == TotalRewardObjective.name:
        return TotalRewardObjective(initial_capital=initial_capital)
    if key == CalmarObjective.name:
        return CalmarObjective()

    allowed = [
        SharpeObjective.name,
        AnnualReturnObjective.name,
        TotalRewardObjective.name,
        CalmarObjective.name,
    ]
    raise ValueError(f"Unsupported objective '{name}'. Allowed: {', '.join(allowed)}")
