"""Objective functions for Bayesian optimization based on finlab report objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tests.oscar_tw_strategy.utils.custom_report_metrics import (
    get_metrics_with_fixed_annual_return,
)


class ObjectiveName(str, Enum):
    SHARPE = "train_sharpe"
    ANNUAL_RETURN = "train_annual_return"
    CALMAR = "train_calmar"


@dataclass(frozen=True)
class ObjectiveResult:
    name: ObjectiveName
    value: float | None


class BaseReportObjective:
    """Base class for objective functions that score a finlab report."""

    name: ObjectiveName

    def evaluate(self, report, start_date: str | None = None, end_date: str | None = None) -> ObjectiveResult:
        raise NotImplementedError


class SharpeObjective(BaseReportObjective):
    name = ObjectiveName.SHARPE

    def evaluate(self, report, start_date: str | None = None, end_date: str | None = None) -> ObjectiveResult:
        metrics = report.get_metrics()
        value = metrics.get("ratio", {}).get("sharpeRatio")
        try:
            value = float(value) if value is not None else None
        except (TypeError, ValueError):
            value = None
        return ObjectiveResult(name=self.name, value=value)


class AnnualReturnObjective(BaseReportObjective):
    name = ObjectiveName.ANNUAL_RETURN

    def evaluate(self, report, start_date: str | None = None, end_date: str | None = None) -> ObjectiveResult:
        metrics = get_metrics_with_fixed_annual_return(report, start_date=start_date, end_date=end_date)
        value = metrics.get("profitability", {}).get("annualReturn")
        try:
            value = float(value) if value is not None else None
        except (TypeError, ValueError):
            value = None
        return ObjectiveResult(name=self.name, value=value)


class CalmarObjective(BaseReportObjective):
    name = ObjectiveName.CALMAR

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


_OBJECTIVES: dict[ObjectiveName, type[BaseReportObjective]] = {
    ObjectiveName.SHARPE: SharpeObjective,
    ObjectiveName.ANNUAL_RETURN: AnnualReturnObjective,
    ObjectiveName.CALMAR: CalmarObjective,
}


def build_objective(name: str | ObjectiveName) -> BaseReportObjective:
    try:
        key = ObjectiveName(name)
    except ValueError:
        allowed = [e.value for e in ObjectiveName]
        raise ValueError(f"Unsupported objective '{name}'. Allowed: {', '.join(allowed)}")
    return _OBJECTIVES[key]()
