"""Custom report metric helpers for annual-return anchoring."""

from __future__ import annotations

import pandas as pd


def _align_timestamp(ts: pd.Timestamp, index_tz) -> pd.Timestamp:
    """Align timestamp timezone with target index timezone."""
    if index_tz is None:
        if ts.tzinfo is not None:
            return ts.tz_convert(None).tz_localize(None)
        return ts

    if ts.tzinfo is None:
        return ts.tz_localize(index_tz)

    return ts.tz_convert(index_tz)


def compute_annual_return_from_creturn(
    creturn: pd.Series,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> float | None:
    """Compute annual return using an explicit backtest window on creturn."""
    if creturn is None or len(creturn) == 0:
        return None

    creturn = creturn.dropna()
    if len(creturn) == 0:
        return None

    start_ts = pd.Timestamp(start_date) if start_date is not None else creturn.index[0]
    end_ts = pd.Timestamp(end_date) if end_date is not None else creturn.index[-1]

    start_ts = _align_timestamp(start_ts, creturn.index.tz)
    end_ts = _align_timestamp(end_ts, creturn.index.tz)

    if end_ts < start_ts:
        return None

    idx = creturn.index
    first_idx = idx[0]
    last_idx = idx[-1]

    if end_ts < first_idx:
        return None

    if end_ts > last_idx:
        end_ts = last_idx

    end_pos = idx.searchsorted(end_ts, side="right") - 1
    if end_pos < 0:
        return None

    # Important: if requested start_date is earlier than the first creturn point,
    # treat that pre-trade window as cash (equity=1.0) instead of shrinking the period.
    if start_ts < first_idx:
        start_anchor_ts = start_ts
        start_value = 1.0
    else:
        start_pos = idx.searchsorted(start_ts, side="left")
        if start_pos >= len(creturn) or end_pos < start_pos:
            return None
        start_anchor_ts = idx[start_pos]
        start_value = float(creturn.iloc[start_pos])

    end_value = float(creturn.iloc[end_pos])

    if start_value <= 0:
        return None

    total_return = (end_value / start_value) - 1.0
    elapsed_days = max((idx[end_pos] - start_anchor_ts).days, 1)
    return (1.0 + total_return) ** (365.25 / elapsed_days) - 1.0


def compute_total_return_annualized(
    creturn: pd.Series,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> float | None:
    """Annualized return using the actual first/last creturn values within [start_date, end_date]."""
    if creturn is None or len(creturn) == 0:
        return None

    creturn = creturn.dropna()
    if len(creturn) == 0:
        return None

    sl = creturn
    if start_date is not None:
        start_ts = _align_timestamp(pd.Timestamp(start_date), creturn.index.tz)
        sl = sl[sl.index >= start_ts]
    if end_date is not None:
        end_ts = _align_timestamp(pd.Timestamp(end_date), creturn.index.tz)
        sl = sl[sl.index <= end_ts]

    if len(sl) < 2:
        return None

    total_return = float(sl.iloc[-1]) / float(sl.iloc[0]) - 1.0
    elapsed_days = max((sl.index[-1] - sl.index[0]).days, 1)
    return (1.0 + total_return) ** (365.25 / elapsed_days) - 1.0


def get_metrics_with_fixed_annual_return(
    report,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> dict:
    """Return report metrics with annualReturn corrected to requested window."""
    metrics = report.get_metrics()
    creturn = getattr(report, "creturn", None)
    corrected = compute_annual_return_from_creturn(
        creturn=creturn,
        start_date=start_date,
        end_date=end_date,
    )

    if corrected is not None:
        profitability = metrics.get("profitability", {})
        profitability["annualReturn"] = corrected
        metrics["profitability"] = profitability

    return metrics
