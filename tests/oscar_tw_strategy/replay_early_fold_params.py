"""Replay early walk-forward best parameters on a later out-of-sample period.

Usage:
python -m tests.oscar_tw_strategy.replay_early_fold_params \
  --experiment-dir assets/OscarTWStrategy/composite_walk_forward/exp_train_val_test_v1 \
  --replay-start-date 2023-01-01 \
  --val-cutoff-date 2022-12-31
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from strategy_class.oscar.oscar_strategy_composite import OscarCompositeStrategy
from strategy_class.oscar.oscar_strategy_composite_params import OscarCompositeParams
from tests.oscar_tw_strategy.utils.custom_report_metrics import (
    compute_total_reward_amount_from_creturn,
    get_metrics_with_fixed_annual_return,
)
from tests.oscar_tw_strategy.walk_forward_composite_validation import (
    _extract_creturn_series,
    _plot_decile_cumulative,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _to_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) and not isinstance(value, (str, bytes)):
        return None
    return value


def _load_summary(experiment_dir: Path) -> pd.DataFrame:
    summary_path = experiment_dir / "walk_forward_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    return pd.read_csv(summary_path)


def _select_eligible_folds(summary_df: pd.DataFrame, val_cutoff_date: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(val_cutoff_date)
    summary_df = summary_df.copy()
    summary_df["val_end"] = pd.to_datetime(summary_df["val_end"])
    summary_df["train_start"] = pd.to_datetime(summary_df["train_start"])
    summary_df["train_end"] = pd.to_datetime(summary_df["train_end"])
    summary_df["test_start"] = pd.to_datetime(summary_df["test_start"])
    summary_df["test_end"] = pd.to_datetime(summary_df["test_end"])
    selected = summary_df.loc[summary_df["val_end"] <= cutoff].copy()
    if selected.empty:
        raise ValueError(f"No folds found with val_end <= {val_cutoff_date}.")
    return selected.sort_values("fold_id")


def _load_fold_best_params(experiment_dir: Path, fold_name: str) -> OscarCompositeParams:
    metrics_path = experiment_dir / fold_name / "test_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return OscarCompositeParams(**payload["best_params"])


def _resolve_replay_end_date(strategy: OscarCompositeStrategy, replay_start_date: str) -> pd.Timestamp:
    market_data = strategy._load_market_data()
    close = market_data["close"]
    replay_start = pd.Timestamp(replay_start_date)
    available = close.index[close.index >= replay_start]
    if len(available) == 0:
        raise ValueError(f"No market data available after {replay_start_date}.")
    return pd.Timestamp(available[-1]).normalize()


def run_replay(
    experiment_dir: str,
    replay_start_date: str,
    val_cutoff_date: str,
    config_path: str,
    output_dir: str,
    smooth_span: int,
    initial_capital: float,
) -> Path:
    experiment_path = (REPO_ROOT / experiment_dir).resolve() if not Path(experiment_dir).is_absolute() else Path(experiment_dir)
    summary_df = _load_summary(experiment_path)
    eligible_folds = _select_eligible_folds(summary_df, val_cutoff_date)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (REPO_ROOT / output_dir / f"replay_{run_tag}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    bootstrap = OscarCompositeStrategy(config_path=config_path)
    shared_market_data = bootstrap._load_market_data()
    replay_end = _resolve_replay_end_date(bootstrap, replay_start_date)

    cumulative_df = pd.DataFrame()
    summary_rows: list[dict] = []

    for row in eligible_folds.itertuples(index=False):
        fold_name = f"fold_{row.train_start:%Y%m%d}_{row.test_end:%Y%m%d}"
        fold_out = out_root / fold_name
        fold_out.mkdir(parents=True, exist_ok=True)

        params = _load_fold_best_params(experiment_path, fold_name)
        strategy = OscarCompositeStrategy(config_path=config_path, market_data=shared_market_data)
        report = strategy.run_strategy(params=params, start_date=replay_start_date)

        report_path = fold_out / "replay_report.html"
        report.display(save_report_path=str(report_path))

        creturn = _extract_creturn_series(report)
        creturn = creturn.loc[replay_start_date:str(replay_end.date())]
        label = f"fold_{int(row.fold_id)}"
        cumulative_df[label] = creturn
        creturn.to_csv(fold_out / "creturn.csv", encoding="utf-8-sig", header=[label])

        metrics = get_metrics_with_fixed_annual_return(
            report,
            start_date=replay_start_date,
            end_date=str(replay_end.date()),
        )
        total_reward = compute_total_reward_amount_from_creturn(
            creturn=getattr(report, "creturn", None),
            initial_capital=initial_capital,
            start_date=replay_start_date,
            end_date=str(replay_end.date()),
        )

        payload = {
            "source_fold_id": int(row.fold_id),
            "source_train_start": str(row.train_start.date()),
            "source_train_end": str(row.train_end.date()),
            "source_val_end": str(row.val_end.date()),
            "replay_start": replay_start_date,
            "replay_end": str(replay_end.date()),
            "total_reward_amount": total_reward,
            "metrics": metrics,
            "best_params": asdict(params),
        }
        with open(fold_out / "replay_metrics.json", "w", encoding="utf-8") as f:
            json.dump(_to_json_safe(payload), f, ensure_ascii=False, indent=2)

        profitability = metrics.get("profitability", {})
        ratio = metrics.get("ratio", {})
        risk = metrics.get("risk", {})
        summary_rows.append(
            {
                "source_fold_id": int(row.fold_id),
                "source_fold_name": fold_name,
                "source_val_end": str(row.val_end.date()),
                "replay_start": replay_start_date,
                "replay_end": str(replay_end.date()),
                "total_reward_amount": total_reward,
                "annual_return": profitability.get("annualReturn"),
                "sharpe_ratio": ratio.get("sharpeRatio"),
                "max_drawdown": risk.get("maxDrawdown"),
                "avg_n_stock": profitability.get("avgNStock"),
                "report_path": str(report_path.relative_to(REPO_ROOT)),
            }
        )

    cumulative_df = cumulative_df.sort_index().ffill()
    cumulative_df.to_csv(out_root / "replay_cumret.csv", encoding="utf-8-sig")
    _plot_decile_cumulative(
        cumret_df=cumulative_df,
        output_png=out_root / "replay_cumret.png",
        title=(
            f"Replay Early-Fold Params from {replay_start_date} to {replay_end.date()} "
            f"(val_end <= {val_cutoff_date})"
        ),
        smooth_span=smooth_span,
    )

    summary_out = pd.DataFrame(summary_rows).sort_values("source_fold_id")
    summary_out.to_csv(out_root / "replay_summary.csv", index=False, encoding="utf-8-sig")

    return out_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay early-fold best params on post-2023 data.")
    parser.add_argument(
        "--experiment-dir",
        default="assets/OscarTWStrategy/composite_walk_forward/exp_train_val_test_v1",
        help="Walk-forward experiment directory.",
    )
    parser.add_argument("--replay-start-date", default="2023-01-01")
    parser.add_argument("--val-cutoff-date", default="2022-12-31")
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument(
        "--output-dir",
        default="assets/OscarTWStrategy/composite_param_replay",
        help="Output base directory for replay analysis.",
    )
    parser.add_argument("--smooth-span", type=int, default=8)
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = run_replay(
        experiment_dir=args.experiment_dir,
        replay_start_date=args.replay_start_date,
        val_cutoff_date=args.val_cutoff_date,
        config_path=args.config_path,
        output_dir=args.output_dir,
        smooth_span=args.smooth_span,
        initial_capital=args.initial_capital,
    )
    print(out_root)


if __name__ == "__main__":
    main()