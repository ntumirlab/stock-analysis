"""Replay top-k trial parameters from early folds on post-2023 data.

Usage:
python -m tests.oscar_tw_strategy.replay_early_fold_topk_trials \
  --experiment-dir assets/OscarTWStrategy/composite_walk_forward/exp_train_val_test_v1 \
  --replay-start-date 2023-01-01 \
  --val-cutoff-date 2022-12-31 \
  --top-k 5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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
    summary = pd.read_csv(summary_path)
    summary["val_end"] = pd.to_datetime(summary["val_end"])
    summary["train_start"] = pd.to_datetime(summary["train_start"])
    summary["train_end"] = pd.to_datetime(summary["train_end"])
    summary["test_end"] = pd.to_datetime(summary["test_end"])
    return summary


def _select_eligible_folds(summary_df: pd.DataFrame, val_cutoff_date: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(val_cutoff_date)
    selected = summary_df.loc[summary_df["val_end"] <= cutoff].copy()
    if selected.empty:
        raise ValueError(f"No folds found with val_end <= {val_cutoff_date}.")
    return selected.sort_values("fold_id")


def _resolve_replay_end_date(strategy: OscarCompositeStrategy, replay_start_date: str) -> pd.Timestamp:
    market_data = strategy._load_market_data()
    close = market_data["close"]
    replay_start = pd.Timestamp(replay_start_date)
    available = close.index[close.index >= replay_start]
    if len(available) == 0:
        raise ValueError(f"No market data available after {replay_start_date}.")
    return pd.Timestamp(available[-1]).normalize()


def _load_topk_trials(trials_csv: Path, top_k: int) -> pd.DataFrame:
    trials_df = pd.read_csv(trials_csv)
    trials_df = trials_df.loc[trials_df["state"] == "TrialState.COMPLETE"].copy()
    if "trial_status" in trials_df.columns:
        trials_df = trials_df.loc[trials_df["trial_status"].fillna("ok") == "ok"].copy()

    trials_df["value"] = pd.to_numeric(trials_df["value"], errors="coerce")
    trials_df = trials_df.dropna(subset=["value", "params"])
    trials_df = trials_df.sort_values("value", ascending=False).head(max(1, int(top_k))).copy()
    if trials_df.empty:
        raise ValueError(f"No eligible completed trials found in {trials_csv}")
    return trials_df


def _build_params_from_trial_params_json(params_json: str) -> OscarCompositeParams:
    raw = json.loads(params_json)

    w_sar = float(raw["weight_sar"])
    w_macd = float(raw["weight_macd"])
    w_volume = float(raw["weight_volume"])
    w_inst = float(raw["weight_institutional"])
    weight_sum = max(1e-12, w_sar + w_macd + w_volume + w_inst)

    payload = {
        "sar_params": {
            "acceleration": float(raw["sar_acceleration"]),
            "maximum": float(raw["sar_maximum"]),
        },
        "macd_params": {
            "fastperiod": int(raw["macd_fast"]),
            "slowperiod": int(raw["macd_slow"]),
            "signalperiod": int(raw["macd_signal"]),
        },
        "volume_above_avg_ratio": float(raw["volume_above_avg_ratio"]),
        "min_avg_volume_30": float(raw["min_avg_volume_30"]),
        "new_high_ratio_120": float(raw["new_high_ratio_120"]),
        "sar_signal_lag_max": int(raw["sar_signal_lag_max"]),
        "sar_event_decay_alpha": float(raw["sar_event_decay_alpha"]),
        "macd_signal_lag_max": int(raw["macd_signal_lag_max"]),
        "macd_event_decay_alpha": float(raw["macd_event_decay_alpha"]),
        "max_volume_spike_ratio": float(raw["max_volume_spike_ratio"]),
        "signal_weights": {
            "sar": w_sar / weight_sum,
            "macd": w_macd / weight_sum,
            "volume": w_volume / weight_sum,
            "institutional": w_inst / weight_sum,
        },
        "buy_score_threshold": float(raw["buy_score_threshold"]),
        "sell_score_threshold": float(raw["sell_score_threshold"]),
        "signal_quantile_bins": {
            "sar": int(raw["bins_sar"]),
            "macd": int(raw["bins_macd"]),
            "volume": int(raw["bins_volume"]),
            "institutional": int(raw["bins_institutional"]),
        },
        "sar_proximity_weight": float(raw["sar_proximity_weight"]),
        "sar_event_weight": float(raw["sar_event_weight"]),
        "macd_proximity_weight": float(raw["macd_proximity_weight"]),
        "macd_event_weight": float(raw["macd_event_weight"]),
        "sar_near_sigmoid_slope": float(raw["sar_near_sigmoid_slope"]),
        "sar_near_distance_scale": float(raw["sar_near_distance_scale"]),
        "sar_event_sigmoid_slope": float(raw["sar_event_sigmoid_slope"]),
        "sar_event_distance_scale": float(raw["sar_event_distance_scale"]),
        "sar_history_lookback": int(raw["sar_history_lookback"]),
        "sar_history_decay_alpha": float(raw["sar_history_decay_alpha"]),
        "macd_near_sigmoid_slope": float(raw["macd_near_sigmoid_slope"]),
        "macd_near_distance_scale": float(raw["macd_near_distance_scale"]),
        "macd_event_sigmoid_slope": float(raw["macd_event_sigmoid_slope"]),
        "macd_event_distance_scale": float(raw["macd_event_distance_scale"]),
        "macd_history_lookback": int(raw["macd_history_lookback"]),
        "macd_history_decay_alpha": float(raw["macd_history_decay_alpha"]),
    }
    return OscarCompositeParams(**payload)


def run_replay_topk(
    experiment_dir: str,
    replay_start_date: str,
    val_cutoff_date: str,
    config_path: str,
    output_dir: str,
    smooth_span: int,
    initial_capital: float,
    top_k: int,
) -> Path:
    experiment_path = (REPO_ROOT / experiment_dir).resolve() if not Path(experiment_dir).is_absolute() else Path(experiment_dir)
    summary_df = _load_summary(experiment_path)
    eligible_folds = _select_eligible_folds(summary_df, val_cutoff_date)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (REPO_ROOT / output_dir / f"replay_topk_{run_tag}").resolve()
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

        bayes_dir = REPO_ROOT / str(row.bayes_dir)
        trials_csv = bayes_dir / "trials.csv"
        top_trials = _load_topk_trials(trials_csv, top_k=top_k)

        fold_cumret_df = pd.DataFrame()

        for rank_idx, trial in enumerate(top_trials.itertuples(index=False), start=1):
            params = _build_params_from_trial_params_json(trial.params)
            strategy = OscarCompositeStrategy(config_path=config_path, market_data=shared_market_data)
            report = strategy.run_strategy(params=params, start_date=replay_start_date)

            trial_label = f"fold{int(row.fold_id)}_r{rank_idx}_t{int(trial.number)}"
            trial_dir = fold_out / f"rank_{rank_idx:02d}_trial_{int(trial.number):03d}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            report_path = trial_dir / "replay_report.html"
            report.display(save_report_path=str(report_path))

            creturn = _extract_creturn_series(report)
            creturn = creturn.loc[replay_start_date:str(replay_end.date())]
            cumulative_df[trial_label] = creturn
            fold_cumret_df[trial_label] = creturn
            creturn.to_csv(trial_dir / "creturn.csv", encoding="utf-8-sig", header=[trial_label])

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

            replay_payload = {
                "source_fold_id": int(row.fold_id),
                "source_fold_name": fold_name,
                "source_train_start": str(row.train_start.date()),
                "source_train_end": str(row.train_end.date()),
                "source_val_end": str(row.val_end.date()),
                "source_trial_rank": rank_idx,
                "source_trial_number": int(trial.number),
                "source_trial_value": float(trial.value),
                "source_validation_sharpe": float(trial.validation_sharpe) if not pd.isna(trial.validation_sharpe) else None,
                "replay_start": replay_start_date,
                "replay_end": str(replay_end.date()),
                "total_reward_amount": total_reward,
                "metrics": metrics,
                "params": asdict(params),
            }
            with open(trial_dir / "replay_metrics.json", "w", encoding="utf-8") as f:
                json.dump(_to_json_safe(replay_payload), f, ensure_ascii=False, indent=2)

            profitability = metrics.get("profitability", {})
            ratio = metrics.get("ratio", {})
            risk = metrics.get("risk", {})
            summary_rows.append(
                {
                    "source_fold_id": int(row.fold_id),
                    "source_fold_name": fold_name,
                    "source_val_end": str(row.val_end.date()),
                    "source_trial_rank": rank_idx,
                    "source_trial_number": int(trial.number),
                    "source_trial_value": float(trial.value),
                    "source_validation_sharpe": float(trial.validation_sharpe) if not pd.isna(trial.validation_sharpe) else None,
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

        fold_cumret_df = fold_cumret_df.sort_index().ffill()
        fold_cumret_df.to_csv(fold_out / "fold_replay_cumret.csv", encoding="utf-8-sig")
        _plot_decile_cumulative(
            cumret_df=fold_cumret_df,
            output_png=fold_out / "fold_replay_cumret.png",
            title=f"{fold_name} Top-{top_k} Trial Replay ({replay_start_date}~{replay_end.date()})",
            smooth_span=smooth_span,
        )

    cumulative_df = cumulative_df.sort_index().ffill()
    cumulative_df.to_csv(out_root / "replay_topk_cumret.csv", encoding="utf-8-sig")
    _plot_decile_cumulative(
        cumret_df=cumulative_df,
        output_png=out_root / "replay_topk_cumret.png",
        title=(
            f"Top-{top_k} Trial Replay from {replay_start_date} to {replay_end.date()} "
            f"(val_end <= {val_cutoff_date})"
        ),
        smooth_span=smooth_span,
    )

    summary_out = pd.DataFrame(summary_rows)
    summary_out = summary_out.sort_values(["source_fold_id", "source_trial_rank"])
    summary_out.to_csv(out_root / "replay_topk_summary.csv", index=False, encoding="utf-8-sig")
    return out_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay top-k trials from early folds.")
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
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = run_replay_topk(
        experiment_dir=args.experiment_dir,
        replay_start_date=args.replay_start_date,
        val_cutoff_date=args.val_cutoff_date,
        config_path=args.config_path,
        output_dir=args.output_dir,
        smooth_span=args.smooth_span,
        initial_capital=args.initial_capital,
        top_k=args.top_k,
    )
    print(out_root)


if __name__ == "__main__":
    main()