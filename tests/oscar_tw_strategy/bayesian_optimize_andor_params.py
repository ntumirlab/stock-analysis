"""Bayesian optimization for OscarAndOrStrategy.

Usage:
python -m tests.oscar_tw_strategy.bayesian_optimize_andor_params
python -m tests.oscar_tw_strategy.bayesian_optimize_andor_params --n-trials 1000 --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import pickle
from datetime import datetime
from pathlib import Path

import optuna
import pandas as pd
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from strategy_class.oscar.oscar_strategy_andor import OscarAndOrStrategy
from tests.oscar_tw_strategy.utils.custom_report_metrics import (
    compute_total_reward_amount_from_creturn,
)
from tests.oscar_tw_strategy.utils.objective_functions import build_objective

MIN_OBJECTIVE_VALUE = -1e18
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKET_DATA_PICKLE = REPO_ROOT / "finlab_db" / "workspace" / "oscar_andor_market_data.pkl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AndOrBayesianOptimizer:
    def __init__(
        self,
        config_path: str = "config.yaml",
        start_date: str = "2020-01-01",
        end_date: str | None = None,
        n_trials: int = 1000,
        workers: int = 1,
        seed: int = 42,
        output_dir: str = "assets/OscarTWStrategy/andor_bayesian_params",
        initial_capital: float = 100_000.0,
        fee_ratio: float = 0.001425,
        tax_ratio: float = 0.003,
        market_data: dict | None = None,
        market_data_pickle_path: str | None = None,
        storage_url: str | None = None,
        allow_market_data_fetch: bool = True,
        objective_name: str = "train_sharpe",
    ):
        self.config_path = config_path
        self.start_date = start_date
        self.end_date = end_date
        self.n_trials = int(n_trials)
        self.workers = max(1, int(workers))
        self.seed = int(seed)
        self.initial_capital = float(initial_capital)
        self.fee_ratio = float(fee_ratio)
        self.tax_ratio = float(tax_ratio)
        self.market_data_pickle_path = (
            Path(market_data_pickle_path) if market_data_pickle_path else DEFAULT_MARKET_DATA_PICKLE
        )
        self.storage_url = storage_url
        self.allow_market_data_fetch = bool(allow_market_data_fetch)
        self.objective_name = objective_name

        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.study_dir = Path(output_dir) / f"andor_{run_tag}"
        self.study_dir.mkdir(parents=True, exist_ok=True)

        self.market_data = market_data
        self.objective = build_objective(self.objective_name, initial_capital=self.initial_capital)

    def _load_market_data_once(self) -> dict:
        if self.market_data is not None:
            return self.market_data

        if not self.allow_market_data_fetch:
            with open(self.market_data_pickle_path, "rb") as f:
                self.market_data = pickle.load(f)
            return self.market_data

        self.market_data = OscarAndOrStrategy.load_market_data()
        return self.market_data

    def _persist_market_data_pickle(self, market_data: dict) -> None:
        self.market_data_pickle_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.market_data_pickle_path.with_suffix(
            self.market_data_pickle_path.suffix + ".tmp"
        )
        with open(tmp_path, "wb") as f:
            pickle.dump(market_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.market_data_pickle_path)

    def _resolve_storage_spec(self) -> dict | None:
        if self.storage_url:
            return {"type": "url", "value": self.storage_url}
        if self.workers > 1:
            return {
                "type": "journal",
                "value": str(self.study_dir / "optuna_journal.log"),
            }
        return None

    @staticmethod
    def _build_storage(storage_spec: dict | None):
        if storage_spec is None:
            return None
        if storage_spec["type"] == "url":
            return storage_spec["value"]
        if storage_spec["type"] == "journal":
            return JournalStorage(JournalFileBackend(storage_spec["value"]))
        raise ValueError(f"Unsupported storage spec: {storage_spec}")

    def _create_or_load_study(self, storage, study_name: str, seed: int):
        return optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=1),
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )

    def _optimize_study(self, storage, study_name: str, n_trials: int, seed: int) -> None:
        study = self._create_or_load_study(storage, study_name, seed)
        study.optimize(self._objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    @staticmethod
    def _snap_upper_to_step(lower: float, upper: float, step: float) -> float:
        if upper <= lower:
            return lower
        step_count = math.floor(((upper - lower) / step) + 1e-12)
        snapped = lower + (step_count * step)
        return round(snapped, 10)

    @staticmethod
    def _safe_float(value, default=None):
        if value is None:
            return default
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(value):
            return default
        return value

    def _suggest_strategy_kwargs(self, trial: optuna.Trial) -> dict:
        sar_acceleration = trial.suggest_float("sar_acceleration", 0.01, 0.06, step=0.005)
        sar_maximum = trial.suggest_float("sar_maximum", max(0.1, sar_acceleration * 2.0), 0.4, step=0.01)

        macd_fast = trial.suggest_int("macd_fast", 8, 20)
        macd_slow = trial.suggest_int("macd_slow", macd_fast + 4, 40)
        macd_signal = trial.suggest_int("macd_signal", 5, 15)

        strategy_kwargs = {
            "config_path": self.config_path,
            "market_data": self.market_data,
            "sar_params": {
                "acceleration": sar_acceleration,
                "maximum": sar_maximum,
            },
            "macd_params": {
                "fastperiod": macd_fast,
                "slowperiod": macd_slow,
                "signalperiod": macd_signal,
            },
            "volume_above_avg_ratio": trial.suggest_float("volume_above_avg_ratio", 0.1, 1.0, step=0.05),
            "new_high_ratio_120": trial.suggest_float("new_high_ratio_120", 0.10, 0.80, step=0.05),
            "min_avg_volume_30": trial.suggest_int("min_avg_volume_30", 500_000, 5_000_000, step=250_000),
            "max_volume_spike_ratio": trial.suggest_float("max_volume_spike_ratio", 3.0, 15.0, step=0.5),
            "sar_signal_lag_min": 0,
            "sar_signal_lag_max": trial.suggest_int("sar_signal_lag_max", 0, 10),
            "macd_signal_lag_min": 0,
            "macd_signal_lag_max": trial.suggest_int("macd_signal_lag_max", 0, 10),
        }
        return strategy_kwargs

    def _suggest_run_kwargs(self, trial: optuna.Trial) -> dict:
        return {
            "start_date": self.start_date,
            "fee_ratio": self.fee_ratio,
            "tax_ratio": self.tax_ratio,
            "sim_resample": "D",
            "max_stocks": trial.suggest_categorical("max_stocks", [None, 5, 10, 20, 50]),
        }

    def _objective(self, trial: optuna.Trial) -> float:
        try:
            strategy_kwargs = self._suggest_strategy_kwargs(trial)
            run_kwargs = self._suggest_run_kwargs(trial)

            strategy = OscarAndOrStrategy(**strategy_kwargs)
            report = strategy.run_strategy(**run_kwargs)

            total_reward = compute_total_reward_amount_from_creturn(
                creturn=getattr(report, "creturn", None),
                initial_capital=self.initial_capital,
                start_date=self.start_date,
                end_date=self.end_date,
            )

            metrics = report.get_metrics()
            sharpe_ratio = self._safe_float(metrics.get("ratio", {}).get("sharpeRatio", None), default=None)
            annual_return = self._safe_float(metrics.get("profitability", {}).get("annualReturn", None), default=None)
            max_drawdown = self._safe_float(metrics.get("risk", {}).get("maxDrawdown", None), default=None)

            objective_result = self.objective.evaluate(
                report,
                start_date=self.start_date,
                end_date=self.end_date,
            )
            objective_value = self._safe_float(objective_result.value, default=None)

            trial.set_user_attr("objective", objective_result.name)
            trial.set_user_attr("sharpe_ratio", sharpe_ratio)
            trial.set_user_attr("annual_return", annual_return)
            trial.set_user_attr("max_drawdown", max_drawdown)

            if objective_value is None:
                trial.set_user_attr("trial_status", "invalid_objective_nan")
                trial.set_user_attr("failure_reason", "objective_value is NaN/inf/None")
                trial.set_user_attr("objective_value", MIN_OBJECTIVE_VALUE)
                trial.set_user_attr("total_reward_amount", total_reward)
                logger.warning("Trial %s invalid objective (NaN/inf/None). params=%s", trial.number, trial.params)
                return MIN_OBJECTIVE_VALUE

            trial.set_user_attr("trial_status", "ok")
            trial.set_user_attr("failure_reason", None)
            trial.set_user_attr("objective_value", objective_value)
            trial.set_user_attr("total_reward_amount", total_reward)
            return objective_value
        except Exception as exc:
            trial.set_user_attr("trial_status", "exception")
            trial.set_user_attr("failure_reason", f"{type(exc).__name__}: {exc}")
            trial.set_user_attr("objective_value", MIN_OBJECTIVE_VALUE)
            trial.set_user_attr("total_reward_amount", None)
            logger.exception("Trial %s failed. params=%s", trial.number, trial.params)
            raise

    def run(self) -> Path:
        study_name = f"oscar_andor_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        market_data = self._load_market_data_once()
        self._persist_market_data_pickle(market_data)

        storage_spec = self._resolve_storage_spec()
        storage = self._build_storage(storage_spec)
        study = self._create_or_load_study(storage, study_name, self.seed)

        if self.workers <= 1:
            study.optimize(self._objective, n_trials=self.n_trials, n_jobs=1, show_progress_bar=True)
        else:
            worker_count = min(self.workers, self.n_trials)
            base_trials = self.n_trials // worker_count
            extra_trials = self.n_trials % worker_count
            trials_per_worker = [
                base_trials + (1 if i < extra_trials else 0)
                for i in range(worker_count)
            ]

            worker_payload = {
                "config_path": self.config_path,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "objective_name": self.objective_name,
                "output_dir": str(self.study_dir.parent),
                "initial_capital": self.initial_capital,
                "fee_ratio": self.fee_ratio,
                "tax_ratio": self.tax_ratio,
                "market_data_pickle_path": str(self.market_data_pickle_path),
            }

            processes = []
            for worker_id, worker_trials in enumerate(trials_per_worker):
                if worker_trials <= 0:
                    continue
                process = mp.Process(
                    target=_run_bayesian_worker,
                    args=(
                        worker_payload,
                        storage_spec,
                        study_name,
                        worker_trials,
                        self.seed + worker_id,
                    ),
                )
                process.start()
                processes.append(process)

            failed = False
            for process in processes:
                process.join()
                if process.exitcode != 0:
                    failed = True

            if failed:
                raise RuntimeError("One or more bayesian worker processes failed.")

            study = self._create_or_load_study(self._build_storage(storage_spec), study_name, self.seed)

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            raise RuntimeError("No completed trials.")

        best_trial = max(
            completed,
            key=lambda t: (
                float(t.user_attrs.get("objective_value", MIN_OBJECTIVE_VALUE)),
                float(t.user_attrs.get("sharpe_ratio", float("-inf")) or float("-inf")),
            ),
        )

        best_report = {
            "best_value": best_trial.user_attrs.get("objective_value"),
            "best_params": dict(best_trial.params),
            "best_user_attrs": best_trial.user_attrs,
            "config": {
                "config_path": self.config_path,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "objective_name": self.objective_name,
                "n_trials": self.n_trials,
                "workers": self.workers,
                "initial_capital": self.initial_capital,
                "fee_ratio": self.fee_ratio,
                "tax_ratio": self.tax_ratio,
            },
        }

        with open(self.study_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(best_report, f, ensure_ascii=False, indent=2)

        rows = []
        for t in study.trials:
            rows.append(
                {
                    "number": t.number,
                    "state": str(t.state),
                    "value": t.value,
                    "trial_status": t.user_attrs.get("trial_status"),
                    "failure_reason": t.user_attrs.get("failure_reason"),
                    "objective_value": t.user_attrs.get("objective_value"),
                    "total_reward_amount": t.user_attrs.get("total_reward_amount"),
                    "objective": t.user_attrs.get("objective"),
                    "sharpe_ratio": t.user_attrs.get("sharpe_ratio"),
                    "annual_return": t.user_attrs.get("annual_return"),
                    "max_drawdown": t.user_attrs.get("max_drawdown"),
                    "params": json.dumps(t.params, ensure_ascii=False, sort_keys=True),
                }
            )

        pd.DataFrame(rows).to_csv(self.study_dir / "trials.csv", index=False, encoding="utf-8-sig")
        try:
            pd.DataFrame(rows).to_html(self.study_dir / "trials.html", index=False)
        except Exception:
            pass

        return self.study_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bayesian optimization for OscarAndOrStrategy.")
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--objective-name",
        default="train_sharpe",
        choices=["train_sharpe", "train_annual_return", "train_total_reward", "train_calmar"],
        help="Objective function computed from finlab report on train window.",
    )
    parser.add_argument("--n-trials", type=int, default=1000, help="Number of optimization trials.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel process workers.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="assets/OscarTWStrategy/andor_bayesian_params")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument(
        "--market-data-pickle-path",
        default=str(DEFAULT_MARKET_DATA_PICKLE),
        help="Shared pickle path for market data. Parent writes once; workers read only.",
    )
    parser.add_argument(
        "--storage-url",
        default=None,
        help="Optional Optuna storage URL for robust parallel execution across processes/machines.",
    )
    return parser


def _run_bayesian_worker(worker_payload, storage_spec, study_name: str, n_trials: int, seed: int) -> None:
    executor = AndOrBayesianOptimizer(
        config_path=worker_payload["config_path"],
        start_date=worker_payload["start_date"],
        end_date=worker_payload["end_date"],
        objective_name=worker_payload.get("objective_name", "train_sharpe"),
        n_trials=n_trials,
        workers=1,
        seed=seed,
        output_dir=worker_payload["output_dir"],
        initial_capital=worker_payload["initial_capital"],
        fee_ratio=worker_payload["fee_ratio"],
        tax_ratio=worker_payload["tax_ratio"],
        market_data=None,
        market_data_pickle_path=worker_payload["market_data_pickle_path"],
        storage_url=None,
        allow_market_data_fetch=False,
    )
    executor._load_market_data_once()
    executor._optimize_study(executor._build_storage(storage_spec), study_name, n_trials, seed)


def main() -> None:
    args = _build_arg_parser().parse_args()

    optimizer = AndOrBayesianOptimizer(
        config_path=args.config_path,
        start_date=args.start_date,
        end_date=args.end_date,
        objective_name=args.objective_name,
        n_trials=args.n_trials,
        workers=args.workers,
        seed=args.seed,
        output_dir=args.output_dir,
        initial_capital=args.initial_capital,
        fee_ratio=0.001425,
        tax_ratio=0.003,
        market_data_pickle_path=args.market_data_pickle_path,
        storage_url=args.storage_url,
    )
    out = optimizer.run()
    print(f"Bayesian optimization completed. Outputs: {out}")


if __name__ == "__main__":
    main()
