"""Bayesian optimization for OscarCompositeStrategy using OscarCompositeParams.

Usage:
python -m tests.oscar_tw_strategy.bayesian_optimize_composite_params
python -m tests.oscar_tw_strategy.bayesian_optimize_composite_params --n-trials 1000 --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import math
import os
import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import optuna
import pandas as pd
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from strategy_class.oscar.oscar_strategy_composite import OscarCompositeStrategy
from strategy_class.oscar.oscar_strategy_composite_params import OscarCompositeParams
from tests.oscar_tw_strategy.utils.objective_functions import ObjectiveName, build_objective
from tests.oscar_tw_strategy.utils.trial_result import TrialResult

MIN_OBJECTIVE_VALUE = -1e18
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKET_DATA_PICKLE = REPO_ROOT / "finlab_db" / "workspace" / "oscar_composite_market_data.pkl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CompositeBayesianOptimizer:
    def __init__(
        self,
        config_path: str = "config.yaml",
        start_date: str = "2020-01-01",
        end_date: str | None = None,
        n_trials: int = 1000,
        workers: int = 1,
        seed: int = 42,
        output_dir: str = "assets/OscarTWStrategy/composite_bayesian_params",
        fee_ratio: float = 0.001425,
        tax_ratio: float = 0.003,
        market_data: dict | None = None,
        market_data_pickle_path: str | None = None,
        allow_market_data_fetch: bool = True,
        objective_name: ObjectiveName = ObjectiveName.ANNUAL_RETURN,
    ):
        self.config_path = config_path
        self.start_date = start_date
        self.end_date = end_date
        self.n_trials = int(n_trials)
        self.workers = max(1, int(workers))
        self.seed = int(seed)
        self.fee_ratio = float(fee_ratio)
        self.tax_ratio = float(tax_ratio)
        self.market_data_pickle_path = (
            Path(market_data_pickle_path) if market_data_pickle_path else DEFAULT_MARKET_DATA_PICKLE
        )
        self.allow_market_data_fetch = bool(allow_market_data_fetch)
        self.objective_name = objective_name

        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.study_dir = Path(output_dir) / f"composite_{run_tag}"
        self.study_dir.mkdir(parents=True, exist_ok=True)

        self.market_data = market_data
        self.base_params = OscarCompositeStrategy(config_path=self.config_path)._load_params_from_config()
        self.objective = build_objective(self.objective_name)

    def _load_market_data_once(self) -> dict:
        if self.market_data is not None:
            return self.market_data

        if not self.allow_market_data_fetch:
            with open(self.market_data_pickle_path, "rb") as f:
                self.market_data = pickle.load(f)
            return self.market_data

        bootstrap = OscarCompositeStrategy(config_path=self.config_path)
        self.market_data = bootstrap._load_market_data()
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

    def _build_storage(self):
        return JournalStorage(JournalFileBackend(str(self.study_dir / "optuna_journal.log")))

    def _create_or_load_study(self, storage, study_name: str, seed: int):
        return optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=1),
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )

    def _optimize_study(self, study_name: str, n_trials: int, seed: int) -> None:
        study = self._create_or_load_study(self._build_storage(), study_name, seed)
        study.optimize(self._objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    @staticmethod
    def _snap_upper_to_step(lower: float, upper: float, step: float) -> float:
        # Optuna expects (upper - lower) to be divisible by step for stepped floats.
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

    def _build_trial_params(self, trial: optuna.Trial) -> OscarCompositeParams:
        params_dict = asdict(self.base_params)

        sar_acceleration = trial.suggest_float("sar_acceleration", 0.01, 0.06, step=0.005)
        sar_maximum = trial.suggest_float("sar_maximum", max(0.1, sar_acceleration * 2.0), 0.4, step=0.01)

        macd_fast = trial.suggest_int("macd_fast", 8, 20)
        macd_slow = trial.suggest_int("macd_slow", macd_fast + 4, 40)
        macd_signal = trial.suggest_int("macd_signal", 5, 15)

        weight_sar = trial.suggest_float("weight_sar", 0.10, 0.60, step=0.05)
        weight_macd = trial.suggest_float("weight_macd", 0.10, 0.60, step=0.05)
        weight_volume = trial.suggest_float("weight_volume", 0.05, 0.40, step=0.05)
        weight_inst = trial.suggest_float("weight_institutional", 0.05, 0.40, step=0.05)
        weight_sum = weight_sar + weight_macd + weight_volume + weight_inst

        buy_score_threshold = trial.suggest_float("buy_score_threshold", 0.55, 0.95, step=0.01)
        sell_lower = 0.15
        sell_upper_raw = max(0.16, buy_score_threshold - 0.05)
        sell_upper = self._snap_upper_to_step(sell_lower, sell_upper_raw, 0.01)
        sell_score_threshold = trial.suggest_float(
            "sell_score_threshold", sell_lower, sell_upper, step=0.01
        )

        sar_signal_lag_max = trial.suggest_int("sar_signal_lag_max", 0, 10)
        macd_signal_lag_max = trial.suggest_int("macd_signal_lag_max", 0, 10)
        sar_event_decay_alpha = trial.suggest_float("sar_event_decay_alpha", 1.1, 4.0, step=0.1)
        macd_event_decay_alpha = trial.suggest_float("macd_event_decay_alpha", 1.1, 4.0, step=0.1)

        params_dict["sar_params"] = {
            "acceleration": sar_acceleration,
            "maximum": sar_maximum,
        }
        params_dict["macd_params"] = {
            "fastperiod": macd_fast,
            "slowperiod": macd_slow,
            "signalperiod": macd_signal,
        }
        params_dict["volume_above_avg_ratio"] = trial.suggest_float("volume_above_avg_ratio", 0.1, 1.0, step=0.05)
        params_dict["min_avg_volume_30"] = trial.suggest_int("min_avg_volume_30", 500_000, 5_000_000, step=250_000)
        params_dict["new_high_ratio_120"] = trial.suggest_float("new_high_ratio_120", 0.10, 0.80, step=0.05)
        params_dict["max_volume_spike_ratio"] = trial.suggest_float("max_volume_spike_ratio", 3.0, 15.0, step=0.5)
        params_dict["signal_weights"] = {
            "sar": weight_sar / weight_sum,
            "macd": weight_macd / weight_sum,
            "volume": weight_volume / weight_sum,
            "institutional": weight_inst / weight_sum,
        }
        params_dict["buy_score_threshold"] = buy_score_threshold
        params_dict["sell_score_threshold"] = sell_score_threshold
        params_dict["sar_signal_lag_max"] = sar_signal_lag_max
        params_dict["macd_signal_lag_max"] = macd_signal_lag_max
        params_dict["sar_event_decay_alpha"] = sar_event_decay_alpha
        params_dict["macd_event_decay_alpha"] = macd_event_decay_alpha
        params_dict["signal_quantile_bins"] = {
            "sar": trial.suggest_categorical("bins_sar", [5, 10, 20, 50, 100, 200, 500, 1000]),
            "macd": trial.suggest_categorical("bins_macd", [5, 10, 20, 50, 100, 200, 500, 1000]),
            "volume": trial.suggest_categorical("bins_volume", [5, 10, 20, 50, 100, 200, 500, 1000]),
            "institutional": trial.suggest_categorical("bins_institutional", [2, 3, 5, 9, 12]),
        }
        params_dict["sar_proximity_weight"] = trial.suggest_float("sar_proximity_weight", 0.1, 2.0, step=0.1)
        params_dict["sar_event_weight"] = trial.suggest_float("sar_event_weight", 0.1, 2.0, step=0.1)
        params_dict["macd_proximity_weight"] = trial.suggest_float("macd_proximity_weight", 0.1, 2.0, step=0.1)
        params_dict["macd_event_weight"] = trial.suggest_float("macd_event_weight", 0.1, 2.0, step=0.1)
        params_dict["sar_near_sigmoid_slope"] = trial.suggest_float("sar_near_sigmoid_slope", 2.0, 16.0, step=0.5)
        params_dict["sar_near_distance_scale"] = trial.suggest_float("sar_near_distance_scale", 0.005, 0.08, step=0.005)
        params_dict["sar_event_sigmoid_slope"] = trial.suggest_float("sar_event_sigmoid_slope", 2.0, 20.0, step=0.5)
        params_dict["sar_event_distance_scale"] = trial.suggest_float("sar_event_distance_scale", 0.005, 0.08, step=0.005)
        params_dict["sar_history_lookback"] = trial.suggest_int("sar_history_lookback", 1, 10)
        params_dict["sar_history_decay_alpha"] = trial.suggest_float("sar_history_decay_alpha", 1.05, 4.0, step=0.05)
        params_dict["macd_near_sigmoid_slope"] = trial.suggest_float("macd_near_sigmoid_slope", 2.0, 16.0, step=0.5)
        params_dict["macd_near_distance_scale"] = trial.suggest_float("macd_near_distance_scale", 0.05, 3.0, step=0.05)
        params_dict["macd_event_sigmoid_slope"] = trial.suggest_float("macd_event_sigmoid_slope", 2.0, 20.0, step=0.5)
        params_dict["macd_event_distance_scale"] = trial.suggest_float("macd_event_distance_scale", 0.05, 3.0, step=0.05)
        params_dict["macd_history_lookback"] = trial.suggest_int("macd_history_lookback", 1, 10)
        params_dict["macd_history_decay_alpha"] = trial.suggest_float("macd_history_decay_alpha", 1.05, 4.0, step=0.05)

        return OscarCompositeParams(**params_dict)

    def _flat_params_to_dataclass(self, flat_params: dict) -> OscarCompositeParams:
        base_params = asdict(self.base_params)
        weight_sar = float(flat_params["weight_sar"])
        weight_macd = float(flat_params["weight_macd"])
        weight_volume = float(flat_params["weight_volume"])
        weight_inst = float(flat_params["weight_institutional"])
        weight_sum = weight_sar + weight_macd + weight_volume + weight_inst
        weight_sum = weight_sum if weight_sum > 0 else 1.0

        base_params["sar_params"] = {
            "acceleration": float(flat_params["sar_acceleration"]),
            "maximum": float(flat_params["sar_maximum"]),
        }
        base_params["macd_params"] = {
            "fastperiod": int(flat_params["macd_fast"]),
            "slowperiod": int(flat_params["macd_slow"]),
            "signalperiod": int(flat_params["macd_signal"]),
        }
        base_params["volume_above_avg_ratio"] = float(flat_params["volume_above_avg_ratio"])
        base_params["min_avg_volume_30"] = int(flat_params["min_avg_volume_30"])
        base_params["new_high_ratio_120"] = float(flat_params["new_high_ratio_120"])
        base_params["max_volume_spike_ratio"] = float(flat_params["max_volume_spike_ratio"])
        base_params["signal_weights"] = {
            "sar": weight_sar / weight_sum,
            "macd": weight_macd / weight_sum,
            "volume": weight_volume / weight_sum,
            "institutional": weight_inst / weight_sum,
        }
        base_params["buy_score_threshold"] = float(flat_params["buy_score_threshold"])
        base_params["sell_score_threshold"] = float(flat_params["sell_score_threshold"])
        base_params["sar_signal_lag_max"] = int(flat_params["sar_signal_lag_max"])
        base_params["macd_signal_lag_max"] = int(flat_params["macd_signal_lag_max"])
        base_params["sar_event_decay_alpha"] = float(flat_params["sar_event_decay_alpha"])
        base_params["macd_event_decay_alpha"] = float(flat_params["macd_event_decay_alpha"])
        base_params["signal_quantile_bins"] = {
            "sar": flat_params.get("bins_sar"),
            "macd": flat_params.get("bins_macd"),
            "volume": flat_params.get("bins_volume"),
            "institutional": flat_params.get("bins_institutional"),
        }
        base_params["sar_proximity_weight"] = float(flat_params["sar_proximity_weight"])
        base_params["sar_event_weight"] = float(flat_params["sar_event_weight"])
        base_params["macd_proximity_weight"] = float(flat_params["macd_proximity_weight"])
        base_params["macd_event_weight"] = float(flat_params["macd_event_weight"])
        base_params["sar_near_sigmoid_slope"] = float(flat_params["sar_near_sigmoid_slope"])
        base_params["sar_near_distance_scale"] = float(flat_params["sar_near_distance_scale"])
        base_params["sar_event_sigmoid_slope"] = float(flat_params["sar_event_sigmoid_slope"])
        base_params["sar_event_distance_scale"] = float(flat_params["sar_event_distance_scale"])
        base_params["sar_history_lookback"] = int(flat_params["sar_history_lookback"])
        base_params["sar_history_decay_alpha"] = float(flat_params["sar_history_decay_alpha"])
        base_params["macd_near_sigmoid_slope"] = float(flat_params["macd_near_sigmoid_slope"])
        base_params["macd_near_distance_scale"] = float(flat_params["macd_near_distance_scale"])
        base_params["macd_event_sigmoid_slope"] = float(flat_params["macd_event_sigmoid_slope"])
        base_params["macd_event_distance_scale"] = float(flat_params["macd_event_distance_scale"])
        base_params["macd_history_lookback"] = int(flat_params["macd_history_lookback"])
        base_params["macd_history_decay_alpha"] = float(flat_params["macd_history_decay_alpha"])

        return OscarCompositeParams(**base_params)

    def _objective(self, trial: optuna.Trial) -> float:
        try:
            params = self._build_trial_params(trial)

            strategy = OscarCompositeStrategy(
                config_path=self.config_path,
                market_data=self.market_data,
            )

            report = strategy.run_strategy(
                params=params,
                start_date=self.start_date,
                fee_ratio=self.fee_ratio,
                tax_ratio=self.tax_ratio,
            )

            objective_result = self.objective.evaluate(
                report,
                start_date=self.start_date,
                end_date=self.end_date,
            )

            result = TrialResult.from_report(report, objective_result, end_date=self.end_date)
            result.apply_to_trial(trial)

            if result.trial_status != "ok":
                logger.warning("Trial %s invalid objective (NaN/inf/None). params=%s", trial.number, trial.params)
                return MIN_OBJECTIVE_VALUE

            return result.objective_value
        except Exception as exc:
            TrialResult.from_exception(exc).apply_to_trial(trial)
            logger.exception("Trial %s failed. params=%s", trial.number, trial.params)
            raise

    def run(self) -> Path:
        study_name = f"oscar_composite_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        market_data = self._load_market_data_once()
        self._persist_market_data_pickle(market_data)

        storage = self._build_storage()
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

            study = self._create_or_load_study(self._build_storage(), study_name, self.seed)

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
        best_params = dict(best_trial.params)
        self.best_report = {
            "best_value": best_trial.user_attrs.get("objective_value"),
            "best_params": best_params,
            "best_user_attrs": best_trial.user_attrs,
            "config": {
                "config_path": self.config_path,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "objective_name": self.objective_name,
                "n_trials": self.n_trials,
                "workers": self.workers,
                "fee_ratio": self.fee_ratio,
                "tax_ratio": self.tax_ratio,
            },
            "best_params_dataclass_view": asdict(self._flat_params_to_dataclass(best_params)),
        }
        best_report = self.best_report

        with open(self.study_dir / f"best_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w", encoding="utf-8") as f:
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

        return self.study_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bayesian optimization for OscarCompositeParams.")
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--objective-name",
        default=ObjectiveName.ANNUAL_RETURN.value,
        choices=[e.value for e in ObjectiveName],
        help="Objective function computed from finlab report on train window.",
    )
    parser.add_argument("--n-trials", type=int, default=1000, help="Number of optimization trials.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel process workers.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=f"assets/OscarCompositeStrategy/bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    parser.add_argument(
        "--market-data-pickle-path",
        default=str(DEFAULT_MARKET_DATA_PICKLE),
        help="Shared pickle path for market data. Parent writes once; workers read only.",
    )
    return parser


def _run_bayesian_worker(worker_payload, study_name: str, n_trials: int, seed: int) -> None:
    executor = CompositeBayesianOptimizer(
        config_path=worker_payload["config_path"],
        start_date=worker_payload["start_date"],
        end_date=worker_payload["end_date"],
        objective_name=ObjectiveName(worker_payload.get("objective_name", ObjectiveName.ANNUAL_RETURN.value)),
        n_trials=n_trials,
        workers=1,
        seed=seed,
        output_dir=worker_payload["output_dir"],
        fee_ratio=worker_payload["fee_ratio"],
        tax_ratio=worker_payload["tax_ratio"],
        market_data=None,
        market_data_pickle_path=worker_payload["market_data_pickle_path"],
        allow_market_data_fetch=False,
    )
    executor._load_market_data_once()
    executor._optimize_study(study_name, n_trials, seed)


def main() -> None:
    args = _build_arg_parser().parse_args()

    optimizer = CompositeBayesianOptimizer(
        config_path=args.config_path,
        start_date=args.start_date,
        end_date=args.end_date,
        objective_name=args.objective_name,
        n_trials=args.n_trials,
        workers=args.workers,
        seed=args.seed,
        output_dir=args.output_dir,
        fee_ratio=0.001425,
        tax_ratio=0.003,
        market_data_pickle_path=args.market_data_pickle_path,
    )
    out = optimizer.run()
    print(f"Bayesian optimization completed. Outputs: {out}")


if __name__ == "__main__":
    main()
