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
from tests.oscar_tw_strategy.utils.objective_functions import ObjectiveName, build_objective
from tests.oscar_tw_strategy.utils.trial_result import TrialResult

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
        fee_ratio: float = 0.001425,
        tax_ratio: float = 0.003,
        market_data: dict | None = None,
        market_data_pickle_path: str | None = None,
        allow_market_data_fetch: bool = True,
        objective_name: ObjectiveName = ObjectiveName.ANNUAL_RETURN,
        study_dir: str | None = None,
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

        if study_dir is not None:
            self.study_dir = Path(study_dir)
        else:
            run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.study_dir = Path(output_dir) / f"andor_{run_tag}"
        self.study_dir.mkdir(parents=True, exist_ok=True)

        self.market_data = market_data
        self.objective = build_objective(self.objective_name)

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
        # catch=(Exception,) prevents a single failing trial from crashing the entire worker process.
        study.optimize(self._objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False, catch=(Exception,))

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

    def _suggest_run_kwargs(self) -> dict:
        return {
            "start_date": self.start_date,
            "fee_ratio": self.fee_ratio,
            "tax_ratio": self.tax_ratio,
            "sim_resample": "D",
        }

    def _objective(self, trial: optuna.Trial) -> float:
        try:
            strategy_kwargs = self._suggest_strategy_kwargs(trial)
            run_kwargs = self._suggest_run_kwargs()

            strategy = OscarAndOrStrategy(**strategy_kwargs)
            report = strategy.run_strategy(**run_kwargs)

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
        study_name = f"oscar_andor_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        market_data = self._load_market_data_once()
        if not self.market_data_pickle_path.exists():
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
                "study_dir": str(self.study_dir),
                "fee_ratio": self.fee_ratio,
                "tax_ratio": self.tax_ratio,
                "market_data_pickle_path": str(self.market_data_pickle_path),
            }

            error_queue = mp.Queue()
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
                        error_queue,
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
                captured_errors = []
                while not error_queue.empty():
                    try:
                        captured_errors.append(error_queue.get_nowait())
                    except Exception:
                        break
                detail = "\n\n".join(captured_errors) if captured_errors else "(no exception details captured — check worker stderr)"
                logger.warning("One or more bayesian worker processes failed (skipping).\n%s", detail)

            study = self._create_or_load_study(self._build_storage(), study_name, self.seed)

            # Mark any zombie RUNNING trials (left by crashed workers) as FAIL
            # so the sampler doesn't avoid those parameter regions unnecessarily.
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.RUNNING:
                    try:
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                        logger.warning("Marked zombie trial %d as FAIL.", trial.number)
                    except Exception:
                        pass

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

        self.best_report = {
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
                "fee_ratio": self.fee_ratio,
                "tax_ratio": self.tax_ratio,
            },
        }

        with open(self.study_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(self.best_report, f, ensure_ascii=False, indent=2)

        rows = []
        for t in study.trials:
            rows.append(
                {
                    "number": t.number,
                    "state": str(t.state),
                    "value": t.value,
                    "trial_status": t.user_attrs.get("trial_status"),
                    "failure_reason": t.user_attrs.get("failure_reason"),
                    "objective_name": t.user_attrs.get("objective_name"),
                    "objective_value": t.user_attrs.get("objective_value"),
                    "sharpe_ratio": t.user_attrs.get("sharpe_ratio"),
                    "annual_return": t.user_attrs.get("annual_return"),
                    "max_drawdown": t.user_attrs.get("max_drawdown"),
                    "params": json.dumps(t.params, ensure_ascii=False, sort_keys=True),
                }
            )

        pd.DataFrame(rows).to_csv(self.study_dir / "trials.csv", index=False, encoding="utf-8-sig")

        return self.study_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bayesian optimization for OscarAndOrStrategy.")
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
    parser.add_argument("--output-dir", default="assets/OscarTWStrategy/andor_bayesian_params")

    parser.add_argument(
        "--market-data-pickle-path",
        default=str(DEFAULT_MARKET_DATA_PICKLE),
        help="Shared pickle path for market data. Parent writes once; workers read only.",
    )
    return parser


def _run_bayesian_worker(
    worker_payload, study_name: str, n_trials: int, seed: int, error_queue=None
) -> None:
    import traceback as _tb

    try:
        executor = AndOrBayesianOptimizer(
            config_path=worker_payload["config_path"],
            start_date=worker_payload["start_date"],
            end_date=worker_payload["end_date"],
            objective_name=ObjectiveName(worker_payload.get("objective_name", ObjectiveName.ANNUAL_RETURN.value)),
            n_trials=n_trials,
            workers=1,
            seed=seed,
            study_dir=worker_payload["study_dir"],
            fee_ratio=worker_payload["fee_ratio"],
            tax_ratio=worker_payload["tax_ratio"],
            market_data=None,
            market_data_pickle_path=worker_payload["market_data_pickle_path"],
            allow_market_data_fetch=False,
        )
        executor._load_market_data_once()
        executor._optimize_study(study_name, n_trials, seed)
    except Exception as exc:
        msg = f"Worker seed={seed}: {type(exc).__name__}: {exc}\n{_tb.format_exc()}"
        logger.error(msg)
        if error_queue is not None:
            try:
                error_queue.put_nowait(msg)
            except Exception:
                pass
        raise


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
        fee_ratio=0.001425,
        tax_ratio=0.003,
        market_data_pickle_path=args.market_data_pickle_path,
    )
    out = optimizer.run()
    print(f"Bayesian optimization completed. Outputs: {out}")


if __name__ == "__main__":
    main()
