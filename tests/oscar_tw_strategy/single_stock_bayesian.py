"""
Single Stock Bayesian Executor

使用 Optuna 對 Oscar 策略進行單一股票參數優化。

設計重點:
1. 市場資料僅載入一次，避免 trial 重複 I/O
2. 使用 TPE sampler 探索參數空間
3. 使用 MedianPruner 在早期回測階段淘汰劣質 trial
4. 輸出 best params、trials CSV 與 Optuna 視覺化圖表
"""

import json
import inspect
import logging
import multiprocessing as mp
import os
import pickle
import resource
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from finlab.backtest import sim

# Prevent each worker process from spawning many native threads (BLAS/OpenMP),
# which can otherwise exhaust OS thread limits under high process counts.
for _env_key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_env_key, "1")

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "optuna is required for single_stock_bayesian.py. "
        "Please install dependencies from environment.yml first."
    ) from exc

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_class.oscar_tw_strategy import AdjustTWMarketInfo, OscarTWStrategy


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SingleStockBayesianExecutor:
    """執行單一股票 Bayesian Optimization。"""

    def __init__(
        self,
        stock_id: str,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        output_dir: str = "assets/OscarTWStrategy/single_stock_bayesian",
        n_trials: int = 400,
        n_jobs: Optional[int] = None,
        process_workers: Optional[int] = None,
        market_data_pickle_path: Optional[str] = None,
        preload_market_data: bool = True,
        allow_market_data_fetch: bool = True,
        seed: int = 42,
        fee_ratio: float = 0.001425,
        tax_ratio: float = 0.003,
        volume_above_avg_ratio: float = 0.25,
        new_high_ratio_120: float = 0.3,
    ):
        self.stock_id = stock_id
        self.start_date = start_date
        self.end_date = end_date
        self.n_trials = n_trials
        self.seed = seed
        self.fee_ratio = fee_ratio
        self.tax_ratio = tax_ratio
        self.volume_above_avg_ratio = volume_above_avg_ratio
        self.new_high_ratio_120 = new_high_ratio_120
        self.market_data_pickle_path = Path(market_data_pickle_path) if market_data_pickle_path else None
        self.allow_market_data_fetch = allow_market_data_fetch
        self._process_workers_user_specified = process_workers is not None

        cpu_count = os.cpu_count() or 4
        # n_jobs = threads per worker process (Optuna uses ThreadPoolExecutor internally).
        self.n_jobs = 1 if n_jobs is None else max(1, int(n_jobs))
        # process_workers = number of OS processes participating in the same Optuna study.
        if process_workers is None:
            if self.n_jobs > 1:
                # If user requests multi-thread trials, keep process count conservative by default.
                self.process_workers = max(1, cpu_count // self.n_jobs)
            else:
                self.process_workers = int(cpu_count * 0.75) if cpu_count > 100 else max(1, cpu_count - 2)
        else:
            self.process_workers = max(1, int(process_workers))

        self._adapt_parallelism(cpu_count)

        # Market-data / compat / trading-days init — always runs regardless of whether
        # parallelism was adjusted (fixes NameError + partially-dead init block).
        self.market_data = None
        if preload_market_data:
            self.market_data = self._load_market_data_once()

        # Cache constructor compatibility once to avoid per-trial TypeError overhead.
        sig_params = set(inspect.signature(OscarTWStrategy.__init__).parameters.keys())
        self._supports_new_lag_params = (
            "sar_signal_lag_min" in sig_params and "sar_signal_lag_max" in sig_params
        )
        self._supports_volume_override = "volume_above_avg_ratio" in sig_params
        self._supports_new_high_override = "new_high_ratio_120" in sig_params

        self.trading_days = None
        if self.market_data is not None:
            self.trading_days = self.market_data["close"].loc[self.start_date :].index
            if self.end_date:
                self.trading_days = self.trading_days[self.trading_days <= pd.Timestamp(self.end_date)]
            if len(self.trading_days) == 0:
                raise ValueError("No trading days available in selected date range.")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.study_dir = self.output_dir / f"{self.stock_id}_{self.start_date}_{self.end_date or 'latest'}"
        self.study_dir.mkdir(parents=True, exist_ok=True)

        logger.info("初始化 SingleStockBayesianExecutor")
        logger.info("股票代碼: %s", self.stock_id)
        logger.info("回測區間: %s -> %s", self.start_date, self.end_date or "latest")
        logger.info(
            "n_trials=%s, process_workers=%s, n_jobs(per worker)=%s",
            self.n_trials,
            self.process_workers,
            self.n_jobs,
        )

    def _adapt_parallelism(self, cpu_count: int) -> None:
        """Balance process/thread parallelism to avoid thread exhaustion while preserving throughput."""
        total_requested_threads = self.process_workers * self.n_jobs

        # Soft budget: keep total Python threads around ~2x CPU cores.
        # This is much safer than process_workers*n_jobs at 10k+ scale.
        soft_thread_budget = max(32, cpu_count * 2)

        # Hard ceiling from OS process/thread quota if available.
        hard_limit = None
        try:
            rlimit_nproc = resource.getrlimit(resource.RLIMIT_NPROC)[0]
            if rlimit_nproc not in (-1, resource.RLIM_INFINITY):
                hard_limit = int(rlimit_nproc)
        except Exception:
            hard_limit = None

        budget = soft_thread_budget
        if hard_limit is not None:
            # Leave headroom for main process, DB threads, BLAS, and system services.
            budget = min(budget, max(8, hard_limit // 4))

        if total_requested_threads <= budget:
            return

        if self._process_workers_user_specified:
            # Respect user process_workers, shrink n_jobs first.
            adjusted_n_jobs = max(1, budget // self.process_workers)
            if adjusted_n_jobs != self.n_jobs:
                logger.warning(
                    "並行設定過大 (process_workers=%s, n_jobs=%s, total=%s)。"
                    "已自動調整 n_jobs -> %s (budget=%s)。",
                    self.process_workers,
                    self.n_jobs,
                    total_requested_threads,
                    adjusted_n_jobs,
                    budget,
                )
                self.n_jobs = adjusted_n_jobs
            return

        # process_workers is auto mode: keep n_jobs, shrink process count to fit budget.
        adjusted_workers = max(1, budget // self.n_jobs)
        if adjusted_workers != self.process_workers:
            logger.warning(
                "自動調整 process_workers: %s -> %s (n_jobs=%s, budget=%s)",
                self.process_workers,
                adjusted_workers,
                self.n_jobs,
                budget,
            )
            self.process_workers = adjusted_workers

    def _load_market_data_once(self):
        """Load market data from local pickle if available; otherwise fetch once and optionally persist."""
        if self.market_data_pickle_path and self.market_data_pickle_path.exists():
            logger.info("從本地 pickle 載入市場資料: %s", self.market_data_pickle_path)
            try:
                return self._load_pickle_with_retry(self.market_data_pickle_path)
            except RuntimeError:
                if not self.allow_market_data_fetch:
                    raise
                logger.warning("本地 pickle 損壞，將重新抓取市場資料: %s", self.market_data_pickle_path)
                try:
                    self.market_data_pickle_path.unlink(missing_ok=True)
                except OSError:
                    pass

        if not self.allow_market_data_fetch:
            raise RuntimeError(
                f"Market data pickle not found and fetching disabled: {self.market_data_pickle_path}"
            )

        logger.info("載入市場資料（單次）...")
        market_data = OscarTWStrategy.load_market_data()
        logger.info("市場資料載入完成")

        if self.market_data_pickle_path:
            self._persist_market_data_pickle(market_data)

        return market_data

    def _persist_market_data_pickle(self, market_data) -> None:
        """Persist market data with atomic replace so workers never see partial content."""
        if not self.market_data_pickle_path:
            return
        self.market_data_pickle_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.market_data_pickle_path.with_suffix(self.market_data_pickle_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(market_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.market_data_pickle_path)
        logger.info("市場資料已寫入 pickle: %s", self.market_data_pickle_path)

    @staticmethod
    def _load_pickle_with_retry(path: Path, retries: int = 30, delay_sec: float = 1.0):
        """Retry reading pickle to avoid transient truncated reads during concurrent startup."""
        last_error = None
        for _ in range(retries):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError, OSError) as e:
                last_error = e
                time.sleep(delay_sec)
        raise RuntimeError(f"Failed to load market data pickle after retries: {path}") from last_error

    def _build_strategy(self, params: Dict) -> OscarTWStrategy:
        shared_kwargs = {
            "sar_params": {
                "acceleration": params["sar_acceleration"],
                "maximum": params["sar_maximum"],
            },
            "macd_params": {
                "fastperiod": params["macd_fast"],
                "slowperiod": params["macd_slow"],
                "signalperiod": params["macd_signal"],
            },
            "market_data": self.market_data,
        }

        if self._supports_volume_override:
            shared_kwargs["volume_above_avg_ratio"] = self.volume_above_avg_ratio
        if self._supports_new_high_override:
            shared_kwargs["new_high_ratio_120"] = self.new_high_ratio_120

        if self._supports_new_lag_params:
            shared_kwargs["sar_signal_lag_min"] = params["sar_signal_lag_min"]
            shared_kwargs["sar_signal_lag_max"] = params["sar_signal_lag_max"]
            return OscarTWStrategy(**shared_kwargs)

        # Legacy fallback for environments that still use sar_max_dots/sar_reject_dots.
        legacy_max_dots = max(1, int(params["sar_signal_lag_max"]))
        shared_kwargs["sar_max_dots"] = legacy_max_dots
        shared_kwargs["sar_reject_dots"] = max(legacy_max_dots + 1, 3)
        return OscarTWStrategy(**shared_kwargs)

    def _single_stock_position(self, strategy: OscarTWStrategy) -> pd.DataFrame:
        base_position = strategy.base_position.reindex(self.trading_days, fill_value=False)

        if self.stock_id not in base_position.columns:
            raise ValueError(f"Stock {self.stock_id} not found in base_position columns.")

        single_stock_position = pd.DataFrame(False, index=base_position.index, columns=[self.stock_id])
        single_stock_position[self.stock_id] = base_position[self.stock_id]
        return single_stock_position

    def _run_backtest(self, position: pd.DataFrame):
        return sim(
            position=position,
            resample=None,
            upload=False,
            market=AdjustTWMarketInfo(),
            fee_ratio=self.fee_ratio,
            tax_ratio=self.tax_ratio,
            position_limit=1.0,
        )

    def _objective(self, trial: optuna.trial.Trial) -> float:
        sar_signal_lag_min = trial.suggest_int("sar_signal_lag_min", 0, 4)
        sar_signal_lag_max = trial.suggest_int("sar_signal_lag_max", sar_signal_lag_min, 10)
        sar_acceleration = trial.suggest_float("sar_acceleration", 0.01, 0.05, step=0.005)
        sar_maximum = trial.suggest_float("sar_maximum", max(0.1, sar_acceleration * 2), 0.4, step=0.01)

        macd_fast = trial.suggest_int("macd_fast", 8, 20)
        macd_slow = trial.suggest_int("macd_slow", macd_fast + 4, 40)
        macd_signal = trial.suggest_int("macd_signal", 5, 15)

        params = {
            "sar_signal_lag_min": sar_signal_lag_min,
            "sar_signal_lag_max": sar_signal_lag_max,
            "sar_acceleration": sar_acceleration,
            "sar_maximum": sar_maximum,
            "macd_fast": macd_fast,
            "macd_slow": macd_slow,
            "macd_signal": macd_signal,
        }

        strategy = self._build_strategy(params)
        position = self._single_stock_position(strategy)

        # 沒有任何訊號時直接給低分，避免浪費完整回測。
        if not position[self.stock_id].any():
            trial.set_user_attr("annual_return", -1.0)
            trial.set_user_attr("max_drawdown", 0.0)
            trial.set_user_attr("sharpe_ratio", None)
            trial.set_user_attr("total_trades", 0)
            return -1.0

        # Early pruning: 先跑前 25% 區間取得初步績效。
        if len(position.index) >= 60:
            warmup_len = max(20, int(len(position.index) * 0.25))
            warmup_end = position.index[warmup_len - 1]
            warmup_position = position.loc[:warmup_end]
            warmup_report = self._run_backtest(warmup_position)
            warmup_metrics = warmup_report.get_metrics()
            warmup_annual_return = warmup_metrics["profitability"]["annualReturn"]
            trial.report(warmup_annual_return, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned by median rule. warmup_return={warmup_annual_return:.4f}")

        report = self._run_backtest(position)
        metrics = report.get_metrics()
        trades = report.get_trades()

        annual_return = metrics["profitability"]["annualReturn"]
        max_drawdown = metrics["risk"]["maxDrawdown"]
        sharpe_ratio = metrics["ratio"].get("sharpeRatio", None)

        trial.set_user_attr("annual_return", annual_return)
        trial.set_user_attr("max_drawdown", max_drawdown)
        trial.set_user_attr("sharpe_ratio", sharpe_ratio)
        trial.set_user_attr("total_trades", len(trades))

        return annual_return

    def _save_study_outputs(self, study: optuna.Study) -> None:
        best = {
            "stock_id": self.stock_id,
            "study_name": study.study_name,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_metrics": {
                "annual_return": study.best_trial.user_attrs.get("annual_return"),
                "max_drawdown": study.best_trial.user_attrs.get("max_drawdown"),
                "sharpe_ratio": study.best_trial.user_attrs.get("sharpe_ratio"),
                "total_trades": study.best_trial.user_attrs.get("total_trades"),
            },
            "fixed_inputs": {
                "volume_above_avg_ratio": self.volume_above_avg_ratio,
                "new_high_ratio_120": self.new_high_ratio_120,
                "start_date": self.start_date,
                "end_date": self.end_date,
            },
        }

        best_path = self.study_dir / "best_params.json"
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=2)

        df_trials = study.trials_dataframe()
        trials_path = self.study_dir / "trials.csv"
        df_trials.to_csv(trials_path, index=False, encoding="utf-8-sig")

        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
            )

            plot_optimization_history(study).write_html(str(self.study_dir / "optimization_history.html"))
            plot_param_importances(study).write_html(str(self.study_dir / "param_importance.html"))
            plot_parallel_coordinate(study).write_html(str(self.study_dir / "parallel_coordinate.html"))
        except Exception as e:
            logger.warning("無法產生 Optuna 視覺化圖表: %s", e)

        logger.info("最佳參數已輸出: %s", best_path)
        logger.info("Trial 紀錄已輸出: %s", trials_path)

    def _create_or_load_study(
        self,
        storage_url: str,
        study_name: str,
        sampler_seed: int,
    ) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(
            seed=sampler_seed,
            n_startup_trials=30,
            multivariate=True,
            group=True,
        )
        pruner = optuna.pruners.MedianPruner(n_startup_trials=30, n_warmup_steps=1)

        return optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=storage_url,
            load_if_exists=True,
        )

    def _optimize_study(self, storage_url: str, study_name: str, n_trials: int, sampler_seed: int) -> None:
        study = self._create_or_load_study(storage_url, study_name, sampler_seed)
        study.optimize(
            self._objective,
            n_trials=n_trials,
            n_jobs=self.n_jobs,
            gc_after_trial=True,
        )

    def optimize(self) -> Dict:
        storage_url = f"sqlite:///{self.study_dir / 'optuna_study.db'}?timeout=120"
        study_name = f"oscar_bayes_{self.stock_id}_{self.start_date}_{self.end_date or 'latest'}"

        # Guard: SQLite cannot handle concurrent writes from multiple OS processes.
        if storage_url.startswith("sqlite") and self.process_workers > 1:
            raise RuntimeError(
                f"SQLite storage cannot be used with process_workers={self.process_workers} > 1 "
                "due to database locking under concurrent writes. "
                "Use process_workers=1 or switch Optuna storage to PostgreSQL/MySQL."
            )

        # Parent process fetches market data once and shares it via local pickle across worker processes.
        # Use per-run file to avoid reading stale/corrupted cache from previous runs.
        market_data_pickle = self.market_data_pickle_path or (self.study_dir / f"market_data_{os.getpid()}.pkl")
        self.market_data_pickle_path = market_data_pickle
        if self.market_data is None:
            self.market_data = self._load_market_data_once()
        # Lazily compute trading_days if not already set (e.g. preload_market_data=False in __init__).
        if self.trading_days is None:
            self.trading_days = self.market_data["close"].loc[self.start_date :].index
            if self.end_date:
                self.trading_days = self.trading_days[self.trading_days <= pd.Timestamp(self.end_date)]
            if len(self.trading_days) == 0:
                raise ValueError("No trading days available in selected date range.")
        # Always materialize current in-memory data to pickle before spawning workers.
        self._persist_market_data_pickle(self.market_data)

        study = self._create_or_load_study(storage_url, study_name, self.seed)
        study.set_user_attr("stock_id", self.stock_id)
        study.set_user_attr("volume_above_avg_ratio", self.volume_above_avg_ratio)
        study.set_user_attr("new_high_ratio_120", self.new_high_ratio_120)
        study.set_user_attr("process_workers", self.process_workers)
        study.set_user_attr("thread_jobs_per_worker", self.n_jobs)

        logger.info("開始 Bayesian Optimization...")
        try:
            if self.process_workers <= 1:
                self._optimize_study(storage_url, study_name, self.n_trials, self.seed)
            else:
                worker_count = min(self.process_workers, self.n_trials)
                base_trials = self.n_trials // worker_count
                extra_trials = self.n_trials % worker_count
                trials_per_worker = [base_trials + (1 if i < extra_trials else 0) for i in range(worker_count)]

                logger.info(
                    "使用多進程優化: workers=%s, trial 分配=%s",
                    worker_count,
                    trials_per_worker,
                )

                worker_payload = {
                    "stock_id": self.stock_id,
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "output_dir": str(self.output_dir),
                    "market_data_pickle_path": str(self.market_data_pickle_path),
                    "n_jobs": self.n_jobs,
                    "seed": self.seed,
                    "fee_ratio": self.fee_ratio,
                    "tax_ratio": self.tax_ratio,
                    "volume_above_avg_ratio": self.volume_above_avg_ratio,
                    "new_high_ratio_120": self.new_high_ratio_120,
                }

                processes = []
                for worker_id, worker_trials in enumerate(trials_per_worker):
                    if worker_trials <= 0:
                        continue
                    p = mp.Process(
                        target=_run_bayesian_worker,
                        args=(worker_payload, storage_url, study_name, worker_trials, worker_id),
                    )
                    p.start()
                    processes.append(p)

                failed = False
                for p in processes:
                    p.join()
                    if p.exitcode != 0:
                        failed = True
                        logger.error("Bayesian worker pid=%s failed with exit code %s", p.pid, p.exitcode)

                if failed:
                    raise RuntimeError("One or more bayesian worker processes failed.")
        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user. Saving current best results.")

        # Reload study state after optimization workers complete.
        study = self._create_or_load_study(storage_url, study_name, self.seed)
        if len(study.trials) == 0 or study.best_trial is None:
            raise RuntimeError("No successful trials were completed.")

        self._save_study_outputs(study)

        result = {
            "stock_id": self.stock_id,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial_number": study.best_trial.number,
            "output_dir": str(self.study_dir),
        }

        logger.info("Bayesian optimization 完成")
        logger.info("Best value (annual return): %.4f", study.best_value)
        logger.info("Best params: %s", study.best_params)
        return result


def _run_bayesian_worker(worker_payload, storage_url, study_name, worker_trials, worker_id):
    """Process entrypoint for parallel Optuna workers sharing the same study storage."""
    executor = SingleStockBayesianExecutor(
        stock_id=worker_payload["stock_id"],
        start_date=worker_payload["start_date"],
        end_date=worker_payload["end_date"],
        output_dir=worker_payload["output_dir"],
        n_trials=worker_trials,
        n_jobs=worker_payload["n_jobs"],
        process_workers=1,
        market_data_pickle_path=worker_payload["market_data_pickle_path"],
        preload_market_data=True,
        allow_market_data_fetch=False,
        seed=worker_payload["seed"] + worker_id,
        fee_ratio=worker_payload["fee_ratio"],
        tax_ratio=worker_payload["tax_ratio"],
        volume_above_avg_ratio=worker_payload["volume_above_avg_ratio"],
        new_high_ratio_120=worker_payload["new_high_ratio_120"],
    )
    executor._optimize_study(storage_url, study_name, worker_trials, worker_payload["seed"] + worker_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Single Stock Bayesian Executor")
    parser.add_argument("--stock_id", type=str, required=True, help="指定股票代碼")
    parser.add_argument("--start_date", type=str, default="2023-01-01", help="回測起始日期")
    parser.add_argument("--enddate", type=str, default=None, help="回測結束日期（可選）")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="assets/OscarTWStrategy/single_stock_bayesian",
        help="結果輸出目錄",
    )
    parser.add_argument("--n_trials", type=int, default=400, help="Bayesian trial 數")
    parser.add_argument("--n_jobs", type=int, default=1, help="每個 worker 內部 thread 數（CPU-heavy 建議 1）")
    parser.add_argument("--process_workers", type=int, default=None, help="Bayesian 多進程 worker 數（預設自動）")
    parser.add_argument("--market_data_pickle", type=str, default=None, help="市場資料 pickle 路徑（可重用，避免重抓）")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    parser.add_argument("--fee_ratio", type=float, default=0.001425, help="手續費率")
    parser.add_argument("--tax_ratio", type=float, default=0.003, help="證交稅率")
    parser.add_argument(
        "--volume_ratio",
        type=float,
        default=0.25,
        help="固定成交量門檻比例（相對 30 日均量）",
    )
    parser.add_argument(
        "--new_high_ratio_120",
        type=float,
        default=0.3,
        help="固定 120 日創新高比例門檻",
    )

    args = parser.parse_args()

    executor = SingleStockBayesianExecutor(
        stock_id=args.stock_id,
        start_date=args.start_date,
        end_date=args.enddate,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        process_workers=args.process_workers,
        market_data_pickle_path=args.market_data_pickle,
        seed=args.seed,
        fee_ratio=args.fee_ratio,
        tax_ratio=args.tax_ratio,
        volume_above_avg_ratio=args.volume_ratio,
        new_high_ratio_120=args.new_high_ratio_120,
    )

    executor.optimize()
