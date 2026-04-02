"""Walk-forward Bayesian optimization for OscarCompositeStrategy.

Logic:
  1. Split the full date range into rolling windows:
       train = [oos_start - train_window_months, oos_start - 1 day]
       oos   = [oos_start, oos_start + oos_window_months - 1 day]
     starting from start_date, rolling forward by oos_window_months each step.
  2. For each window: run CompositeBayesianOptimizer on the training window,
     read the best params from the saved JSON, reconstruct OscarCompositeStrategy
     with those params, and collect the equal-weight position for the OOS slice.
  3. Concatenate all OOS positions, run finlab sim once, save the report.

Usage:
  python -m tests.oscar_tw_strategy.walk_forward_bayes_composite
  python -m tests.oscar_tw_strategy.walk_forward_bayes_composite \\
      --start-date 2020-01-01 --end-date 2024-12-31 \\
      --train-window-months 12 --oos-window-months 3 \\
      --n-trials 300 --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta
from finlab.backtest import sim

from strategy_class.oscar.oscar_strategy_composite import OscarCompositeStrategy
from tests.oscar_tw_strategy.bayesian_optimize_composite_params import (
    CompositeBayesianOptimizer,
)
from tests.oscar_tw_strategy.utils.objective_functions import ObjectiveName

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKET_DATA_PICKLE = (
    REPO_ROOT / "finlab_db" / "workspace" / "oscar_composite_market_data.pkl"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class WalkForwardComposite:
    """Walk-forward optimizer for OscarCompositeStrategy.

    For each rolling window the best Bayesian params are found on the training
    slice; those params are then used to generate out-of-sample positions.
    All OOS positions are concatenated and passed to a single finlab sim call.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        start_date: str = "2020-01-01",
        end_date: str | None = None,
        train_window_months: int = 12,
        oos_window_months: int = 3,
        n_trials: int = 300,
        workers: int = 1,
        seed: int = 42,
        output_dir: str = "assets/OscarTWStrategy/walk_forward_composite",
        fee_ratio: float = 0.001425,
        tax_ratio: float = 0.003,
        market_data_pickle_path: str | None = None,
        objective_name: ObjectiveName = ObjectiveName.ANNUAL_RETURN,
    ):
        self.config_path = config_path
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        self.train_window_months = int(train_window_months)
        self.oos_window_months = int(oos_window_months)
        self.n_trials = int(n_trials)
        self.workers = max(1, int(workers))
        self.seed = int(seed)
        self.fee_ratio = float(fee_ratio)
        self.tax_ratio = float(tax_ratio)
        self.market_data_pickle_path = (
            Path(market_data_pickle_path) if market_data_pickle_path else DEFAULT_MARKET_DATA_PICKLE
        )
        self.objective_name = objective_name

        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"wf_composite_{run_tag}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.market_data: dict | None = None

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def _load_market_data_once(self) -> dict:
        if self.market_data is not None:
            return self.market_data
        if self.market_data_pickle_path.exists():
            logger.info("Loading market data from pickle: %s", self.market_data_pickle_path)
            with open(self.market_data_pickle_path, "rb") as f:
                self.market_data = pickle.load(f)
        else:
            logger.info("Fetching market data from finlab...")
            bootstrap = OscarCompositeStrategy(config_path=self.config_path)
            self.market_data = bootstrap._load_market_data()
            self._persist_market_data_pickle(self.market_data)
        return self.market_data

    def _persist_market_data_pickle(self, market_data: dict) -> None:
        self.market_data_pickle_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.market_data_pickle_path.with_suffix(self.market_data_pickle_path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(market_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.market_data_pickle_path)
        logger.info("Market data persisted to %s", self.market_data_pickle_path)

    # ------------------------------------------------------------------
    # Window generation
    # ------------------------------------------------------------------

    def _generate_windows(self) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Return list of (train_start, train_end, oos_start, oos_end) tuples."""
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)
        windows = []
        oos_start = start + relativedelta(months=self.train_window_months)
        while oos_start <= end:
            train_start = oos_start - relativedelta(months=self.train_window_months)
            train_end = oos_start - relativedelta(days=1)
            oos_end_raw = oos_start + relativedelta(months=self.oos_window_months) - relativedelta(days=1)
            oos_end = min(oos_end_raw, end)
            windows.append((train_start, train_end, oos_start, oos_end))
            oos_start = oos_start + relativedelta(months=self.oos_window_months)
        return windows

    # ------------------------------------------------------------------
    # Per-window optimization
    # ------------------------------------------------------------------

    def _optimize_window(
        self,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        window_idx: int,
    ) -> dict:
        """Run Bayesian optimization on [train_start, train_end], return best flat params dict."""
        window_output_dir = self.output_dir / f"window_{window_idx:03d}"
        optimizer = CompositeBayesianOptimizer(
            config_path=self.config_path,
            start_date=train_start.strftime("%Y-%m-%d"),
            end_date=train_end.strftime("%Y-%m-%d"),
            n_trials=self.n_trials,
            workers=self.workers,
            seed=self.seed + window_idx,
            output_dir=str(window_output_dir),
            fee_ratio=self.fee_ratio,
            tax_ratio=self.tax_ratio,
            market_data=self.market_data,
            market_data_pickle_path=str(self.market_data_pickle_path),
            allow_market_data_fetch=False,
            objective_name=self.objective_name,
        )
        optimizer.run()
        flat_params = optimizer.best_report["best_params"]
        return optimizer._flat_params_to_dataclass(flat_params)

    # ------------------------------------------------------------------
    # OOS position
    # ------------------------------------------------------------------

    def _get_oos_position(
        self,
        best_params,
        oos_start: pd.Timestamp,
        oos_end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Run strategy with best_params and return OOS position slice."""
        import numpy as np

        start_str = oos_start.strftime("%Y-%m-%d")
        end_str = oos_end.strftime("%Y-%m-%d")

        strategy = OscarCompositeStrategy(
            config_path=self.config_path,
            market_data=self.market_data,
        )
        full_market_data = strategy._load_market_data()
        truncated = strategy._truncate_market_data_for_start_date(full_market_data, best_params, start_str)
        strategy._compute_signals(best_params, market_data_override=truncated)
        base_position = strategy._base_position.loc[start_str:end_str]
        if len(base_position.index) == 0:
            raise ValueError(f"No trading days in OOS window [{start_str}, {end_str}].")
        selected_mask = base_position.astype(bool)
        selected_count = selected_mask.sum(axis=1)
        return selected_mask.div(selected_count.replace(0, np.nan), axis=0).fillna(0.0)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        """Execute walk-forward optimization and return finlab sim report."""
        logger.info(
            "Walk-forward Composite | start=%s end=%s train=%dm oos=%dm n_trials=%d workers=%d",
            self.start_date, self.end_date,
            self.train_window_months, self.oos_window_months,
            self.n_trials, self.workers,
        )

        self._load_market_data_once()
        if not self.market_data_pickle_path.exists():
            self._persist_market_data_pickle(self.market_data)

        windows = self._generate_windows()
        if not windows:
            raise ValueError(
                f"No walk-forward windows generated. Check that end_date ({self.end_date}) "
                f"is at least {self.train_window_months} months after start_date ({self.start_date})."
            )
        logger.info("Generated %d walk-forward windows.", len(windows))

        oos_positions: list[pd.DataFrame] = []
        summary_rows: list[dict] = []

        for idx, (train_start, train_end, oos_start, oos_end) in enumerate(windows):
            logger.info(
                "Window %d/%d: train=[%s, %s]  oos=[%s, %s]",
                idx + 1, len(windows),
                train_start.date(), train_end.date(),
                oos_start.date(), oos_end.date(),
            )

            best_params = self._optimize_window(train_start, train_end, idx)
            oos_pos = self._get_oos_position(best_params, oos_start, oos_end)
            oos_positions.append(oos_pos)

            from dataclasses import asdict
            summary_rows.append({
                "window": idx,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "oos_start": str(oos_start.date()),
                "oos_end": str(oos_end.date()),
                "oos_trading_days": len(oos_pos),
                "best_params": asdict(best_params),
            })
            logger.info("Window %d OOS trading days: %d", idx + 1, len(oos_pos))

        # Save summary
        summary_path = self.output_dir / "walk_forward_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, ensure_ascii=False, indent=2)
        logger.info("Summary saved to %s", summary_path)

        # Concatenate all OOS positions
        combined = pd.concat(oos_positions).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        positions_path = self.output_dir / "walk_forward_positions.csv"
        combined.to_csv(positions_path, encoding="utf-8-sig")
        logger.info("Combined position shape: %s  saved to %s", combined.shape, positions_path)

        # Single sim call
        report = sim(
            position=combined,
            resample="D",
            upload=False,
            trade_at_price="open",
            fee_ratio=self.fee_ratio,
            tax_ratio=self.tax_ratio,
            position_limit=1.0,
        )

        report_path = self.output_dir / "walk_forward_report.html"
        report.display(save_report_path=str(report_path))
        logger.info("Walk-forward report saved to %s", report_path)

        return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Walk-forward Bayesian optimization for OscarCompositeStrategy."
    )
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--train-window-months", type=int, default=12)
    parser.add_argument("--oos-window-months", type=int, default=3)
    parser.add_argument(
        "--objective-name",
        default=ObjectiveName.ANNUAL_RETURN.value,
        choices=[e.value for e in ObjectiveName],
    )
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="assets/OscarTWStrategy/walk_forward_composite",
    )
    parser.add_argument(
        "--market-data-pickle-path",
        default=str(DEFAULT_MARKET_DATA_PICKLE),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    wf = WalkForwardComposite(
        config_path=args.config_path,
        start_date=args.start_date,
        end_date=args.end_date,
        train_window_months=args.train_window_months,
        oos_window_months=args.oos_window_months,
        n_trials=args.n_trials,
        workers=args.workers,
        seed=args.seed,
        output_dir=args.output_dir,
        fee_ratio=0.001425,
        tax_ratio=0.003,
        market_data_pickle_path=args.market_data_pickle_path,
        objective_name=ObjectiveName(args.objective_name),
    )
    report = wf.run()
    print(f"Walk-forward Composite complete. Output dir: {wf.output_dir}")
    return report


if __name__ == "__main__":
    main()
