from __future__ import annotations

import json
import os
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from strategy_class.oscar.oscar_strategy_andor import OscarAndOrStrategy


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "assets" / "OscarTWStrategy" / "andor_look_ahead_bias"


class OscarAndOrLookAheadBiasRunner:
    """Single-run artifact generator for manual look-ahead bias inspection."""

    def __init__(
        self,
        config_path: Path | None = None,
        env_path: Path | None = None,
        start_date: str = "2020-01-01",
    ):
        self.config_path = config_path or (REPO_ROOT / "config.yaml")
        self.env_path = env_path or (REPO_ROOT / ".env")
        self.start_date = start_date

    @staticmethod
    def _to_json_safe(value):
        if isinstance(value, dict):
            return {str(k): OscarAndOrLookAheadBiasRunner._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [OscarAndOrLookAheadBiasRunner._to_json_safe(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if pd.isna(value) and not isinstance(value, (str, bytes)):
            return None
        return value

    @staticmethod
    def _build_equal_weight_position(base_position: pd.DataFrame) -> pd.DataFrame:
        selected_mask = base_position.astype(bool)
        selected_count = selected_mask.sum(axis=1)
        return selected_mask.div(selected_count.replace(0, np.nan), axis=0).fillna(0.0)

    def _validate_local_requirements(self) -> None:
        if not self.env_path.exists():
            raise unittest.SkipTest(".env not found; this integration test requires local credentials.")
        if not self.config_path.exists():
            raise unittest.SkipTest("config.yaml not found.")

        load_dotenv(dotenv_path=self.env_path, override=True)
        if not os.environ.get("FINLAB_API_TOKEN"):
            raise unittest.SkipTest("FINLAB_API_TOKEN missing; cannot run FinLab dry run.")

    def run_once(self) -> Path:
        self._validate_local_requirements()

        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_ROOT / f"run_{run_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        strategy = OscarAndOrStrategy(config_path=str(self.config_path))
        report = strategy.run_strategy(start_date=self.start_date)

        report_html_path = out_dir / "report.html"
        report.display(save_report_path=str(report_html_path))

        metrics_path = out_dir / "report_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self._to_json_safe(report.get_metrics()), f, ensure_ascii=False, indent=2)

        creturn = getattr(report, "creturn", None)
        if creturn is not None:
            creturn.to_csv(out_dir / "report_creturn.csv", encoding="utf-8-sig")

        base_position = strategy.base_position.loc[self.start_date:]
        if base_position.empty:
            raise AssertionError("Base position is empty after strategy run.")
        base_position.astype(int).to_csv(out_dir / "base_position_bool.csv", encoding="utf-8-sig")

        final_position = self._build_equal_weight_position(base_position)
        final_position.to_csv(out_dir / "final_position_equal_weight.csv", encoding="utf-8-sig")

        strategy.buy_signal.loc[self.start_date:].astype(int).to_csv(
            out_dir / "buy_signal_bool.csv",
            encoding="utf-8-sig",
        )
        strategy.sell_signal.loc[self.start_date:].astype(int).to_csv(
            out_dir / "sell_signal_bool.csv",
            encoding="utf-8-sig",
        )

        latest_dir = OUTPUT_ROOT / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        with open(latest_dir / "last_run_path.txt", "w", encoding="utf-8") as f:
            f.write(str(out_dir.relative_to(REPO_ROOT)))

        if not report_html_path.exists():
            raise AssertionError("Report HTML was not generated.")
        if not metrics_path.exists():
            raise AssertionError("Report metrics JSON was not generated.")

        return out_dir


class TestOscarAndOrLookAheadBias(unittest.TestCase):
    def test_single_dry_run_dump_artifacts(self) -> None:
        runner = OscarAndOrLookAheadBiasRunner()
        runner.run_once()


if __name__ == "__main__":
    runner = OscarAndOrLookAheadBiasRunner()
    output_dir = runner.run_once()
    print(output_dir)
