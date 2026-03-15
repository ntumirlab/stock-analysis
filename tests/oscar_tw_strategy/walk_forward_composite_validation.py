"""Walk-forward train-test and score-decile diagnostics for OscarCompositeStrategy.

Features:
1) Walk-forward optimization:
   - train window: 12 months
    - test window: 3 months
    - roll step: 3 months
2) Per-window outputs:
   - best params json
   - test finlab report html
   - test metrics json
3) Score decile check (no fee/tax):
   - build 10 equal-weight portfolios by daily composite score deciles
    - export Train/Test cumulative return csv + smoothed line chart png

Usage:
python -m tests.oscar_tw_strategy.walk_forward_composite_validation --start-date 2020-01-01 --end-date 2025-12-31 --n-trials 200 --workers 8
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from finlab.backtest import sim
from strategy_class.oscar.oscar_strategy_composite import (
    AdjustTWMarketInfo,
    OscarCompositeStrategy,
)
from strategy_class.oscar.oscar_strategy_composite_params import OscarCompositeParams
from tests.oscar_tw_strategy.bayesian_optimize_composite_params import (
    CompositeBayesianOptimizer,
)
from tests.oscar_tw_strategy.utils.custom_report_metrics import (
    compute_total_reward_amount_from_creturn,
    get_metrics_with_fixed_annual_return,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _to_json_safe(value):
    """遞迴轉換物件為可被 JSON 序列化的型別。

    Args:
        value: 任意輸入值，可能包含 dict/list、NumPy scalar、Timestamp 或 NaN。

    Returns:
        轉換後可安全寫入 json.dump 的 Python 基本型別。
    """
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


def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
    """回傳輸入日期所在月份的月末日期（正規化為 00:00:00）。

    Args:
        ts: 輸入時間戳。

    Returns:
        月底日期的 Timestamp。
    """
    return (ts + pd.offsets.MonthEnd(0)).normalize()


def _window_ranges(
    start_date: str,
    end_date: str,
    train_months: int,
    test_months: int,
    step_months: int,
) -> list[dict]:
    """建立 Train/Test 的滾動時間視窗。

    Args:
        start_date: 實驗起始日字串（YYYY-MM-DD）。
        end_date: 實驗結束日字串（YYYY-MM-DD）。
        train_months: 每個 fold 的訓練月數。
        test_months: 每個 fold 的測試月數。
        step_months: 每次向前滾動的月數。

    Returns:
        每個 fold 的日期資訊清單，包含 train/test 的起迄日期。
    """
    windows: list[dict] = []

    cursor = pd.Timestamp(start_date).normalize()
    hard_end = pd.Timestamp(end_date).normalize()

    fold_id = 1
    while True:
        train_start = cursor
        train_end = _month_end(train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1))
        test_start = (train_end + pd.Timedelta(days=1)).normalize()
        test_end = _month_end(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1))

        if test_start > hard_end:
            break
        if test_end > hard_end:
            test_end = hard_end

        windows.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        fold_id += 1
        cursor = (cursor + pd.DateOffset(months=step_months)).normalize()

        if cursor > hard_end:
            break

    return windows


def _build_final_position(base_position: pd.DataFrame) -> pd.DataFrame:
    """將選股矩陣轉為每日等權重部位矩陣。

    Args:
        base_position: 原始選股矩陣（非零/True 代表持有）。

    Returns:
        每日權重總和為 1 的部位矩陣；若當日無持股則為 0。
    """
    selected_mask = base_position.astype(bool)
    selected_count = selected_mask.sum(axis=1)
    return selected_mask.div(
        selected_count.replace(0, np.nan),
        axis=0,
    ).fillna(0.0)


def _simulate_test_window(
    strategy: OscarCompositeStrategy,
    params: OscarCompositeParams,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    fee_ratio: float,
    tax_ratio: float,
):
    """以固定參數執行單一測試區間回測並回傳 finlab 報表。

    Args:
        strategy: 策略實例。
        params: 策略參數。
        test_start: 測試起始日。
        test_end: 測試結束日。
        fee_ratio: 手續費率。
        tax_ratio: 交易稅率。

    Returns:
        finlab 回測報表物件。
    """
    # Recompute with the same path as run_strategy and then clip to test window.
    full_market_data = strategy._load_market_data()
    truncated_market_data = strategy._truncate_market_data_for_start_date(
        full_market_data,
        params,
        str(test_start.date()),
    )
    strategy._compute_signals(params, market_data_override=truncated_market_data)

    base_position = strategy.base_position.loc[str(test_start.date()): str(test_end.date())]
    if len(base_position.index) == 0:
        raise ValueError("No test trading days in selected window.")

    final_position = _build_final_position(base_position)

    report = sim(
        position=final_position,
        resample="D",
        upload=False,
        market=AdjustTWMarketInfo(),
        trade_at_price="open",
        fee_ratio=fee_ratio,
        tax_ratio=tax_ratio,
        position_limit=1.0,
    )
    return report


def _build_decile_position(
    score_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    decile: int,
) -> pd.DataFrame:
    """建立單一分位（decile）的每日等權重部位矩陣。

    Args:
        score_df: 每日各標的分數矩陣。
        start_date: 區間起始日。
        end_date: 區間結束日。
        decile: 目標分位，範圍為 1 到 10。

    Returns:
        該分位的每日部位矩陣。
    """
    if decile < 1 or decile > 10:
        raise ValueError("decile must be in [1, 10].")

    score = score_df.loc[str(start_date.date()): str(end_date.date())]
    if score.empty:
        return pd.DataFrame(index=score.index, columns=score.columns, dtype=float).fillna(0.0)

    position = pd.DataFrame(0.0, index=score.index, columns=score.columns, dtype=float)
    for dt in score.index:
        s = score.loc[dt]
        valid = s.notna()
        if valid.sum() == 0:
            continue

        s_valid = s[valid]
        ranks = s_valid.rank(method="average", pct=True)
        deciles = np.ceil(ranks * 10).clip(1, 10).astype(int)
        members = deciles == decile
        member_symbols = deciles.index[members]
        if len(member_symbols) == 0:
            continue

        w = 1.0 / float(len(member_symbols))
        position.loc[dt, member_symbols] = w

    return position


def _simulate_position_report(
    position: pd.DataFrame,
    fee_ratio: float,
    tax_ratio: float,
):
    """使用指定部位矩陣執行 finlab 回測。

    Args:
        position: 回測部位矩陣。
        fee_ratio: 手續費率。
        tax_ratio: 交易稅率。

    Returns:
        finlab 回測報表物件。
    """
    if position.empty:
        raise ValueError("Position is empty for simulation.")

    return sim(
        position=position,
        resample="D",
        upload=False,
        market=AdjustTWMarketInfo(),
        trade_at_price="open",
        fee_ratio=fee_ratio,
        tax_ratio=tax_ratio,
        position_limit=1.0,
    )


def _extract_creturn_series(report) -> pd.Series:
    """從 finlab 報表擷取累積報酬序列。

    Args:
        report: finlab 回測報表物件。

    Returns:
        清理後的累積報酬序列；若不可用則回傳空序列。
    """
    creturn = getattr(report, "creturn", None)
    if creturn is None:
        return pd.Series(dtype=float)
    if isinstance(creturn, pd.DataFrame):
        if creturn.shape[1] == 0:
            return pd.Series(dtype=float)
        series = creturn.iloc[:, 0]
    else:
        series = pd.Series(creturn)

    series = pd.to_numeric(series, errors="coerce").dropna()
    return series


def _smooth_series(series: pd.Series, smooth_span: int) -> pd.Series:
    """以 EWM 平滑線條，降低視覺噪音。

    Args:
        series: 原始時間序列。
        smooth_span: EWM 的 span，<=1 代表不平滑。

    Returns:
        平滑後序列。
    """
    if smooth_span <= 1:
        return series
    return series.ewm(span=smooth_span, adjust=False).mean()


def _plot_decile_cumulative(
    cumret_df: pd.DataFrame,
    output_png: Path,
    title: str,
    smooth_span: int,
) -> None:
    """繪製十分位累積報酬線圖（matplotlib，含平滑）。

    Args:
        cumret_df: 欄位為 decile 的累積報酬矩陣。
        output_png: 輸出圖檔路徑。
        title: 圖表標題。
        smooth_span: 線條平滑參數。

    Returns:
        None。
    """
    if cumret_df.empty:
        output_png.with_suffix(".txt").write_text(
            f"No decile data to plot for: {title}",
            encoding="utf-8",
        )
        return

    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    plt.figure(figsize=(12, 7))
    for col in cumret_df.columns:
        smoothed = _smooth_series(cumret_df[col].astype(float), smooth_span=smooth_span)
        plt.plot(cumret_df.index, smoothed, label=col, linewidth=1.3)
    plt.title(f"{title} (smooth_span={smooth_span})")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def run_walk_forward(
    config_path: str,
    start_date: str,
    end_date: str,
    train_months: int,
    test_months: int,
    step_months: int,
    n_trials: int,
    workers: int,
    seed: int,
    output_dir: str,
    initial_capital: float,
    fee_ratio: float,
    tax_ratio: float,
    market_data_pickle_path: str,
    smooth_span: int,
    experiment_id: str,
    objective_name: str,
) -> Path:
    """執行 Train/Test 的 walk-forward 實驗並輸出結果。

    Args:
        config_path: 策略設定檔路徑。
        start_date: 實驗起始日。
        end_date: 實驗結束日。
        train_months: 訓練月數。
        test_months: 測試月數。
        step_months: 每次滾動月數。
        n_trials: 每個 fold 的 Bayesian trials。
        workers: 每個 fold 的平行 worker 數。
        seed: 隨機種子。
        output_dir: 實驗輸出根目錄。
        initial_capital: 初始資金。
        fee_ratio: 手續費率。
        tax_ratio: 交易稅率。
        market_data_pickle_path: 市場資料 pickle 路徑。
        smooth_span: 圖表平滑參數。
        experiment_id: 實驗資料夾名稱（可留空自動命名）。
        objective_name: Bayesian optimization 目標函數名稱。

    Returns:
        本次實驗輸出目錄路徑。
    """
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    if experiment_id.strip():
        experiment_name = experiment_id.strip()
    else:
        experiment_name = f"composite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    out_root = out_base / experiment_name
    out_root.mkdir(parents=True, exist_ok=True)

    windows = _window_ranges(
        start_date=start_date,
        end_date=end_date,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
    )
    if not windows:
        raise ValueError("No valid walk-forward windows were generated.")

    # Load shared market data once for all folds.
    bootstrap = OscarCompositeStrategy(config_path=config_path)
    shared_market_data = bootstrap._load_market_data()

    fold_summary_rows = []

    for window in windows:
        fold_id = window["fold_id"]
        train_start = window["train_start"]
        train_end = window["train_end"]
        test_start = window["test_start"]
        test_end = window["test_end"]

        fold_name = f"fold_{train_start:%Y%m%d}_{test_end:%Y%m%d}"
        fold_dir = out_root / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        deciles_root = fold_dir / "deciles"
        deciles_root.mkdir(parents=True, exist_ok=True)

        optimizer = CompositeBayesianOptimizer(
            config_path=config_path,
            start_date=str(train_start.date()),
            end_date=str(train_end.date()),
            objective_name=objective_name,
            n_trials=n_trials,
            workers=workers,
            seed=seed + fold_id,
            output_dir=str(fold_dir / "bayes"),
            initial_capital=initial_capital,
            fee_ratio=fee_ratio,
            tax_ratio=tax_ratio,
            market_data=shared_market_data,
            market_data_pickle_path=market_data_pickle_path,
            allow_market_data_fetch=False,
        )
        bayes_dir = optimizer.run()

        best_json_path = bayes_dir / "best_params.json"
        with open(best_json_path, "r", encoding="utf-8") as f:
            best_json = json.load(f)

        best_params = optimizer._flat_params_to_dataclass(best_json["best_params"])

        strategy = OscarCompositeStrategy(
            config_path=config_path,
            market_data=shared_market_data,
        )

        # Compute once from train_start so train/test decile diagnostics share the same signal path.
        full_market_data = strategy._load_market_data()
        truncated_market_data = strategy._truncate_market_data_for_start_date(
            full_market_data,
            best_params,
            str(train_start.date()),
        )
        strategy._compute_signals(best_params, market_data_override=truncated_market_data)

        # Use a dedicated strategy instance so test-report simulation does not
        # overwrite decile signal matrices computed from train_start.
        test_strategy = OscarCompositeStrategy(
            config_path=config_path,
            market_data=shared_market_data,
        )
        test_report = _simulate_test_window(
            strategy=test_strategy,
            params=best_params,
            test_start=test_start,
            test_end=test_end,
            fee_ratio=fee_ratio,
            tax_ratio=tax_ratio,
        )

        test_report_path = fold_dir / "test_report.html"
        test_report.display(save_report_path=str(test_report_path))

        test_metrics = get_metrics_with_fixed_annual_return(
            test_report,
            start_date=str(test_start.date()),
            end_date=str(test_end.date()),
        )
        test_total_reward = compute_total_reward_amount_from_creturn(
            creturn=getattr(test_report, "creturn", None),
            initial_capital=initial_capital,
            start_date=str(test_start.date()),
            end_date=str(test_end.date()),
        )

        with open(fold_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            payload = _to_json_safe(
                {
                    "fold_id": fold_id,
                    "train_start": str(train_start.date()),
                    "train_end": str(train_end.date()),
                    "test_start": str(test_start.date()),
                    "test_end": str(test_end.date()),
                    "total_reward_amount": test_total_reward,
                    "metrics": test_metrics,
                    "best_params": asdict(best_params),
                }
            )
            json.dump(
                payload,
                f,
                ensure_ascii=False,
                indent=2,
            )

        score_df = strategy.signal_power["composite"]

        train_curves: dict[str, pd.Series] = {}
        test_curves: dict[str, pd.Series] = {}

        for decile in range(1, 11):
            decile_name = f"decile_{decile}"
            decile_dir = deciles_root / decile_name
            decile_dir.mkdir(parents=True, exist_ok=True)

            train_position = _build_decile_position(
                score_df=score_df,
                start_date=train_start,
                end_date=train_end,
                decile=decile,
            )
            train_position.to_csv(decile_dir / "train_position.csv", encoding="utf-8-sig")
            train_report = _simulate_position_report(
                position=train_position,
                fee_ratio=0.0,
                tax_ratio=0.0,
            )
            train_report.display(save_report_path=str(decile_dir / "train_report.html"))
            train_curves[decile_name] = _extract_creturn_series(train_report)

            test_position = _build_decile_position(
                score_df=score_df,
                start_date=test_start,
                end_date=test_end,
                decile=decile,
            )
            test_position.to_csv(decile_dir / "test_position.csv", encoding="utf-8-sig")
            test_report_decile = _simulate_position_report(
                position=test_position,
                fee_ratio=0.0,
                tax_ratio=0.0,
            )
            test_report_decile.display(save_report_path=str(decile_dir / "test_report.html"))
            test_curves[decile_name] = _extract_creturn_series(test_report_decile)

        train_curve_df = pd.concat(train_curves, axis=1).sort_index().ffill().dropna(how="all")
        test_curve_df = pd.concat(test_curves, axis=1).sort_index().ffill().dropna(how="all")

        train_curve_df.to_csv(fold_dir / "decile_train_cumret.csv", encoding="utf-8-sig")
        test_curve_df.to_csv(fold_dir / "decile_test_cumret.csv", encoding="utf-8-sig")

        _plot_decile_cumulative(
            train_curve_df,
            output_png=fold_dir / "decile_train_cumret.png",
            title=f"Fold {fold_id} Train Composite Score Deciles ({train_start.date()} to {train_end.date()})",
            smooth_span=smooth_span,
        )
        _plot_decile_cumulative(
            test_curve_df,
            output_png=fold_dir / "decile_test_cumret.png",
            title=f"Fold {fold_id} Test Composite Score Deciles ({test_start.date()} to {test_end.date()})",
            smooth_span=smooth_span,
        )

        fold_summary_rows.append(
            {
                "fold_id": fold_id,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "test_total_reward_amount": test_total_reward,
                "test_annual_return": test_metrics.get("profitability", {}).get("annualReturn"),
                "test_sharpe": test_metrics.get("ratio", {}).get("sharpeRatio"),
                "test_max_drawdown": test_metrics.get("risk", {}).get("maxDrawdown"),
                "bayes_dir": str(bayes_dir),
                "test_report": str(test_report_path),
                "deciles_dir": str(deciles_root),
                "decile_train_csv": str(fold_dir / "decile_train_cumret.csv"),
                "decile_train_png": str(fold_dir / "decile_train_cumret.png"),
                "decile_test_csv": str(fold_dir / "decile_test_cumret.csv"),
                "decile_test_png": str(fold_dir / "decile_test_cumret.png"),
            }
        )

    summary_df = pd.DataFrame(fold_summary_rows)
    summary_df.to_csv(out_root / "walk_forward_summary.csv", index=False, encoding="utf-8-sig")
    summary_df.to_html(out_root / "walk_forward_summary.html", index=False)

    return out_root


def _build_parser() -> argparse.ArgumentParser:
    """建立命令列參數解析器。

    Args:
        無。

    Returns:
        argparse.ArgumentParser 物件。
    """
    parser = argparse.ArgumentParser(description="Walk-forward train-test for Oscar composite strategy.")
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--step-months", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="assets/OscarTWStrategy/composite_walk_forward")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--fee-ratio", type=float, default=0.001425)
    parser.add_argument("--tax-ratio", type=float, default=0.003)
    parser.add_argument(
        "--smooth-span",
        type=int,
        default=12,
        help="EWM span used to smooth decile cumulative-return lines for plotting.",
    )
    parser.add_argument(
        "--market-data-pickle-path",
        default=str(REPO_ROOT / "finlab_db" / "workspace" / "oscar_composite_market_data.pkl"),
    )
    parser.add_argument(
        "--experiment-id",
        default="",
        help="Optional output experiment directory name under --output-dir.",
    )
    parser.add_argument(
        "--objective-name",
        default="train_sharpe",
        choices=["train_sharpe", "train_annual_return", "train_total_reward", "train_calmar"],
        help="Objective function used in Bayesian optimization over train window.",
    )
    return parser


def main() -> None:
    """腳本進入點：解析參數並執行 walk-forward。

    Args:
        無。

    Returns:
        None。
    """
    args = _build_parser().parse_args()
    out = run_walk_forward(
        config_path=args.config_path,
        start_date=args.start_date,
        end_date=args.end_date,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        n_trials=args.n_trials,
        workers=args.workers,
        seed=args.seed,
        output_dir=args.output_dir,
        initial_capital=args.initial_capital,
        fee_ratio=args.fee_ratio,
        tax_ratio=args.tax_ratio,
        market_data_pickle_path=args.market_data_pickle_path,
        smooth_span=args.smooth_span,
        experiment_id=args.experiment_id,
        objective_name=args.objective_name,
    )
    print(f"Walk-forward outputs: {out}")


if __name__ == "__main__":
    main()
