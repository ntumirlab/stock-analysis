"""
GoldenAI 回測執行腳本（支援補跑歷史日期）

用法：
  # 今天
  python run_golden_ai_backtest.py --strategy weekly

  # 補跑指定日期
  python run_golden_ai_backtest.py --strategy weekly --backtest_date 2025-04-23

  # 補跑前 N 天（每天各存一筆 DB 記錄）
  python run_golden_ai_backtest.py --strategy weekly --days 30
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from utils.config_loader import ConfigLoader
from dao.golden_ai_backtest_metrics_dao import GoldenAIBacktestMetricsDAO


TZ = ZoneInfo("Asia/Taipei")


def _run_one_date(strategy_cls, strategy_class_name, date_str, root_dir):
    """指定日期跑一次完整回測並存報告"""
    override_params = {'backtest_date': date_str}
    strategy = strategy_cls(override_params=override_params)

    report = strategy.run_strategy()

    report_dir = os.path.join(root_dir, "assets", strategy_class_name)
    os.makedirs(report_dir, exist_ok=True)
    save_path = os.path.join(report_dir, f"{date_str}_00-00-00.html")
    report.display(save_report_path=save_path)
    print(f"  -> 報告已儲存：{save_path}")


def main():
    parser = argparse.ArgumentParser(description="執行 GoldenAI 回測（支援補跑歷史日期）")
    parser.add_argument(
        "--strategy", required=True, choices=["weekly", "monthly"],
        help="策略類型：weekly 或 monthly"
    )
    parser.add_argument(
        "--backtest_date", type=str, default=None,
        help="指定單一回測日期 YYYY-MM-DD（與 --days 擇一使用）"
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="補跑最近 N 天（從今天往回算，每天各存一筆 DB 記錄）"
    )
    args = parser.parse_args()

    if args.backtest_date and args.days:
        print("錯誤：--backtest_date 與 --days 不能同時使用")
        sys.exit(1)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    config_loader = ConfigLoader(os.path.join(root_dir, "config.yaml"))
    config_loader.load_global_env_vars()

    if args.strategy == "weekly":
        from strategy_class.golden_ai_tw_strategy_weekly import GoldenAITWStrategyWeekly as strategy_cls
        strategy_class_name = "GoldenAITWStrategyWeekly"
    else:
        from strategy_class.golden_ai_tw_strategy_monthly import GoldenAITWStrategyMonthly as strategy_cls
        strategy_class_name = "GoldenAITWStrategyMonthly"

    today = datetime.now(TZ).date()

    dao = GoldenAIBacktestMetricsDAO()
    strategy_key = "weekly" if args.strategy == "weekly" else "monthly"

    if args.days:
        dates = [
            (today - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(args.days, 0, -1)
        ]
        print(f"補跑 {args.days} 天：{dates[0]} ～ {dates[-1]}")
        for i, date_str in enumerate(dates, 1):
            if dao.exists_for_date(date_str, strategy_key):
                print(f"[{i}/{args.days}] {date_str} 已存在，跳過")
                continue
            print(f"\n[{i}/{args.days}] 回測日期：{date_str}")
            _run_one_date(strategy_cls, strategy_class_name, date_str, root_dir)
    else:
        date_str = args.backtest_date or today.strftime("%Y-%m-%d")
        if dao.exists_for_date(date_str, strategy_key):
            print(f"{date_str} 已存在 DB，跳過（如需重跑請先手動刪除該筆記錄）")
        else:
            print(f"回測日期：{date_str}")
            _run_one_date(strategy_cls, strategy_class_name, date_str, root_dir)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
