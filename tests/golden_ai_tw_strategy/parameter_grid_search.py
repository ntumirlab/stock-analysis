"""
strategy_grid_backtest.py

Golden AI TW 策略網格參數回測實驗腳本。
逐一枚舉所有參數組合，執行回測，並將績效指標彙整輸出為 CSV。

參數說明：
  frequency       : 策略頻率，'weekly' 或 'monthly'
  max_stocks      : 最多持有股票數
  use_db_sl       : 是否使用 DB 絕對停損價（分析師設定）
  use_db_tp       : 是否使用 DB 絕對停利價（分析師設定）
  global_sl       : 全域比例停損（None = 不啟用）；DB 有值時 DB 優先
  global_tp       : 全域比例停利（None = 不啟用）；DB 有值時 DB 優先
  trade_at_price  : 成交價模式
  buy_weekday     : 買入星期（1=週一 ... 5=週五）
  sell_weekday    : 賣出星期（1=週一 ... 5=週五）
"""

import os
import itertools
import traceback
from datetime import datetime

import pandas as pd

from strategy_class.golden_ai_tw_strategy_weekly import GoldenAITWStrategyWeekly
from strategy_class.golden_ai_tw_strategy_monthly import GoldenAITWStrategyMonthly

PARAM_GRID = {
    'frequency'       : ['weekly', 'monthly'],
    'max_stocks'      : [1, 2, 3, 4, 5, 6, 7, 8],
    'use_db_sl'       : [True, False],
    'use_db_tp'       : [True, False],
    'global_sl'       : [None, 0.05, 0.1, 0.15],
    'global_tp'       : [None, 0.15, 0.2, 0.25],
    'trade_at_price'  : ['open', 'open_close_mix'],
    'buy_weekday'     : [1],
    'sell_weekday'    : [5, 1],
}

OUTPUT_DIR = 'assets/tests/golden_ai_tw_strategy/'

def _build_combinations(grid: dict) -> list[dict]:
    """將 param_grid 展開為所有參數組合的 list。"""
    keys, values = zip(*grid.items())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _extract_metrics(report, strategy_label: str, params: dict) -> dict:
    """從單一 report 物件取出績效指標，組成一筆 result dict。"""
    try:
        metrics   = report.get_metrics()
        trades    = report.get_trades()
        profit    = metrics.get('profitability', {})
        risk      = metrics.get('risk', {})
        ratio_m   = metrics.get('ratio', {})
        winrate_m = metrics.get('winrate', {})
    except Exception:
        profit = risk = ratio_m = winrate_m = {}
        trades = []

    weekday_label = f"週{'一二三四五'[params['buy_weekday']-1]}買/週{'一二三四五'[params['sell_weekday']-1]}賣"

    return {
        'Strategy'         : strategy_label,
        'Weekdays'         : weekday_label,
        'Max Stocks'       : params['max_stocks'],
        'Trade Price'      : params['trade_at_price'],
        'Use DB SL'        : params['use_db_sl'],
        'Use DB TP'        : params['use_db_tp'],
        'Global SL'        : params['global_sl'],
        'Global TP'        : params['global_tp'],
        'Annual Return (%)': round(profit.get('annualReturn', 0) * 100, 2),
        'Sharpe Ratio'     : round(ratio_m.get('sharpeRatio', 0), 4),
        'Max Drawdown (%)' : round(risk.get('maxDrawdown', 0) * 100, 2),
        'Win Rate (%)'     : round(winrate_m.get('winRate', 0) * 100, 2),
        'Total Trades'     : len(trades),
    }


def _run_one(params: dict) -> list[dict]:
    """
    執行單一參數組合的回測，回傳 result rows（月策略會有 4 筆）。
    失敗時回傳空 list 並印出錯誤。
    """
    freq = params['frequency']

    override_params = {
        'buy_weekday'   : params['buy_weekday'],
        'sell_weekday'  : params['sell_weekday'],
        'max_stocks'    : params['max_stocks'],
        'use_db_sl'     : params['use_db_sl'],
        'use_db_tp'     : params['use_db_tp'],
        'global_sl'     : params['global_sl'],
        'global_tp'     : params['global_tp'],
        'trade_at_price': params['trade_at_price'],
    }

    try:
        if freq == 'weekly':
            strategy = GoldenAITWStrategyWeekly(override_params=override_params)
        else:
            strategy = GoldenAITWStrategyMonthly(override_params=override_params)

        report_obj = strategy.run_strategy()

        rows = []
        if hasattr(report_obj, 'reports_dict'):
            for week_name, r in report_obj.reports_dict.items():
                label = f"Monthly ({week_name})"
                rows.append(_extract_metrics(r, label, params))
        else:
            label = freq.capitalize()
            rows.append(_extract_metrics(report_obj, label, params))

        return rows

    except Exception as e:
        print(f"    [ERROR] {e}")
        traceback.print_exc()
        return []

def main():
    combinations = _build_combinations(PARAM_GRID)
    total = len(combinations)
    print(f"總共將執行 {total} 種參數組合的回測實驗...")

    all_results = []

    for i, params in enumerate(combinations, start=1):
        freq = params['frequency']
        print(
            f"\n[{i}/{total}] [{freq}] "
            f"max_stocks={params['max_stocks']} | "
            f"trade_at={params['trade_at_price']} | "
            f"use_db_sl={params['use_db_sl']} use_db_tp={params['use_db_tp']} | "
            f"global_sl={params['global_sl']} global_tp={params['global_tp']}"
        )

        rows = _run_one(params)
        all_results.extend(rows)

    df = pd.DataFrame(all_results)
    if df.empty:
        print("\n沒有產生任何結果，請檢查策略與參數設定。")
        return

    df = df.sort_values(by='Sharpe Ratio', ascending=False).reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(OUTPUT_DIR, f'golden_ai_tw_strategy_grid_backtest_{timestamp}.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n實驗完成！共產生 {len(df)} 筆紀錄，結果已儲存至 {output_path}")
    print(df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()