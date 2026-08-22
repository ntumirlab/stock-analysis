"""GoldenAI 單期清單（節點）回測。

一個節點 = 一份推薦清單 × 一組 ranks 的一次獨立回測。與 backtest_executor 的差別：
那支跑的是「今天回頭看三個月」的滾動回測，同一份清單每天都會被重算；這支算的是
「某一份清單的單期結果」，結算後永不改變，所以重跑會被 UNIQUE 索引擋下、天生
idempotent。也因此它可以每晚跑而不必擔心重複。

日期運算在 core/node_backtest.py（純運算、CI 有測），這裡只負責建 position、跑 sim、
把結果交給 DAO。

排程（docker/crontab）用 --days，只看最近結算的那幾個：
    python -m jobs.golden_ai_node_executor --strategy weekly --days 14 --all-ranks

手動補歷史或查單一節點：
    python -m jobs.golden_ai_node_executor \
        --strategy weekly --list-date 2026-08-02 2026-08-09 --dry-run
    python -m jobs.golden_ai_node_executor \
        --strategy weekly --date-range 2025-09-24 2026-08-09 --all-ranks

**日期一定要給一個**，沒有「全部」這個預設——每一次執行的範圍都必須是明寫出來的。
排程用的 --days 也因此是有界的：即使表還是空的，第一晚也只算最近 14 天內結算的節點
（約 13 分鐘），不會變成一次全量回填（那要約 14 小時，請手動跑 --date-range）。
"""

import argparse
import logging
import os
import traceback
from itertools import combinations

import numpy as np
import pandas as pd

from finlab import data
from finlab.backtest import sim
from finlab.dataframe import FinlabDataFrame

from core.node_backtest import (
    HOLD_WEEKS, align_to_sunday, check_trades, is_settled,
    node_dates, node_return, node_window,
)
from core.tranche_schedule import tranche_of
from dao.golden_ai_backtest_nodes_dao import GoldenAIBacktestNodesDAO
from dao.recommendation_dao import RecommendationDAO
from markets.target_weekday_tw_market import TargetWeekdayTWMarket
from strategy_class.golden_ai_tw_strategy_monthly import GoldenAITWStrategyMonthly
from strategy_class.golden_ai_tw_strategy_weekly import GoldenAITWStrategyWeekly
from strategy_class.golden_ai_tw_strategy_weekly_4w import GoldenAITWStrategyWeekly4W
from utils.authentication import Authenticator
from utils.config_loader import ConfigLoader
from utils.notifier import create_notification_manager

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.yaml')

STRATEGY_CLASS_MAP = {
    'weekly': GoldenAITWStrategyWeekly,
    'monthly': GoldenAITWStrategyMonthly,
    'weekly_4w': GoldenAITWStrategyWeekly4W,
}

FULL_RANKS = '1,2,3,4,5,6,7,8'


def parse_args():
    parser = argparse.ArgumentParser(description='GoldenAI 單期清單（節點）回測')
    parser.add_argument('--strategy', required=True, choices=list(STRATEGY_CLASS_MAP))

    # 必填：三種都是有界的。沒有「全部」這個選項，排程才不會在初始回填之前
    # 誤觸整段歷史
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--list-date', nargs='+', metavar='YYYY-MM-DD',
                       help='只補這幾份清單（清單日＝對齊後的週日）')
    group.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                       help='只補這個區間內的清單日')
    group.add_argument('--days', type=int, nargs='?', const=14, metavar='N',
                       help='只補最近 N 天內結算的節點（不給數字＝14），排程用')

    ranks = parser.add_mutually_exclusive_group()
    ranks.add_argument('--ranks', default=FULL_RANKS, metavar='1,2,3',
                       help=f'單一組合（預設 {FULL_RANKS}）')
    ranks.add_argument('--all-ranks', action='store_true',
                       help='跑 1~8 的完整 powerset（255 組）')

    parser.add_argument('--db', default=os.path.join(PROJECT_ROOT, 'data_prod.db'),
                        help='節點寫到哪個 DB。推薦清單與行情不受它影響，'
                             '一律跟策略走預設路徑')
    parser.add_argument('--dry-run', action='store_true', help='只印結果，不寫 DB')
    return parser.parse_args()


def real_list_dates(strategy) -> set:
    """實際存在的清單日（對齊到週日）。

    **不能從 position 的 index 推**：`_create_df` 會 `resample('D').ffill()`，缺漏的那一週
    會被前一份清單填滿，照著跑就會生出一個內容與上週完全相同的假節點。實測 2026-08-16
    沒有清單，position 上卻有 8/09 的那八檔。
    """
    records = RecommendationDAO(frequency=strategy.recommendation_frequency).load()
    return {align_to_sunday(r.date) for r in records if r.date and r.stocks}


def all_rank_combos(rank_start: int, rank_end: int) -> list:
    pool = list(range(rank_start, rank_end + 1))
    return [','.join(map(str, c))
            for r in range(1, len(pool) + 1)
            for c in combinations(pool, r)]


def run_node(strategy, position_all, universe_index, list_date, ranks_str, hold_weeks):
    """跑一個節點。回不出乾淨的節點就回 (None, 原因)。"""
    list_date = pd.Timestamp(list_date)
    entry_date, exit_date = node_dates(
        list_date, strategy.buy_weekday, strategy.sell_weekday, hold_weeks)

    # 結算檢查放最前面：最新一份清單可能比市場資料還新（週日出清單、行情只到週五），
    # 那種情況 position 會被裁到資料尾端、index 裡根本沒有那個清單日。
    if not is_settled(exit_date, universe_index):
        # 還沒結算不是錯誤，只是還輪不到它——回填一整段時最後幾期本來就會落在這裡
        return None, f'pending until {exit_date.date()} settles'
    if list_date not in position_all.index:
        return None, f'pending until {exit_date.date()}'

    n_stocks = int(position_all.loc[list_date].sum())
    if n_stocks == 0:
        # 這組 ranks 在這份清單上取不到股票（清單不滿 8 檔時，例如只選第 8 名）。
        # 跑 --all-ranks 時這是常態，不是錯誤。
        return None, 'no position for these ranks'

    # 只有這一份清單的進場訊號，所以不可能跟其他週的部位淨換倉
    entries = position_all & (position_all.index == entry_date)[:, np.newaxis]
    exits = pd.DataFrame(
        np.broadcast_to((position_all.index == exit_date)[:, np.newaxis],
                        position_all.shape).copy(),
        index=position_all.index, columns=position_all.columns)

    final = FinlabDataFrame(entries).hold_until(FinlabDataFrame(exits))
    final = final.shift(-1).ffill().fillna(False).astype(bool)

    start, end = node_window(final, entry_date, universe_index, exit_date)
    final = final[(final.index >= start) & (final.index <= end)]

    report = sim(
        position=final,
        fee_ratio=1.425 / 1000,
        tax_ratio=3 / 1000,
        market=TargetWeekdayTWMarket(buy_weekday=strategy.buy_weekday),
        trade_at_price=strategy.trade_at_price,
        resample=None,
        upload=False,
        notification_enable=False,
    )

    trades = report.trades
    problem = check_trades(trades, entry_date, exit_date, n_stocks)
    if problem:
        return None, problem

    return {
        'strategy': strategy.task_name,
        'list_date': list_date.strftime('%Y-%m-%d'),
        'ranks': ranks_str,
        'entry_date': pd.Timestamp(trades['entry_date'].iloc[0]).strftime('%Y-%m-%d'),
        'exit_date': pd.Timestamp(trades['exit_date'].iloc[0]).strftime('%Y-%m-%d'),
        # 只有四週策略需要分相位；weekly 一週一輪，這個維度不存在
        'tranche': tranche_of(strategy.task_name, list_date) if hold_weeks > 1 else None,
        'n_stocks': n_stocks,
        'node_return': node_return(trades),
        'report': report,
        'window': (start, end),
    }, None


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    logging.getLogger('finlab').setLevel(logging.ERROR)

    config = ConfigLoader(CONFIG_PATH)
    config.load_global_env_vars()
    Authenticator(config).login_finlab()

    strategy = STRATEGY_CLASS_MAP[args.strategy](config_path=CONFIG_PATH)
    hold_weeks = HOLD_WEEKS[args.strategy]
    universe = data.get('price:收盤價')

    rank_combos = (all_rank_combos(strategy.rank_start, strategy.rank_end)
                   if args.all_ranks else [args.ranks])

    # 錨點連續輪動下每個週日都恰好被一份 tranche 買到，所以每份清單都該有節點
    available = real_list_dates(strategy)
    if args.list_date:
        list_dates = []
        for raw in args.list_date:
            d = align_to_sunday(raw)
            # 清單日是週日；給的若是別天會被對齊過去。講出來，免得把出場日當成清單日
            if d != pd.Timestamp(raw):
                logger.info(f'{raw} is not a Sunday, aligned to list date {d.date()}')
            if d in available:
                list_dates.append(d)
            else:
                logger.warning(f'{d.date()}: no recommendation list, skipped')
    elif args.date_range:
        lo, hi = (pd.Timestamp(d) for d in args.date_range)
        list_dates = sorted(d for d in available if lo <= d <= hi)
    else:
        # 按**出場日**篩而不是清單日：4W 策略今天結算的節點，清單日在四週前。
        # 視窗要比一天寬——休市會讓成交順延，機器停一天也要補得回來。
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=args.days)
        list_dates = sorted(
            d for d in available
            if node_dates(d, strategy.buy_weekday, strategy.sell_weekday,
                          hold_weeks)[1] >= cutoff)
    logger.info(f'{len(list_dates)} list dates x {len(rank_combos)} ranks combos')

    dao = None if args.dry_run else GoldenAIBacktestNodesDAO(db_path=args.db)

    # position 只跟 ranks 有關，同一組 ranks 的所有清單日共用一次建構
    ok = skipped = pending = empty = failed = 0
    for ranks_str in rank_combos:
        ranks = [int(r) for r in ranks_str.split(',')]
        position_all, _, _ = strategy._create_df(universe, ranks=ranks)

        stored = (dao.stored_list_dates(strategy.task_name, ranks_str)
                  if dao is not None else set())

        for list_date in list_dates:
            key = f'{strategy.task_name} {list_date.date()} Ranks[{ranks_str}]'
            if list_date.strftime('%Y-%m-%d') in stored:
                skipped += 1
                continue

            node, problem = run_node(
                strategy, position_all, universe.index, list_date, ranks_str, hold_weeks)
            if node is None:
                if problem.startswith('pending'):
                    logger.info(f'{key}: {problem}')
                    pending += 1
                elif problem.startswith('no position'):
                    logger.debug(f'{key}: {problem}')
                    empty += 1
                else:
                    logger.warning(f'{key}: skipped — {problem}')
                    failed += 1
                continue

            start, end = node.pop('window')
            report = node.pop('report')
            logger.info(
                f'{key}: {node["entry_date"]} ~ {node["exit_date"]}  '
                f'{node["n_stocks"]} 檔  節點報酬 {node["node_return"]:+.4%}  '
                f'(視窗 {start.date()} ~ {end.date()})')

            if dao is None:
                ok += 1
            # exists() 之後仍可能是既有的（同一批跑兩次），以 DAO 的回傳為準
            elif dao.save(report=report, **node):
                ok += 1
            else:
                skipped += 1

    verb = 'computed (dry run, nothing written)' if args.dry_run else 'saved'
    logger.info(f'done — {ok} {verb}, already stored {skipped}, '
                f'not settled yet {pending}, no position {empty}, failed {failed}')


if __name__ == '__main__':
    # 與其他排程 job 一致（見 jobs/backtest_executor.py）：整支掛掉要發 Telegram，
    # 否則「單期清單」只會靜靜地停止更新，沒有人會發現。
    _notifier = create_notification_manager(
        ConfigLoader(CONFIG_PATH).config.get('notification', {}), logger)
    try:
        main()
    except Exception as e:
        logger.exception(e)
        _notifier.send_error(
            task_name='單期清單（節點）回測',
            error_message=str(e),
            error_traceback=traceback.format_exc(),
        )
        raise SystemExit(1)
