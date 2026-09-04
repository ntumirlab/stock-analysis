import os
import sys
import argparse
import traceback
from datetime import date, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from strategy_class.golden_ai_tw_strategy_weekly import GoldenAITWStrategyWeekly
from strategy_class.golden_ai_tw_strategy_monthly import GoldenAITWStrategyMonthly
from strategy_class.golden_ai_tw_strategy_weekly_4w import GoldenAITWStrategyWeekly4W

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.yaml')

STRATEGY_CLASS_MAP = {
    'weekly': GoldenAITWStrategyWeekly,
    'monthly': GoldenAITWStrategyMonthly,
    'weekly_4w': GoldenAITWStrategyWeekly4W,
}


def parse_args():
    parser = argparse.ArgumentParser(description='GoldenAI 回測補跑工具')
    parser.add_argument('--strategy', required=True, choices=['weekly', 'monthly', 'weekly_4w'])

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--backtest_date', metavar='YYYY-MM-DD')
    group.add_argument('--days', type=int, metavar='N', help='補跑今天往前 N 天（不含今天）')
    group.add_argument('--date-range', nargs=2, metavar=('START', 'END'), help='補跑 START~END 每天（含兩端）')

    parser.add_argument('--workers', type=int, default=1, metavar='N', help='平行 worker 數（預設 1，sequential）')
    # 預設維持完整 powerset（255 組），與排程走的路徑一致。節點制那支的預設相反
    # （單一組合），因為它從第一天就是這樣，兩邊各自照自己的歷史。
    parser.add_argument('--ranks', metavar='1,2,3',
                        help='只跑這一組（如 1,2,3,4,5,6,7,8）。不給＝跑完整 powerset '
                             '255 組，補跑一整段歷史時那是 255 倍的 sim 次數')

    return parser.parse_args()


def parse_ranks(raw):
    if raw is None:
        return None
    return [int(r) for r in raw.split(',')]


def resolve_dates(args):
    today = date.today()
    if args.backtest_date:
        return [args.backtest_date]
    elif args.days is not None:
        return [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(args.days, 0, -1)]
    else:
        start = date.fromisoformat(args.date_range[0])
        end = date.fromisoformat(args.date_range[1])
        dates, current = [], start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        return dates


def run_one(strategy_name, backtest_date, num_workers=1, ranks=None):
    strategy = STRATEGY_CLASS_MAP[strategy_name](
        config_path=CONFIG_PATH,
        override_params={'backtest_date': backtest_date},
    )
    strategy.run_strategy(num_workers=num_workers, ranks=ranks)


def main():
    args = parse_args()
    dates = resolve_dates(args)
    ranks = parse_ranks(args.ranks)

    scope = f"Ranks[{args.ranks}] 一組" if ranks else "完整 powerset 255 組"
    print(f"補跑 {args.strategy}，共 {len(dates)} 天：{dates[0]} ~ {dates[-1]}（{scope}）")

    success, failed = 0, []
    for d in dates:
        print(f"\n{'='*60}")
        print(f"[{d}] 開始補跑")
        try:
            run_one(args.strategy, d, num_workers=args.workers, ranks=ranks)
            success += 1
            print(f"[{d}] 完成")
        except Exception:
            traceback.print_exc()
            failed.append(d)

    print(f"\n{'='*60}")
    print(f"完成。成功 {success} 天，失敗 {len(failed)} 天")
    if failed:
        print(f"失敗日期：{failed}")


if __name__ == '__main__':
    main()
