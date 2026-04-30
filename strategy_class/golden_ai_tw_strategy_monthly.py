import os
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from finlab import data
from finlab.backtest import sim
from finlab.dataframe import FinlabDataFrame
from strategy_class.golden_ai_tw_strategy_base import GoldenAITWStrategyBase, MultiReportWrapper
from dao.golden_ai_backtest_metrics_dao import GoldenAIBacktestMetricsDAO
from markets.target_weekday_tw_market import TargetWeekdayTWMarket


class GoldenAITWStrategyMonthly(GoldenAITWStrategyBase):
    def __init__(self, config_path="config.yaml", override_params=None):
        super().__init__(task_name="monthly", config_path=config_path, override_params=override_params)

    def _get_nth_sundays(self, date_range, n):
        """每個月第 n 個週日（n 從 1 開始）；若該月不足 n 個週日則跳過"""
        results = []
        months = pd.period_range(start=date_range.min(), end=date_range.max(), freq='M')
        for month in months:
            first_day = month.to_timestamp()
            days_to_sunday = (6 - first_day.weekday()) % 7
            first_sunday = first_day + pd.Timedelta(days=days_to_sunday)
            nth_sunday = first_sunday + pd.Timedelta(weeks=n - 1)
            if nth_sunday.month == first_day.month:
                results.append(nth_sunday)
        return pd.DatetimeIndex(results)

    def _run_core(self, ranks):
        """月策略核心：對給定 ranks 跑 Week1~4，回傳 {'Week1': report, ...}"""
        universe = data.get('price:收盤價')
        if self.backtest_date is not None:
            universe = universe[universe.index <= self.backtest_date]
        base_position, sl_df, tp_df = self._create_df(universe, ranks=ranks)

        use_db_sl_tp = self.use_db_sl or self.use_db_tp
        use_touched_exit = (
            not use_db_sl_tp
            and (self.global_sl is not None or self.global_tp is not None)
        )

        pre_raw_low, pre_raw_high = None, None
        if not use_touched_exit and (self.use_db_sl or self.use_db_tp):
            pre_raw_low  = data.get('price:最低價').reindex(index=base_position.index, columns=base_position.columns)
            pre_raw_high = data.get('price:最高價').reindex(index=base_position.index, columns=base_position.columns)

        reports = {}
        for offset in range(4):
            selected_weeks = self._get_nth_sundays(base_position.index, offset + 1)
            entry_dates = selected_weeks + pd.Timedelta(days=1 + self.buy_weekday)
            exit_dates  = selected_weeks + pd.Timedelta(days=22 + self.sell_weekday)

            entry_mask = base_position.index.isin(entry_dates)
            entries = base_position & entry_mask[:, np.newaxis]

            if use_touched_exit:
                sl_tp_exits = pd.DataFrame(False, index=base_position.index, columns=base_position.columns)
            else:
                sl_tp_exits = self._build_sl_tp_exits(
                    entries, base_position, sl_df, tp_df,
                    raw_low=pre_raw_low, raw_high=pre_raw_high
                )

            exit_mask = base_position.index.isin(exit_dates)
            normal_exits = pd.DataFrame(
                np.broadcast_to(exit_mask[:, np.newaxis], base_position.shape).copy(),
                index=base_position.index,
                columns=base_position.columns
            )

            if self.buy_weekday == self.sell_weekday:
                normal_exits = normal_exits & ~entries

            exits = FinlabDataFrame(normal_exits | sl_tp_exits)
            final_position = FinlabDataFrame(entries).hold_until(exits)
            final_position = final_position.shift(-1).fillna(False).astype(bool)
            final_position = self._apply_cutoff(final_position)

            if use_touched_exit:
                report = sim(
                    position=final_position,
                    stop_loss=self.global_sl,
                    take_profit=self.global_tp,
                    touched_exit=True,
                    fee_ratio=1.425/1000,
                    tax_ratio=3/1000,
                    market=TargetWeekdayTWMarket(buy_weekday=self.buy_weekday),
                    trade_at_price=self.trade_at_price,
                    resample=None,
                    upload=False,
                    notification_enable=False
                )
            else:
                report = sim(
                    position=final_position,
                    fee_ratio=1.425/1000,
                    tax_ratio=3/1000,
                    market=TargetWeekdayTWMarket(buy_weekday=self.buy_weekday),
                    trade_at_price=self.trade_at_price,
                    resample=None,
                    upload=False,
                    notification_enable=False
                )
            reports[f"Week{offset + 1}"] = report

        return reports

    def run_strategy(self, report_dir=None):
        from itertools import combinations as _combinations

        dao = GoldenAIBacktestMetricsDAO()
        if self.backtest_date is not None:
            timestamp = self.backtest_date.strftime("%Y-%m-%d %H:%M:%S")
        else:
            timestamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")

        date_str = timestamp[:10]
        time_str = timestamp[11:].replace(':', '-')

        if report_dir is not None:
            os.makedirs(report_dir, exist_ok=True)

        ranks_pool = list(range(self.rank_start, self.rank_end + 1))
        all_subsets = [list(c) for r in range(1, len(ranks_pool) + 1) for c in _combinations(ranks_pool, r)]
        total = len(all_subsets)
        print(f"開始執行 {total} 組 Ranks × Week1~4 回測（Rank {self.rank_start}~{self.rank_end}）...")

        for i, ranks in enumerate(all_subsets, 1):
            ranks_str = ','.join(map(str, ranks))
            if dao.exists_for_date(date_str, 'monthly', ranks_str):
                print(f"[{i}/{total}] Ranks[{ranks_str}] 已存在，跳過")
                continue
            print(f"[{i}/{total}] 回測 Ranks[{ranks_str}]...")
            week_reports = self._run_core(ranks=ranks)
            for week_name, report in week_reports.items():
                dao.save(timestamp=timestamp, strategy='monthly', week=week_name, ranks=ranks_str, report=report)
            if report_dir is not None:
                wrapper = MultiReportWrapper(week_reports)
                save_path = os.path.join(report_dir, f"{date_str}_{time_str}_Ranks[{ranks_str}].html")
                wrapper.display(save_report_path=save_path)

        print("全部完成。")


if __name__ == '__main__':
    strategy = GoldenAITWStrategyMonthly()
    strategy.run_strategy()
    report = strategy.get_report()
