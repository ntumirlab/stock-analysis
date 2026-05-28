import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from finlab import data
from finlab.backtest import sim
from finlab.dataframe import FinlabDataFrame
from strategy_class.golden_ai_tw_strategy_base import GoldenAITWStrategyBase
from dao.golden_ai_backtest_metrics_dao import GoldenAIBacktestMetricsDAO
from markets.target_weekday_tw_market import TargetWeekdayTWMarket


class GoldenAITWStrategyWeekly4W(GoldenAITWStrategyBase):
    def __init__(self, config_path="config.yaml", override_params=None):
        super().__init__(task_name="weekly_4w", config_path=config_path, override_params=override_params)
        # 推薦清單共用 weekly（與 GoldenAITWStrategyWeekly 同來源）
        self.recommendation_frequency = 'weekly'

    def _get_target_entry_sunday(self):
        """從 backtest_date（或當下）反推這次要記錄的 trade 的 entry_sunday：
        - exit_date_target = today 起算「下一個 Friday >= today」
        - entry_sunday = exit_date_target - 26 天
        Trade 還沒完成時 sim 會用 today 收盤評估浮動損益。
        """
        today = self.backtest_date if self.backtest_date is not None else pd.Timestamp.today().normalize()
        days_to_friday = (4 - today.weekday()) % 7  # Mon=0, Fri=4
        exit_date_target = today + pd.Timedelta(days=days_to_friday)
        entry_sunday = exit_date_target - pd.Timedelta(days=26)
        return entry_sunday

    def _run_core(self, ranks):
        try:
            if self.backtest_date is not None:
                data.truncate_end = self.backtest_date.strftime('%Y-%m-%d')
            universe = data.get('price:收盤價')
            if self.backtest_date is not None:
                universe = universe[universe.index <= self.backtest_date]
            base_position, sl_df, tp_df = self._create_df(universe, ranks=ranks)

            use_db_sl_tp = self.use_db_sl or self.use_db_tp
            use_touched_exit = (
                not use_db_sl_tp
                and (self.global_sl is not None or self.global_tp is not None)
            )

            entry_sunday = self._get_target_entry_sunday()
            entry_date = entry_sunday + pd.Timedelta(days=1 + self.buy_weekday)
            exit_date  = entry_sunday + pd.Timedelta(days=22 + self.sell_weekday)

            entry_mask = base_position.index.isin([entry_date])
            entries = base_position & entry_mask[:, np.newaxis]

            if use_touched_exit:
                sl_tp_exits = pd.DataFrame(False, index=base_position.index, columns=base_position.columns)
            else:
                sl_tp_exits = self._build_sl_tp_exits(entries, base_position, sl_df, tp_df)

            exit_mask = base_position.index.isin([exit_date])
            normal_exits = pd.DataFrame(
                np.broadcast_to(exit_mask[:, np.newaxis], base_position.shape).copy(),
                index=base_position.index,
                columns=base_position.columns
            )

            if self.buy_weekday == self.sell_weekday:
                normal_exits = normal_exits & ~entries

            exits = FinlabDataFrame(normal_exits | sl_tp_exits)
            final_position = FinlabDataFrame(entries).hold_until(exits)
            final_position = final_position.shift(-1).ffill().fillna(False).astype(bool)
            final_position = self._apply_cutoff(final_position)

            if use_touched_exit:
                return sim(
                    position=final_position,
                    stop_loss=self.global_sl,
                    take_profit=self.global_tp,
                    touched_exit=True,
                    fee_ratio=1.425/1000,
                    tax_ratio=3/1000,
                    market=TargetWeekdayTWMarket(buy_weekday=self.buy_weekday, backtest_date=self.backtest_date),
                    trade_at_price=self.trade_at_price,
                    resample=None,
                    upload=False,
                    notification_enable=False
                )
            else:
                return sim(
                    position=final_position,
                    fee_ratio=1.425/1000,
                    tax_ratio=3/1000,
                    market=TargetWeekdayTWMarket(buy_weekday=self.buy_weekday, backtest_date=self.backtest_date),
                    trade_at_price=self.trade_at_price,
                    resample=None,
                    upload=False,
                    notification_enable=False
                )
        finally:
            data.truncate_end = None

    def run_strategy(self, report_dir=None, num_workers=None):
        """單一 ranks [rank_start..rank_end]、不跑 combinations、不需要 multi-worker"""
        dao = GoldenAIBacktestMetricsDAO()
        if self.backtest_date is not None:
            timestamp = self.backtest_date.strftime("%Y-%m-%d %H:%M:%S")
        else:
            timestamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")

        date_str = timestamp[:10]
        time_str = timestamp[11:].replace(':', '-')

        if report_dir is not None:
            os.makedirs(report_dir, exist_ok=True)

        ranks = list(range(self.rank_start, self.rank_end + 1))
        print(f"[{self.task_name}] 策略參數: 週{'一二三四五'[self.buy_weekday]}買, 週{'一二三四五'[self.sell_weekday]}賣, Rank {self.rank_start}~{self.rank_end}")
        print(f"執行單一 Ranks 回測 (固定 [{','.join(map(str, ranks))}])...")

        t_start = time.monotonic()
        self._run_one_ranks(ranks, dao, timestamp, date_str, time_str, report_dir, 1, 1)
        elapsed = time.monotonic() - t_start
        mins, secs = divmod(int(elapsed), 60)
        print(f"完成。耗時 {mins} 分 {secs} 秒")


if __name__ == '__main__':
    strategy = GoldenAITWStrategyWeekly4W()
    strategy.run_strategy()
    report = strategy.get_report()
