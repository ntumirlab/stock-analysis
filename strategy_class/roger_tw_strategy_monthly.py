import os
import numpy as np
import pandas as pd
from finlab import data
from finlab.backtest import sim
from finlab.dataframe import FinlabDataFrame
from strategy_class.roger_tw_strategy_base import RogerTWStrategyBase
from markets.target_weekday_tw_market import TargetWeekdayTWMarket

class MultiReportWrapper:
    def __init__(self, reports_dict):
        self.reports_dict = reports_dict

    def display(self, save_report_path=None, **kwargs):
        """將多個回測報告分別儲存，以不同後綴區分"""
        base_dir, file_name = os.path.split(save_report_path)
        file_base, ext = os.path.splitext(file_name)

        for name, report in self.reports_dict.items():
            new_path = os.path.join(base_dir, f"{file_base}_{name}{ext}")
            print(f"[{name}] 儲存報告至: {new_path}")
            report.display(save_report_path=new_path, **kwargs)

class RogerTWStrategyMonthly(RogerTWStrategyBase):
    def __init__(self, config_path="config.yaml", override_params=None):
        super().__init__(task_name="monthly", config_path=config_path, override_params=override_params)

    def run_strategy(self):
        universe = data.get('price:收盤價')
        base_position, sl_df, tp_df = self._create_df(universe)

        use_db_sl_tp = self.use_db_sl or self.use_db_tp
        use_touched_exit = (
            not use_db_sl_tp
            and (self.global_sl is not None or self.global_tp is not None)
        )

        # selected_weeks 為每週日（resample('W') 預設以週日為週末錨點）
        weekly_dates = base_position.resample('W').last().index

        pre_raw_low, pre_raw_high = None, None
        if not use_touched_exit and (self.use_db_sl or self.use_db_tp):
            pre_raw_low  = data.get('price:最低價').reindex(index=base_position.index, columns=base_position.columns)
            pre_raw_high = data.get('price:最高價').reindex(index=base_position.index, columns=base_position.columns)

        reports = {}
        print("開始執行 4 種不同起始週的回測...")

        for offset in range(4):
            selected_weeks = weekly_dates[offset::4]

            entry_dates = selected_weeks + pd.Timedelta(days=1 + self.buy_weekday)
            exit_dates = selected_weeks + pd.Timedelta(days=22 + self.sell_weekday)

            # entries：推薦清單內 AND 第 1 週進場日
            entry_mask = base_position.index.isin(entry_dates)
            entries = base_position & entry_mask[:, np.newaxis]

            # SL/TP 出場：touched_exit 模式下交給 sim()，hold_until 不處理
            if use_touched_exit:
                sl_tp_exits = pd.DataFrame(False, index=base_position.index, columns=base_position.columns)
            else:
                sl_tp_exits = self._build_sl_tp_exits(
                    entries, base_position, sl_df, tp_df,
                    raw_low=pre_raw_low, raw_high=pre_raw_high
                )

            # 正常出場：第 4 週賣出日
            exit_mask = base_position.index.isin(exit_dates)
            normal_exits = pd.DataFrame(
                np.broadcast_to(exit_mask[:, np.newaxis], base_position.shape).copy(),
                index=base_position.index,
                columns=base_position.columns
            )

            # buy == sell 時，進場日不觸發正常出場（避免當天進出）
            if self.buy_weekday == self.sell_weekday:
                normal_exits = normal_exits & ~entries

            exits = FinlabDataFrame(normal_exits | sl_tp_exits)

            # hold_until → shift(-1) → sim
            final_position = FinlabDataFrame(entries).hold_until(exits)
            final_position = final_position.shift(-1).fillna(False).astype(bool)
            final_position = self._apply_cutoff(final_position)

            print(f"-> 正在回測 Week {offset + 1}...")
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

        self.report = MultiReportWrapper(reports)
        return self.report

if __name__ == '__main__':
    strategy = RogerTWStrategyMonthly()
    strategy.run_strategy()
    report = strategy.get_report()