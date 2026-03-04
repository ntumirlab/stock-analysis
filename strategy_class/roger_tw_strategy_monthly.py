import os
import pandas as pd
from finlab import data
from finlab.backtest import sim
from strategy_class.roger_tw_strategy_base import RogerTWStrategyBase
from markets.target_weekday_tw_market import TargetWeekdayTWMarket

class MultiReportWrapper:
    def __init__(self, reports_dict):
        self.reports_dict = reports_dict

    def display(self, save_report_path=None, **kwargs):
        """
        將多個回測報告分別儲存到不同的檔案，以不同後綴區分
        """
        base_dir, file_name = os.path.split(save_report_path)
        file_base, ext = os.path.splitext(file_name)
        
        for name, report in self.reports_dict.items():
            new_file_name = f"{file_base}_{name}{ext}"
            new_path = os.path.join(base_dir, new_file_name)
            
            print(f"[{name}] 儲存報告至: {new_path}")
            report.display(save_report_path=new_path, **kwargs)

class RogerTWStrategyMonthly(RogerTWStrategyBase):
    def __init__(self, config_path="config.yaml", override_params=None):
        super().__init__(task_name="monthly", config_path=config_path, override_params=override_params)

    def _apply_trading_window(self, position, selected_weeks):
        """
        月循環的交易視窗，確保在「第一週的買入日」抱到「第四週的賣出日」
        """
        cycle_starts = pd.Series(index=position.index, data=pd.NaT)
        valid_starts = selected_weeks[selected_weeks.isin(position.index)]
        cycle_starts.loc[valid_starts] = valid_starts
        cycle_starts = cycle_starts.ffill()
        
        days_since_start = (position.index - cycle_starts).dt.days
        
        buy_day = 1 + self.buy_weekday
        sell_day = 22 + self.sell_weekday
        
        mask = (days_since_start >= buy_day) & (days_since_start < sell_day)
        
        return position.mul(mask, axis=0)

    def run_strategy(self):
        universe = data.get('price:收盤價')
        
        base_position = self._create_position_df(universe)
        
        weekly_dates = base_position.resample('W').last().index
        
        reports = {}
        
        print("開始執行 4 種不同起始週的回測...")
        for offset in range(4):
            selected_weeks = weekly_dates[offset::4]
            
            pos_offset = base_position.reindex(selected_weeks)
            
            pos_offset = pos_offset.resample('D').ffill()
            
            pos_offset = pos_offset.reindex(base_position.index, method='ffill')
            
            pos_offset = pos_offset.fillna(False) 
            
            if self.buy_weekday != self.sell_weekday:
                pos_offset = self._apply_trading_window(pos_offset, selected_weeks)
            
            # 由於此策略在買賣日「前一天」即決定隔天是否買賣，因此將 position 向前移動一天
            pos_offset = pos_offset.shift(-1).fillna(False).astype(bool)
            
            print(f"-> 正在回測 Week {offset + 1}...")
            report = sim(
                position=pos_offset,
                fee_ratio=1.425/1000,
                tax_ratio=3/1000,
                stop_loss=self.stop_loss,
                take_profit=self.take_profit,
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