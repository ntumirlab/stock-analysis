import os
import logging
import tempfile
import numpy as np
import pandas as pd
from finlab import data
from finlab.backtest import sim
from finlab.dataframe import FinlabDataFrame
from strategy_class.golden_ai_tw_strategy_base import GoldenAITWStrategyBase, MultiReportWrapper, _extract_report_json
from markets.target_weekday_tw_market import TargetWeekdayTWMarket
from core.tranche_schedule import NUM_TRANCHES, tranche_sundays

logger = logging.getLogger(__name__)


class GoldenAITWStrategyMonthly(GoldenAITWStrategyBase):
    def __init__(self, config_path="config.yaml", override_params=None):
        super().__init__(task_name="monthly", config_path=config_path, override_params=override_params)

    def _run_core(self, ranks):
        """月策略核心：對給定 ranks 跑四份 tranche，回傳 {'tranche1': report, ...}"""
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

            pre_raw_low, pre_raw_high = None, None
            if use_db_sl_tp:
                pre_raw_low  = data.get('price:最低價').reindex(index=base_position.index, columns=base_position.columns)
                pre_raw_high = data.get('price:最高價').reindex(index=base_position.index, columns=base_position.columns)

            reports = {}
            for offset in range(NUM_TRANCHES):
                # 進場週由錨點連續輪動決定，與實盤的 tranche 排程同一套定義
                selected_weeks = tranche_sundays(self.task_name, base_position.index, offset + 1)
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
                final_position = final_position.shift(-1).ffill().fillna(False).astype(bool)
                final_position = self._apply_cutoff(final_position, universe.index)

                if use_touched_exit:
                    report = sim(
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
                    report = sim(
                        position=final_position,
                        fee_ratio=1.425/1000,
                        tax_ratio=3/1000,
                        market=TargetWeekdayTWMarket(buy_weekday=self.buy_weekday, backtest_date=self.backtest_date),
                        trade_at_price=self.trade_at_price,
                        resample=None,
                        upload=False,
                        notification_enable=False
                    )
                reports[f"tranche{offset + 1}"] = report

            return reports
        finally:
            data.truncate_end = None

    def _run_one_ranks(self, ranks, dao, timestamp, date_str, time_str, report_dir, i, total):
        ranks_str = ','.join(map(str, ranks))
        if dao.exists_for_date(date_str, self.task_name, ranks_str):
            print(f"[{i}/{total}] Ranks[{ranks_str}] 已存在，跳過")
            return
        print(f"[{i}/{total}] 回測 Ranks[{ranks_str}]...")
        tranche_reports = self._run_core(ranks=ranks)
        for tranche_name, report in tranche_reports.items():
            dao.save(timestamp=timestamp, strategy=self.task_name, tranche=tranche_name, ranks=ranks_str, report=report)

        if report_dir is not None:
            wrapper = MultiReportWrapper(tranche_reports)
            save_path = os.path.join(report_dir, f"{date_str}_{time_str}_Ranks[{ranks_str}].html")
            if self.backtest_date is not None:
                data.truncate_end = self.backtest_date.strftime('%Y-%m-%d')
            try:
                wrapper.display(save_report_path=save_path)
            finally:
                data.truncate_end = None
            base_dir, file_name = os.path.split(save_path)
            file_base, ext = os.path.splitext(file_name)
            for tranche_name in tranche_reports:
                tranche_path = os.path.join(base_dir, f"{file_base}_{tranche_name}{ext}")
                rj, pj = _extract_report_json(tranche_path)
                if rj:
                    dao.save_report(timestamp=timestamp, strategy=self.task_name, tranche=tranche_name,
                                    ranks=ranks_str, report_json=rj, position_json=pj)
                else:
                    logger.warning(
                        f"報告資料抽取失敗（finlab 輸出格式可能已變），未存入 DB: "
                        f"{self.task_name} {tranche_name} Ranks[{ranks_str}] @ {timestamp}"
                    )
        else:
            for tranche_name, report in tranche_reports.items():
                tmp = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
                tmp_path = tmp.name
                tmp.close()
                if self.backtest_date is not None:
                    data.truncate_end = self.backtest_date.strftime('%Y-%m-%d')
                try:
                    report.display(save_report_path=tmp_path)
                finally:
                    data.truncate_end = None
                rj, pj = _extract_report_json(tmp_path)
                os.unlink(tmp_path)
                if rj:
                    dao.save_report(timestamp=timestamp, strategy=self.task_name, tranche=tranche_name,
                                    ranks=ranks_str, report_json=rj, position_json=pj)
                else:
                    logger.warning(
                        f"報告資料抽取失敗（finlab 輸出格式可能已變），未存入 DB: "
                        f"{self.task_name} {tranche_name} Ranks[{ranks_str}] @ {timestamp}"
                    )


if __name__ == '__main__':
    strategy = GoldenAITWStrategyMonthly()
    strategy.run_strategy()
    report = strategy.get_report()
