import logging
import numpy as np
import pandas as pd
from finlab import data
from finlab.backtest import sim
from finlab.dataframe import FinlabDataFrame
from strategy_class.golden_ai_tw_strategy_base import GoldenAITWStrategyBase
from dao.recommendation_dao import RecommendationDAO
from markets.target_weekday_tw_market import TargetWeekdayTWMarket

logger = logging.getLogger(__name__)


class GoldenAIOrderAdapter(GoldenAITWStrategyBase):
    """Reads Golden AI recommendations from DB and produces a finlab Report
    using rolling N-week hold cycles, for use with OrderExecutor.

    週期以 cycle_start_date（錨點）純日期運算往後鋪排，與市場資料、
    下單紀錄無關；假日的買賣訊號交給 finlab 對齊到下一個交易日（與回測一致）。
    """

    def __init__(self, frequency='weekly', hold_weeks=1, cycle_start_date=None,
                 config_path="config.yaml", backtest_date=None):
        override_params = {'backtest_date': backtest_date} if backtest_date else None
        super().__init__(task_name=frequency, config_path=config_path,
                          override_params=override_params)
        if cycle_start_date is None:
            raise ValueError(
                "GoldenAIOrderAdapter 需要 cycle_start_date（第一個買入日，如 '2026-07-06'）"
            )
        self.cycle_start_date = pd.Timestamp(cycle_start_date).normalize()
        self.hold_weeks = hold_weeks
        self.ranks = list(range(self.rank_start, self.rank_end + 1))

    def _compute_cycles(self, until):
        """從錨點鋪出各週期的 (買入日, 賣出日)，直到 until（含）之後的第一個買入日為止。

        賣出日 = 買入日之後（含當天）的第 hold_weeks 個賣出 weekday；
        下一個買入日 = 賣出日之後的第一個買入 weekday。日期落在假日時
        由 finlab 訊號對齊處理，行程表本身不位移。
        """
        anchor = self.cycle_start_date
        entry = anchor + pd.Timedelta(days=(self.buy_weekday - anchor.dayofweek) % 7)
        sell_offset = (self.sell_weekday - self.buy_weekday) % 7
        cycles = []
        while entry <= until:
            exit_d = entry + pd.Timedelta(days=sell_offset + 7 * (self.hold_weeks - 1))
            cycles.append((entry, exit_d))
            step = (self.buy_weekday - exit_d.dayofweek) % 7
            entry = exit_d + pd.Timedelta(days=step if step else 7)
        return cycles

    def _compute_historical_cycles(self, index, before):
        """錨點之前的歷史週期（資料驅動、僅完整週期，且整個週期須在 before 之前結束）。

        僅供報告延續性：finlab Report 會把全為 1.0 的 creturn 截斷成空序列，
        導致 Portfolio 建構時 iloc[0] 崩潰，所以報告視窗內必須有可定價的既往交易。
        今日目標持股永遠由錨點行程表（_compute_cycles）決定，歷史週期不影響。
        """
        dow = index.dayofweek
        all_buy_days = index[dow == self.buy_weekday]
        all_sell_days = index[dow == self.sell_weekday]
        cycles = []
        current_entry = all_buy_days[0] if len(all_buy_days) > 0 else None
        while current_entry is not None and current_entry < before:
            sell_days_after = all_sell_days[all_sell_days >= current_entry]
            if len(sell_days_after) < self.hold_weeks:
                break
            current_exit = sell_days_after[self.hold_weeks - 1]
            if current_exit >= before:
                break
            cycles.append((current_entry, current_exit))
            next_buy_days = all_buy_days[all_buy_days > current_exit]
            current_entry = next_buy_days[0] if len(next_buy_days) > 0 else None
        return cycles

    def _run_core(self, ranks):
        try:
            if self.backtest_date is not None:
                data.truncate_end = self.backtest_date.strftime('%Y-%m-%d')

            universe = data.get('price:收盤價')
            if self.backtest_date is not None:
                universe = universe[universe.index <= self.backtest_date]

            today = self.backtest_date if self.backtest_date is not None else pd.Timestamp.today().normalize()
            # 延伸到今天：早上跑的時候市場資料只到前一交易日，
            # 不延伸的話最新週日清單與今日進場訊號都會被裁掉
            end_date = max(universe.index.max(), today)
            position, sl_df, tp_df = self._create_df(universe, ranks=ranks, end_date=end_date)

            use_db_sl_tp = self.use_db_sl or self.use_db_tp
            use_touched_exit = (
                not use_db_sl_tp
                and (self.global_sl is not None or self.global_tp is not None)
            )

            # until 至少要涵蓋錨點的第一個 cycle，避免在錨點之前太久執行時
            # cycles 是空的、log 顯示不出「下一個買入日」
            cycles = self._compute_cycles(until=max(end_date, self.cycle_start_date) + pd.Timedelta(days=7))
            first_entry = cycles[0][0] if cycles else None
            # 歷史週期必須在「錨點與今天中較早者」之前整個結束，
            # 確保今天絕不會落在歷史週期內（今日目標只由錨點行程表決定）
            hist_before = today if first_entry is None else min(first_entry, today)
            hist_cycles = self._compute_historical_cycles(position.index, before=hist_before)
            cycle_entries = [entry for entry, _ in hist_cycles] + [entry for entry, _ in cycles]
            cycle_exits = [exit_d for _, exit_d in hist_cycles] + [exit_d for _, exit_d in cycles]

            current_cycle_idx = None
            for i, (entry, exit_d) in enumerate(zip(cycle_entries, cycle_exits)):
                if entry <= today <= exit_d:
                    current_cycle_idx = i
                    break
            if current_cycle_idx is not None:
                current_week = (today - cycle_entries[current_cycle_idx]).days // 7 + 1
                logger.info(
                    f"當前持有第 {current_week}/{self.hold_weeks} 周, "
                    f"買入日={cycle_entries[current_cycle_idx].strftime('%Y-%m-%d')}, "
                    f"賣出日={cycle_exits[current_cycle_idx].strftime('%Y-%m-%d')}"
                )
            else:
                future = [e for e in cycle_entries if e > today]
                if future:
                    logger.info(f"當前無持倉, 下一個買入日: {future[0].strftime('%Y-%m-%d')}")
                else:
                    logger.info("當前無持倉, 無後續週期")

            entry_mask = position.index.isin(cycle_entries)
            exit_mask = position.index.isin(cycle_exits)

            entries = position & entry_mask[:, np.newaxis]

            if use_touched_exit:
                sl_tp_exits = pd.DataFrame(
                    False, index=position.index, columns=position.columns
                )
            else:
                sl_tp_exits = self._build_sl_tp_exits(
                    entries, position, sl_df, tp_df
                )

            normal_exits = pd.DataFrame(
                np.broadcast_to(
                    exit_mask[:, np.newaxis], position.shape
                ).copy(),
                index=position.index,
                columns=position.columns,
            )

            if self.buy_weekday == self.sell_weekday:
                normal_exits = normal_exits & ~entries

            exits = FinlabDataFrame(normal_exits | sl_tp_exits)
            final_position = FinlabDataFrame(entries).hold_until(exits)
            final_position = (
                final_position.shift(-1).ffill().fillna(False).astype(bool)
            )
            final_position = self._apply_cutoff(final_position)

            if today in final_position.index:
                today_held = final_position.loc[today]
                held_ids = today_held[today_held].index.tolist()
                logger.info(f"今日目標持股: {held_ids if held_ids else '無'}")
            else:
                logger.info(f"今日 ({today.strftime('%Y-%m-%d')}) 不在 position 範圍內")

            if not final_position.to_numpy().any():
                # 全空 position 會產生空 creturn 的報告，Portfolio 建構會崩潰
                logger.info("回測窗口內無任何持倉（歷史與錨點週期皆空），不產生報告、跳過下單")
                return None

            sim_kwargs = dict(
                position=final_position,
                fee_ratio=1.425 / 1000,
                tax_ratio=3 / 1000,
                market=TargetWeekdayTWMarket(
                    buy_weekday=self.buy_weekday,
                    backtest_date=self.backtest_date,
                ),
                trade_at_price=self.trade_at_price,
                resample=None,
                upload=False,
                notification_enable=False,
            )

            if use_touched_exit:
                sim_kwargs.update(
                    stop_loss=self.global_sl,
                    take_profit=self.global_tp,
                    touched_exit=True,
                )

            return sim(**sim_kwargs)
        finally:
            data.truncate_end = None

    def _check_recommendation_freshness(self, latest_rec, today):
        """進行中的週期若缺少進場日應使用的週日清單，擋下下單並拋錯（由外層通知）。

        清單晚入庫時每日 sync 會自動補進場，寧可晚買也不要默默用過期清單買。
        """
        cycles = self._compute_cycles(until=today + pd.Timedelta(days=7))
        current = next(((e, x) for e, x in cycles if e <= today <= x), None)
        if current is None:
            return
        entry = current[0]
        expected_sunday = entry - pd.Timedelta(days=(entry.dayofweek - 6) % 7)
        if latest_rec is None:
            raise RuntimeError(
                f"DB 無推薦清單：當前週期（買入日 {entry:%Y-%m-%d}）"
                f"需要 {expected_sunday:%Y-%m-%d} 的清單，不下單"
            )
        rec_date = pd.to_datetime(latest_rec.date)
        aligned = rec_date + pd.Timedelta(days=6 - rec_date.weekday())
        if aligned < expected_sunday:
            raise RuntimeError(
                f"推薦清單過期：最新清單日期 {latest_rec.date}（對齊週日 {aligned:%Y-%m-%d}），"
                f"當前週期買入日 {entry:%Y-%m-%d} 應使用 {expected_sunday:%Y-%m-%d} 的清單，不下單"
            )

    def run_strategy(self):
        weekday_names = '一二三四五六日'
        ranks_str = ','.join(map(str, self.ranks))
        logger.info(
            f"GoldenAI order adapter: frequency={self.task_name}, "
            f"hold_weeks={self.hold_weeks}, ranks=[{ranks_str}], "
            f"買入日=週{weekday_names[self.buy_weekday]}, "
            f"賣出日=週{weekday_names[self.sell_weekday]}, "
            f"cycle_start_date={self.cycle_start_date:%Y-%m-%d}"
        )

        dao = RecommendationDAO(frequency=self.recommendation_frequency)
        latest_rec = dao.get_latest()
        if latest_rec:
            all_stocks = latest_rec.stocks
            selected = [all_stocks[r - 1] for r in self.ranks if r <= len(all_stocks)]
            stocks_info = ', '.join(f"{s.id}({s.name or '?'})" for s in selected)
            logger.info(f"最新推薦清單 ({latest_rec.date}): [{stocks_info}]")
        else:
            logger.warning("DB 無推薦清單")

        today = self.backtest_date if self.backtest_date is not None else pd.Timestamp.today().normalize()
        self._check_recommendation_freshness(latest_rec, today)

        return self._run_core(self.ranks)
