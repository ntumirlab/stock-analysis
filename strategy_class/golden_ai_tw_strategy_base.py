import os
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from finlab import data
from finlab.backtest import sim
from finlab.dataframe import FinlabDataFrame
from utils.config_loader import ConfigLoader
from dao.recommendation_dao import RecommendationDAO
from dao.golden_ai_backtest_metrics_dao import GoldenAIBacktestMetricsDAO
from markets.target_weekday_tw_market import TargetWeekdayTWMarket


class MultiReportWrapper:
    def __init__(self, reports_dict):
        self.reports_dict = reports_dict

    def display(self, save_report_path=None, **kwargs):
        base_dir, file_name = os.path.split(save_report_path)
        file_base, ext = os.path.splitext(file_name)
        for name, report in self.reports_dict.items():
            new_path = os.path.join(base_dir, f"{file_base}_{name}{ext}")
            print(f"[{name}] 儲存報告至: {new_path}")
            report.display(save_report_path=new_path, **kwargs)


class GoldenAITWStrategyBase:
    def __init__(self, task_name, config_path="config.yaml", override_params=None):
        self.task_name = task_name  # 'weekly' or 'monthly'
        self.report = None
        self.config_loader = ConfigLoader(config_path)
        golden_ai_config = self.config_loader.config.get('golden_ai', {}).get(task_name, {})

        if override_params is None:
            override_params = {}

        self.buy_weekday = override_params.get('buy_weekday', golden_ai_config.get('buy_weekday', 1)) - 1
        self.sell_weekday = override_params.get('sell_weekday', golden_ai_config.get('sell_weekday', 5)) - 1
        self.rank_start = override_params.get('rank_start', golden_ai_config.get('rank_start', 1))
        self.rank_end = override_params.get('rank_end', golden_ai_config.get('rank_end', 5))
        self.use_db_sl = override_params.get('use_db_sl', golden_ai_config.get('use_db_sl', True))
        self.global_sl = override_params.get('global_sl', golden_ai_config.get('global_sl', None))
        self.use_db_tp = override_params.get('use_db_tp', golden_ai_config.get('use_db_tp', True))
        self.global_tp = override_params.get('global_tp', golden_ai_config.get('global_tp', None))
        self.trade_at_price = override_params.get('trade_at_price', golden_ai_config.get('trade_at_price', 'open'))
        self.lookback_months = override_params.get('lookback_months', golden_ai_config.get('lookback_months', None))

        backtest_date_raw = override_params.get('backtest_date', None)
        self.backtest_date = pd.Timestamp(backtest_date_raw).normalize() if backtest_date_raw else None

        print(f"[{task_name}] 策略參數: 週{'一二三四五'[self.buy_weekday]}買, 週{'一二三四五'[self.sell_weekday]}賣, Rank {self.rank_start}~{self.rank_end}")

    def _create_df(self, universe, ranks):
        """
        讀取推薦 DAO 並轉換為 Finlab 可用的 Position DataFrame
        支援 stocks 為物件列表，從 stock.id 取代號

        日期對齊規則：
        - 周中（週一～週六）產出的清單 → 對齊到「下一個週日」（代表下週的推薦）
        - 週日當天產出的清單 → 留在當天（代表本週的推薦）
        """

        dao = RecommendationDAO(frequency=self.task_name)
        recommendation_records = dao.load()

        weekly_batches = {}

        for record in recommendation_records:
            date = record.date
            stocks = record.stocks
            if not date or not stocks:
                continue

            dt = pd.to_datetime(date)

            days_to_sunday = 6 - dt.weekday()
            aligned_date = dt + pd.Timedelta(days=days_to_sunday)
            if days_to_sunday > 0:
                print(f"原始日期: {dt.date()} (週{'一二三四五六日'[dt.weekday()]})，"
                      f"對齊後日期: {aligned_date.date()} (週{'一二三四五六日'[aligned_date.weekday()]})")

            if aligned_date in weekly_batches:
                existing_original_date, _ = weekly_batches[aligned_date]
                if dt > existing_original_date:
                    weekly_batches[aligned_date] = (dt, stocks)
                    print(f"更新對齊日期 {aligned_date.date()} 的推薦，"
                          f"使用較新的原始日期 {dt.date()}")
            else:
                weekly_batches[aligned_date] = (dt, stocks)

        # 先收集所有週出現過的 stock_id，用來對缺席的股票補 0
        all_stock_ids = set()
        for _, (_, stock_list) in weekly_batches.items():
            for stock in stock_list:
                sid = str(getattr(stock, 'id', ''))
                if sid:
                    all_stock_ids.add(sid)

        records, sl_records, tp_records = [], [], []

        for aligned_date, (_, stock_list) in weekly_batches.items():
            stock_list = sorted(stock_list, key=lambda x: getattr(x, 'priority', float('inf')))
            selected = [stock_list[r - 1] for r in ranks if r <= len(stock_list)]
            selected_ids = set()

            for stock in selected:
                stock_id = getattr(stock, 'id', None)
                if not stock_id:
                    continue
                sid = str(stock_id)
                selected_ids.add(sid)

                records.append({'date': aligned_date, 'stock_id': sid, 'signal': 1})

                if self.use_db_sl:
                    sl_price = getattr(stock, 'SL', None)
                    if sl_price is not None:
                        sl_records.append({'date': aligned_date, 'stock_id': sid, 'sl_price': sl_price})

                if self.use_db_tp:
                    tp_price = getattr(stock, 'TP', None)
                    if tp_price is not None:
                        tp_records.append({'date': aligned_date, 'stock_id': sid, 'tp_price': tp_price})

            # 本週未選到的股票明確補 0，防止 ffill 跨週帶入上週持倉
            for sid in all_stock_ids - selected_ids:
                records.append({'date': aligned_date, 'stock_id': sid, 'signal': 0})

        def _pivot(recs, value_col, fallback_index):
            df = pd.DataFrame(recs)
            if df.empty:
                return pd.DataFrame(index=fallback_index, dtype=float)
            df = df.drop_duplicates(subset=['date', 'stock_id'])
            return df.pivot(index='date', columns='stock_id', values=value_col).fillna(0)

        df = pd.DataFrame(records).drop_duplicates(subset=['date', 'stock_id'])
        position = df.pivot(index='date', columns='stock_id', values='signal').fillna(0)

        sl_df = _pivot(sl_records, 'sl_price', position.index)
        tp_df = _pivot(tp_records, 'tp_price', position.index)

        position = position.resample('D').ffill()
        sl_df    = sl_df.resample('D').ffill()
        tp_df    = tp_df.resample('D').ffill()

        latest_market_date = universe.index.max()
        if latest_market_date > position.index.max():
            extended_index = pd.date_range(start=position.index.min(),
                                           end=latest_market_date, freq='D')
            position = position.reindex(extended_index, method='ffill')
            sl_df    = sl_df.reindex(extended_index, method='ffill')
            tp_df    = tp_df.reindex(extended_index, method='ffill')
        elif latest_market_date < position.index.max():
            position = position[position.index <= latest_market_date]
            sl_df    = sl_df[sl_df.index <= latest_market_date]
            tp_df    = tp_df[tp_df.index <= latest_market_date]

        position = position.reindex(columns=universe.columns, fill_value=0)
        sl_df    = sl_df.reindex(columns=universe.columns, fill_value=0)
        tp_df    = tp_df.reindex(columns=universe.columns, fill_value=0)

        return position.astype(bool), sl_df, tp_df

    def _apply_cutoff(self, final_position):
        if self.lookback_months is None:
            return final_position
        from dateutil.relativedelta import relativedelta
        ref = self.backtest_date if self.backtest_date is not None else pd.Timestamp.today().normalize()
        cutoff = ref - relativedelta(months=self.lookback_months)
        return final_position[final_position.index >= cutoff]

    def _build_sl_tp_exits(self, entries, position, sl_df, tp_df,
                           raw_low=None, raw_high=None):
        """
        計算個股 SL/TP 出場訊號矩陣
        回傳值：FinlabDataFrame bool，True 代表當日觸發 SL 或 TP 應出場
        買入當天不觸發（避免同日進出）

        價格基準說明：
        - DB 絕對價（分析師設定的實際市價）：
            使用非還原價 price:最低價 / 最高價，與 DB 設定單位一致
        - global 比例 SL/TP （與 sim() 內部邏輯一致）：
            使用還原價 etl:adj_open 計算進場成本，etl:adj_low / adj_high 判斷觸發，
            確保除權息後的報酬計算與 sim() 採用相同基準
            若同一股票已有 DB 絕對價，該股不再套用 global 比例（DB 優先）

        raw_low / raw_high：
            可選，供月策略在 loop 外預先載好傳入，避免每個 offset 重複 reindex
            未傳入時（週策略預設行為）在內部載入
        """
        has_any_config = (
            self.use_db_sl or self.use_db_tp
            or self.global_sl is not None
            or self.global_tp is not None
        )
        if not has_any_config:
            return pd.DataFrame(False, index=position.index, columns=position.columns)

        sl_exit = pd.DataFrame(False, index=position.index, columns=position.columns)
        tp_exit = pd.DataFrame(False, index=position.index, columns=position.columns)

        if self.use_db_sl or self.use_db_tp:
            if raw_low is None:
                raw_low = data.get('price:最低價').reindex(index=position.index, columns=position.columns)
            if raw_high is None:
                raw_high = data.get('price:最高價').reindex(index=position.index, columns=position.columns)

            if self.use_db_sl:
                db_sl = sl_df.replace(0, float('nan'))
                if db_sl.notna().any(axis=None):
                    sl_exit = sl_exit | (raw_low < db_sl).fillna(False)

            if self.use_db_tp:
                db_tp = tp_df.replace(0, float('nan'))
                if db_tp.notna().any(axis=None):
                    tp_exit = tp_exit | (raw_high > db_tp).fillna(False)

        if self.global_sl is not None or self.global_tp is not None:
            adj_open = data.get('etl:adj_open').reindex(index=position.index, columns=position.columns)
            adj_entry_price = adj_open.where(entries).ffill()

            if self.global_sl is not None:
                db_sl_mask = (
                    sl_df.replace(0, float('nan')).notna()
                    if self.use_db_sl
                    else pd.DataFrame(False, index=position.index, columns=position.columns)
                )
                adj_low = data.get('etl:adj_low').reindex(index=position.index, columns=position.columns)
                global_sl_price = adj_entry_price * (1 - self.global_sl)
                global_sl_exit = (adj_low < global_sl_price).fillna(False) & ~db_sl_mask
                sl_exit = sl_exit | global_sl_exit

            if self.global_tp is not None:
                db_tp_mask = (
                    tp_df.replace(0, float('nan')).notna()
                    if self.use_db_tp
                    else pd.DataFrame(False, index=position.index, columns=position.columns)
                )
                adj_high = data.get('etl:adj_high').reindex(index=position.index, columns=position.columns)
                global_tp_price = adj_entry_price * (1 + self.global_tp)
                global_tp_exit = (adj_high > global_tp_price).fillna(False) & ~db_tp_mask
                tp_exit = tp_exit | global_tp_exit

        return FinlabDataFrame((sl_exit | tp_exit) & ~entries)

    def _run_core(self, ranks):
        """週策略核心邏輯，供 run_strategy() ranks 迴圈呼叫"""
        universe = data.get('price:收盤價')
        if self.backtest_date is not None:
            universe = universe[universe.index <= self.backtest_date]
        position, sl_df, tp_df = self._create_df(universe, ranks=ranks)

        use_db_sl_tp = self.use_db_sl or self.use_db_tp
        use_touched_exit = (
            not use_db_sl_tp
            and (self.global_sl is not None or self.global_tp is not None)
        )

        dow = pd.Series(position.index.dayofweek, index=position.index)
        buy_mask = (dow == self.buy_weekday).to_numpy()
        entries = position & buy_mask[:, np.newaxis]

        if use_touched_exit:
            sl_tp_exits = pd.DataFrame(False, index=position.index, columns=position.columns)
        else:
            sl_tp_exits = self._build_sl_tp_exits(entries, position, sl_df, tp_df)

        sell_mask = (dow == self.sell_weekday).to_numpy()
        normal_exits = pd.DataFrame(
            np.broadcast_to(sell_mask[:, np.newaxis], position.shape).copy(),
            index=position.index,
            columns=position.columns
        )

        if self.buy_weekday == self.sell_weekday:
            normal_exits = normal_exits & ~entries

        exits = FinlabDataFrame(normal_exits | sl_tp_exits)
        final_position = FinlabDataFrame(entries).hold_until(exits)
        final_position = final_position.shift(-1).fillna(False).astype(bool)
        final_position = self._apply_cutoff(final_position)

        if use_touched_exit:
            return sim(
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
            return sim(
                position=final_position,
                fee_ratio=1.425/1000,
                tax_ratio=3/1000,
                market=TargetWeekdayTWMarket(buy_weekday=self.buy_weekday),
                trade_at_price=self.trade_at_price,
                resample=None,
                upload=False,
                notification_enable=False
            )

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
        print(f"開始執行 {total} 組 Ranks 回測（Rank {self.rank_start}~{self.rank_end}）...")

        for i, ranks in enumerate(all_subsets, 1):
            ranks_str = ','.join(map(str, ranks))
            if dao.exists_for_date(date_str, self.task_name, ranks_str):
                print(f"[{i}/{total}] Ranks[{ranks_str}] 已存在，跳過")
                continue
            print(f"[{i}/{total}] 回測 Ranks[{ranks_str}]...")
            report = self._run_core(ranks=ranks)
            dao.save(timestamp=timestamp, strategy=self.task_name, week=None, ranks=ranks_str, report=report)
            if report_dir is not None:
                save_path = os.path.join(report_dir, f"{date_str}_{time_str}_Ranks[{ranks_str}].html")
                report.display(save_report_path=save_path)

        print("全部完成。")

    def get_report(self):
        return "GoldenAI 策略報告已存至 assets/ 及 DB"
