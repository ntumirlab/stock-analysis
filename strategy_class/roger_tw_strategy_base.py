import numpy as np
import pandas as pd
from finlab import data
from finlab.backtest import sim
from finlab.dataframe import FinlabDataFrame
from utils.config_loader import ConfigLoader
from dao.recommendation_dao import RecommendationDAO
from markets.target_weekday_tw_market import TargetWeekdayTWMarket

class RogerTWStrategyBase:
    def __init__(self, task_name, config_path="config.yaml", override_params=None):
        self.task_name = task_name  # 'weekly' or 'monthly'
        self.report = None
        self.config_loader = ConfigLoader(config_path)
        roger_config = self.config_loader.config.get('roger', {}).get(task_name, {})

        # 如果有傳入實驗參數，就用實驗的；否則就讀 config.yaml 中的
        if override_params is None:
            override_params = {}

        self.buy_weekday = override_params.get('buy_weekday', roger_config.get('buy_weekday', 1)) - 1
        self.sell_weekday = override_params.get('sell_weekday', roger_config.get('sell_weekday', 5)) - 1
        self.max_stocks = override_params.get('max_stocks', roger_config.get('max_stocks', 5))
        self.use_db_sl = override_params.get('use_db_sl', roger_config.get('use_db_sl', True))
        self.global_sl = override_params.get('global_sl', roger_config.get('global_sl', None))
        self.use_db_tp = override_params.get('use_db_tp', roger_config.get('use_db_tp', True))
        self.global_tp = override_params.get('global_tp', roger_config.get('global_tp', None))
        self.trade_at_price = override_params.get('trade_at_price', roger_config.get('trade_at_price', 'open'))

        print(f"[{task_name}] 策略參數: 週{'一二三四五'[self.buy_weekday]}買, 週{'一二三四五'[self.sell_weekday]}賣, 上限 {self.max_stocks} 檔")

    def _create_df(self, universe):
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

        records, sl_records, tp_records = [], [], []

        for aligned_date, (_, stock_list) in weekly_batches.items():
            stock_list = sorted(stock_list, key=lambda x: getattr(x, 'priority', float('inf')))
            if self.max_stocks is not None:
                stock_list = stock_list[:self.max_stocks]

            for stock in stock_list:
                stock_id = getattr(stock, 'id', None)
                if not stock_id:
                    continue
                sid = str(stock_id)

                records.append({'date': aligned_date, 'stock_id': sid, 'signal': 1})

                if self.use_db_sl:
                    sl_price = getattr(stock, 'SL', None)
                    if sl_price is not None:
                        sl_records.append({'date': aligned_date, 'stock_id': sid, 'sl_price': sl_price})

                if self.use_db_tp:
                    tp_price = getattr(stock, 'TP', None)
                    if tp_price is not None:
                        tp_records.append({'date': aligned_date, 'stock_id': sid, 'tp_price': tp_price})

        def _pivot(recs, value_col, ref_position):
            df = pd.DataFrame(recs)
            if df.empty:
                return pd.DataFrame(0, index=ref_position.index, columns=ref_position.columns)
            df = df.drop_duplicates(subset=['date', 'stock_id'])
            df = df.pivot(index='date', columns='stock_id', values=value_col).fillna(0)
            return df

        df = pd.DataFrame(records).drop_duplicates(subset=['date', 'stock_id'])
        position = df.pivot(index='date', columns='stock_id', values='signal').fillna(0)

        sl_df = _pivot(sl_records, 'sl_price', position)
        tp_df = _pivot(tp_records, 'tp_price', position)

        # 轉為每日資料並 Forward Fill
        position = position.resample('D').ffill()
        sl_df    = sl_df.resample('D').ffill()
        tp_df    = tp_df.resample('D').ffill()

        # 對齊日期範圍至最新市場日
        latest_market_date = universe.index.max()
        if latest_market_date > position.index.max():
            extended_index = pd.date_range(start=position.index.min(),
                                           end=latest_market_date, freq='D')
            position = position.reindex(extended_index, method='ffill')
            sl_df    = sl_df.reindex(extended_index, method='ffill')
            tp_df    = tp_df.reindex(extended_index, method='ffill')

        # 對齊全市場股票代號
        position = position.reindex(columns=universe.columns, fill_value=0)
        sl_df    = sl_df.reindex(columns=universe.columns, fill_value=0)
        tp_df    = tp_df.reindex(columns=universe.columns, fill_value=0)

        return position.astype(bool), sl_df, tp_df

    def _compute_sl_tp_prices(self, entries, position, sl_df, tp_df):
        """
        計算每股的最終 SL/TP 絕對價格矩陣（NaN = 不觸發）。
        """
        open_ = data.get('price:開盤價').reindex(index=position.index, columns=position.columns)
        entry_price = open_.where(entries).ffill()

        final_sl = pd.DataFrame(float('nan'), index=position.index, columns=position.columns)
        final_tp = pd.DataFrame(float('nan'), index=position.index, columns=position.columns)

        # 停損：DB 絕對價優先；DB 空值才用 global_sl 比例補
        if self.use_db_sl:
            db_sl = sl_df.replace(0, float('nan'))
            final_sl = db_sl.copy()
            if self.global_sl is not None:
                global_sl_price = entry_price * (1 - self.global_sl)
                final_sl = final_sl.where(final_sl.notna(), global_sl_price)
        elif self.global_sl is not None:
            final_sl = entry_price * (1 - self.global_sl)

        # 停利：DB 絕對價優先；DB 空值才用 global_tp 比例補
        if self.use_db_tp:
            db_tp = tp_df.replace(0, float('nan'))
            final_tp = db_tp.copy()
            if self.global_tp is not None:
                global_tp_price = entry_price * (1 + self.global_tp)
                final_tp = final_tp.where(final_tp.notna(), global_tp_price)
        elif self.global_tp is not None:
            final_tp = entry_price * (1 + self.global_tp)

        return final_sl, final_tp

    def _build_sl_tp_exits(self, entries, position, sl_df, tp_df):
        """
        計算個股 SL/TP 出場訊號矩陣。
        回傳值：FinlabDataFrame bool，True 代表當日觸發 SL 或 TP 應出場。
        買入當天不觸發（避免同日進出）。
        """
        has_any_config = (
            self.use_db_sl or self.use_db_tp
            or self.global_sl is not None
            or self.global_tp is not None
        )
        if not has_any_config:
            return pd.DataFrame(False, index=position.index, columns=position.columns)

        low  = data.get('price:最低價').reindex(index=position.index, columns=position.columns)
        high = data.get('price:最高價').reindex(index=position.index, columns=position.columns)

        final_sl, final_tp = self._compute_sl_tp_prices(entries, position, sl_df, tp_df)

        sl_exit = pd.DataFrame(False, index=position.index, columns=position.columns)
        tp_exit = pd.DataFrame(False, index=position.index, columns=position.columns)

        if final_sl.notna().any(axis=None):
            sl_exit = (low < final_sl).fillna(False)
        if final_tp.notna().any(axis=None):
            tp_exit = (high > final_tp).fillna(False)

        # 買入當天不觸發 SL/TP
        return FinlabDataFrame((sl_exit | tp_exit) & ~entries)

    def run_strategy(self):
        """
        流程：
        1. _create_df：建立推薦持倉、SL、TP 矩陣
        2. entries：推薦清單內 AND 買入日 → 進場訊號
        3. exits ：賣出日 OR SL/TP  → 出場訊號
        4. hold_until：進場後持倉直到 exits 觸發
        5. shift(-1)：今日訊號 → 明日執行（FinLab sim 慣例）

        touched_exit 模式（use_db_sl=False, use_db_tp=False, global_sl/tp 有值）：
        SL/TP 完全交給 sim() 處理，hold_until 不介入，避免雙重計算。
        """
        universe = data.get('price:收盤價')
        position, sl_df, tp_df = self._create_df(universe)

        use_db_sl_tp = self.use_db_sl or self.use_db_tp
        use_touched_exit = (
            not use_db_sl_tp
            and (self.global_sl is not None or self.global_tp is not None)
        )

        dow = pd.Series(position.index.dayofweek, index=position.index)

        # entries：推薦清單內 AND 買入日 → 進場訊號
        buy_mask = (dow == self.buy_weekday).to_numpy()
        entries = position & buy_mask[:, np.newaxis]

        # SL/TP 出場：touched_exit 模式下交給 sim()，hold_until 不處理
        if use_touched_exit:
            sl_tp_exits = pd.DataFrame(False, index=position.index, columns=position.columns)
        else:
            sl_tp_exits = self._build_sl_tp_exits(entries, position, sl_df, tp_df)

        # 正常出場：賣出日
        sell_mask = (dow == self.sell_weekday).to_numpy()
        normal_exits = pd.DataFrame(
            np.broadcast_to(sell_mask[:, np.newaxis], position.shape),
            index=position.index,
            columns=position.columns
        )

        # buy == sell 時，進場日不觸發正常出場（避免當天進出）
        if self.buy_weekday == self.sell_weekday:
            normal_exits = normal_exits & ~entries

        exits = FinlabDataFrame(normal_exits | sl_tp_exits)

        # hold_until → shift(-1) → sim
        final_position = FinlabDataFrame(entries).hold_until(exits)
        final_position = final_position.shift(-1).fillna(False).astype(bool)

        if use_touched_exit:
            self.report = sim(
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
            self.report = sim(
                position=final_position,
                fee_ratio=1.425/1000,
                tax_ratio=3/1000,
                market=TargetWeekdayTWMarket(buy_weekday=self.buy_weekday),
                trade_at_price=self.trade_at_price,
                resample=None,
                upload=False,
                notification_enable=False
            )
        return self.report

    def get_report(self):
        return self.report if self.report else "report物件為空，請先運行策略"