import pandas as pd
from finlab import data
from finlab.backtest import sim
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
                print(f"原始日期: {dt.date()} (週{'一二三四五六日'[dt.weekday()]})，對齊後日期: {aligned_date.date()} (週{'一二三四五六日'[aligned_date.weekday()]})")

            if aligned_date in weekly_batches:
                existing_original_date, _ = weekly_batches[aligned_date]
                if dt > existing_original_date:
                    weekly_batches[aligned_date] = (dt, stocks)
                    print(f"更新對齊日期 {aligned_date.date()} (週{'一二三四五六日'[aligned_date.weekday()]}) 的推薦，使用較新的原始日期 {dt.date()} (週{'一二三四五六日'[dt.weekday()]})")
            else:
                weekly_batches[aligned_date] = (dt, stocks)

        # 產生持倉 DataFrame
        records = []
        for aligned_date, (original_date, stock_list) in weekly_batches.items():
            stock_list = sorted(stock_list, key=lambda x: getattr(x, 'priority', float('inf')))
            if self.max_stocks is not None:
                stock_list = stock_list[:self.max_stocks]
            for stock in stock_list:
                stock_id = getattr(stock, 'id', None)
                if not stock_id:
                    continue
                records.append({
                    'date': aligned_date,
                    'stock_id': str(stock_id),
                    'signal': 1
                })

        # 產生停損 DataFrame
        sl_records = []
        if self.use_db_sl:
            for aligned_date, (original_date, stock_list) in weekly_batches.items():
                stock_list = sorted(stock_list, key=lambda x: getattr(x, 'priority', float('inf')))
                if self.max_stocks is not None:
                    stock_list = stock_list[:self.max_stocks]
                for stock in stock_list:
                    stock_id = getattr(stock, 'id', None)
                    sl_price = getattr(stock, 'SL', None)
                    if not stock_id or sl_price is None:
                        continue
                    sl_records.append({
                        'date': aligned_date,
                        'stock_id': str(stock_id),
                        'sl_price': sl_price
                    })

        # 產生停利 DataFrame
        tp_records = []
        if self.use_db_tp:
            for aligned_date, (original_date, stock_list) in weekly_batches.items():
                stock_list = sorted(stock_list, key=lambda x: getattr(x, 'priority', float('inf')))
                if self.max_stocks is not None:
                    stock_list = stock_list[:self.max_stocks]
                for stock in stock_list:
                    stock_id = getattr(stock, 'id', None)
                    tp_price = getattr(stock, 'TP', None)
                    if not stock_id or tp_price is None:
                        continue
                    tp_records.append({
                        'date': aligned_date,
                        'stock_id': str(stock_id),
                        'tp_price': tp_price
                    })

        # 將 records 轉換為 DataFrame 並 pivot 成 Position DataFrame
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=['date', 'stock_id'])
        position = df.pivot(index='date', columns='stock_id', values='signal')
        position = position.fillna(0)

        sl_df = pd.DataFrame(sl_records)
        if sl_df.empty:
            sl_df = pd.DataFrame(index=position.index, columns=position.columns).fillna(0)
        else:
            sl_df = sl_df.drop_duplicates(subset=['date', 'stock_id'])
            sl_df = sl_df.pivot(index='date', columns='stock_id', values='sl_price')
            sl_df = sl_df.fillna(0)

        tp_df = pd.DataFrame(tp_records)
        if tp_df.empty:
            tp_df = pd.DataFrame(index=position.index, columns=position.columns).fillna(0)
        else:
            tp_df = tp_df.drop_duplicates(subset=['date', 'stock_id'])
            tp_df = tp_df.pivot(index='date', columns='stock_id', values='tp_price')
            tp_df = tp_df.fillna(0)
        
        # 轉為每日資料並 Forward Fill
        position = position.resample('D').ffill()
        sl_df = sl_df.resample('D').ffill()
        tp_df = tp_df.resample('D').ffill()

        # 對齊日期範圍
        latest_market_date = universe.index.max()
        if latest_market_date > position.index.max():
            extended_index = pd.date_range(start=position.index.min(), end=latest_market_date, freq='D')
            position = position.reindex(extended_index, method='ffill')
            sl_df = sl_df.reindex(extended_index, method='ffill')
            tp_df = tp_df.reindex(extended_index, method='ffill')
        
        # 對齊全市場股票代號
        position = position.reindex(columns=universe.columns, fill_value=0)
        sl_df = sl_df.reindex(columns=universe.columns, fill_value=0)
        tp_df = tp_df.reindex(columns=universe.columns, fill_value=0)
        
        return position.astype(bool), sl_df, tp_df

    def _apply_trading_window(self, position):
        """
        此為虛擬方法，子類別需要根據買賣週期設定交易視窗
        """
        return position

    def _apply_sl_tp(self, position, sl_df, tp_df, universe):
        open_ = data.get('price:開盤價').reindex(index=position.index, columns=position.columns)
        low = data.get('price:最低價').reindex(index=position.index, columns=position.columns)
        high = data.get('price:最高價').reindex(index=position.index, columns=position.columns)
        new_entry = position & ~position.shift(1, fill_value=False)
        entry_price = open_.where(new_entry).ffill()

        # 建立最終停損停利絕對價格（NaN = 不觸發）
        final_sl = pd.DataFrame(float('nan'), index=position.index, columns=position.columns)
        final_tp = pd.DataFrame(float('nan'), index=position.index, columns=position.columns)

        # 停損：DB 優先，DB 空值才用 global_sl 比例補
        if self.use_db_sl:
            db_sl = sl_df.replace(0, float('nan'))
            final_sl = db_sl.copy()
            if self.global_sl is not None:
                global_sl_price = entry_price * (1 - self.global_sl)
                final_sl = final_sl.where(final_sl.notna(), global_sl_price)
        elif self.global_sl is not None:
            final_sl = entry_price * (1 - self.global_sl)

        # 停利：DB 優先，DB 空值才用 global_tp 比例補
        if self.use_db_tp:
            db_tp = tp_df.replace(0, float('nan'))
            final_tp = db_tp.copy()
            if self.global_tp is not None:
                global_tp_price = entry_price * (1 + self.global_tp)
                final_tp = final_tp.where(final_tp.notna(), global_tp_price)
        elif self.global_tp is not None:
            final_tp = entry_price * (1 + self.global_tp)

        has_sl = final_sl.notna().any(axis=None)
        has_tp = final_tp.notna().any(axis=None)

        if not has_sl and not has_tp:
            return position

        triggered = pd.DataFrame(False, index=position.index, columns=position.columns)
        if has_sl:
            triggered |= (low < final_sl) & position
        if has_tp:
            triggered |= (high > final_tp) & position

        result = pd.DataFrame(False, index=position.index, columns=position.columns)
        running = pd.Series(False, index=position.columns)
        for date in position.index:
            running[new_entry.loc[date]] = False
            running |= triggered.loc[date]
            result.loc[date] = running.values

        return position & ~result


    def run_strategy(self):
        universe = data.get('price:收盤價')
        position, sl_df, tp_df = self._create_df(universe)

        use_db_sl_tp = self.use_db_sl or self.use_db_tp
        if use_db_sl_tp:
            position = self._apply_sl_tp(position, sl_df, tp_df, universe)  
        
        if self.buy_weekday != self.sell_weekday:
            position = self._apply_trading_window(position)

        position = position.shift(-1).fillna(False).astype(bool)

        if use_db_sl_tp:
            self.report = sim(
                position=position,
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
                position=position,
                fee_ratio=1.425/1000,
                tax_ratio=3/1000,
                stop_loss=self.global_sl,
                take_profit=self.global_tp,
                touched_exit=True,
                market=TargetWeekdayTWMarket(buy_weekday=self.buy_weekday),
                trade_at_price=self.trade_at_price,
                resample=None,
                upload=False,
                notification_enable=False
            )

        return self.report

    def get_report(self):
        return self.report if self.report else "report物件為空，請先運行策略"