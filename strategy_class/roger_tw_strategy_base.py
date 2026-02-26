import pandas as pd
from finlab import data
from finlab.backtest import sim
from utils.config_loader import ConfigLoader
from dao.recommendation_dao import RecommendationDAO


class RogerTWStrategyBase:
    def __init__(self, task_name, config_path="config.yaml"):
        self.task_name = task_name  # 'weekly' or 'monthly'
        self.report = None
        self.config_loader = ConfigLoader(config_path)
        roger_config = self.config_loader.config.get('roger', {}).get(task_name, {})

        self.max_stocks = roger_config.get('max_stocks', 5)
        self.buy_weekday = roger_config.get('buy_weekday', 1) - 1
        self.sell_weekday = roger_config.get('sell_weekday', 5) - 1

        print(
            f"[{task_name}] 策略參數: 週{'一二三四五'[self.buy_weekday]}買, 週{'一二三四五'[self.sell_weekday]}賣, 上限 {self.max_stocks} 檔"
        )

    def _create_position_df(self, universe):
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
                print(
                    f"原始日期: {dt.date()} (週{'一二三四五六日'[dt.weekday()]})，對齊後日期: {aligned_date.date()} (週{'一二三四五六日'[aligned_date.weekday()]})"
                )

            if aligned_date in weekly_batches:
                existing_original_date, _ = weekly_batches[aligned_date]
                if dt > existing_original_date:
                    weekly_batches[aligned_date] = (dt, stocks)
                    print(
                        f"更新對齊日期 {aligned_date.date()} (週{'一二三四五六日'[aligned_date.weekday()]}) 的推薦，使用較新的原始日期 {dt.date()} (週{'一二三四五六日'[dt.weekday()]})"
                    )
            else:
                weekly_batches[aligned_date] = (dt, stocks)

        records = []
        for aligned_date, (original_date, stock_list) in weekly_batches.items():
            stock_list = sorted(stock_list, key=lambda x: getattr(x, 'priority', float('inf')))
            if self.max_stocks is not None:
                stock_list = stock_list[: self.max_stocks]
            for stock in stock_list:
                stock_id = getattr(stock, 'id', None)
                if not stock_id:
                    continue
                records.append({'date': aligned_date, 'stock_id': str(stock_id), 'signal': 1})

        if not records:
            return None

        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=['date', 'stock_id'])

        position = df.pivot(index='date', columns='stock_id', values='signal')
        position = position.fillna(0)

        # 轉為每日資料並 Forward Fill
        position = position.resample('D').ffill()

        # 對齊日期範圍
        latest_market_date = universe.index.max()
        if latest_market_date > position.index.max():
            extended_index = pd.date_range(start=position.index.min(), end=latest_market_date, freq='D')
            position = position.reindex(extended_index, method='ffill')

        # 對齊全市場股票代號
        position = position.reindex(columns=universe.columns, fill_value=0)

        return position.astype(bool)

    def _apply_trading_window(self, position):
        """
        根據買賣星期幾設定交易視窗
        """
        dow = position.index.dayofweek
        buy = self.buy_weekday
        sell = self.sell_weekday

        if buy == sell:
            return position
        elif buy < sell:
            mask = (dow >= buy) & (dow < sell)
        else:
            mask = (dow >= buy) | (dow < sell)

        position = position.loc[mask].reindex(position.index, fill_value=False)
        return position

    def run_strategy(self):
        universe = data.get('price:收盤價')
        position = self._create_position_df(universe)

        if self.buy_weekday != self.sell_weekday:
            position = self._apply_trading_window(position)

        # 由於此策略在買賣日「前一天」即決定隔天是否買賣，因此將 position 向前移動一天
        position = position.shift(-1)

        self.report = sim(position=position, resample=None, fee_ratio=1.425 / 1000, tax_ratio=3 / 1000, upload=False)

        return self.report

    def get_report(self):
        return self.report if self.report else "report物件為空，請先運行策略"
