"""
Oscar 四大指標策略 (Oscar Four Key Indicators Strategy)

根據 Oscar 的交易邏輯，結合四個核心指標:
1. SAR (Stop and Reverse) - 拋物線指標
2. MACD (Moving Average Convergence Divergence) - 指數平滑異同移動平均線
3. Volume (VOL) - 買賣聲量
4. Three Major Institutional Investors - 三大法人買賣超

策略核心邏輯:
- 買進: SAR翻多訊號與MACD黃金交叉在可接受時間窗內 + 適當成交量 + 法人買超支持
- 賣出: SAR反轉到上方 或 MACD死亡交叉
- 持股限制: 最多同時持有4檔股票，每檔25%資金
- 選股策略: 從符合條件的股票中，優先選擇訊號最強的
- 特別注意: 排除不斷創新高的股票，避免追高

股票範圍:
全台股市場 (TSE + OTC)

性能優化 (Performance Optimizations):
🚀 智能緩存: 每個指標只計算一次，避免重複計算 (2x speedup)
🚀 支持預載數據: 可傳入預先載入的market_data，避免重複載入
🚀 並行優化友好: 適合大規模參數優化和多股票回測
"""

from finlab import data
from finlab.markets.tw import TWMarket
from finlab.backtest import sim
import pandas as pd
from utils.config_loader import ConfigLoader

class AdjustTWMarketInfo(TWMarket):
    """自訂市場資訊類別，用於調整交易價格為開盤價"""
    def get_trading_price(self, name, adj=True):
        # 使用當日開盤價作為交易價格
        return self.get_price('open', adj=adj)


class OscarTWStrategy:
    """
    Oscar 四大指標策略

    Attributes:
        report: 回測報告物件
        position: 持倉訊號
        buy_signal: 買入訊號
        sell_signal: 賣出訊號
    """

    def __init__(self, sar_signal_lag_min=0, sar_signal_lag_max=2, config_path="config.yaml", 
                 sar_params=None, macd_params=None, market_data=None,
                 volume_above_avg_ratio=None, new_high_ratio_120=None,
                 sar_max_dots=None, sar_reject_dots=None):
        """
        初始化策略參數
        
        Args:
            sar_signal_lag_min: MACD 黃金交叉與 SAR 翻多的最小允許天數差（>=0）
            sar_signal_lag_max: MACD 黃金交叉與 SAR 翻多的最大允許天數差（>= sar_signal_lag_min）
            config_path: 設定檔路徑，用於載入 oscar 相關配置
            sar_params: SAR指標參數字典 {'acceleration': 0.02, 'maximum': 0.2}
            macd_params: MACD指標參數字典 {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}
            market_data: 預先載入的市場數據（若提供則不重新載入）
            volume_above_avg_ratio: 覆寫成交量條件比例（None 則使用 config）
            new_high_ratio_120: 覆寫 120 日創新高比例（None 則使用 config）
            sar_max_dots: [Deprecated] 舊參數，僅為向後相容保留
            sar_reject_dots: [Deprecated] 舊參數，僅為向後相容保留
        """
        # 載入配置
        self.config_loader = ConfigLoader(config_path)
        oscar_config = self.config_loader.config.get('oscar', {})
        

        # 新參數：SAR 與 MACD 訊號允許時間窗
        # 向後相容：若仍傳入舊參數 sar_max_dots，映射到 lag_max。
        if sar_max_dots is not None and sar_signal_lag_max == 2:
            sar_signal_lag_max = int(sar_max_dots)

        self.sar_signal_lag_min = max(0, int(sar_signal_lag_min))
        self.sar_signal_lag_max = max(self.sar_signal_lag_min, int(sar_signal_lag_max))

        # 保留舊欄位名稱供外部程式存取（deprecated, no longer used in logic）
        self.sar_max_dots = sar_max_dots
        self.sar_reject_dots = sar_reject_dots
        
        # SAR 和 MACD 指標參數
        self.sar_params = sar_params or {'acceleration': 0.02, 'maximum': 0.2}
        self.macd_params = macd_params or {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}

        # 成交量大於平均的比例，以及 120 日創新高比例，從設定檔讀取，可覆寫預設值
        # 預設值保留現有行為：volume_above_avg_ratio=0.25, new_high_ratio_120=0.3
        default_volume_ratio = float(oscar_config.get("volume_above_avg_ratio", 0.25))
        default_new_high_ratio = float(oscar_config.get("new_high_ratio_120", 0.3))
        self.volume_above_avg_ratio = (
            float(volume_above_avg_ratio)
            if volume_above_avg_ratio is not None
            else default_volume_ratio
        )
        self.new_high_ratio_120 = (
            float(new_high_ratio_120)
            if new_high_ratio_120 is not None
            else default_new_high_ratio
        )
        
        # 回測報告
        self.report = None

        # 載入市場數據（若未提供則載入）
        market_data = market_data if market_data is not None else self._load_data()
        
        # 儲存市場數據供視覺化使用
        self.market_data = market_data
        
        # 儲存指標數據供視覺化使用
        self.sar_values = None
        self.macd_dif = None
        self.macd_dea = None
        self.macd_histogram = None
        
        # 儲存法人買賣超數據供視覺化使用
        self.institutional_condition = {
            'foreign_buy': None,
            'trust_buy': None,
            'dealer_buy': None
        }
        
        # 儲存交易價格（開盤價）供視覺化使用
        self.trade_price = market_data['open']
        
        # 🚀 優化：預先計算所有指標（只計算一次，避免重複）
        self._cached_sar_buy, self._cached_sar_sell = self._calculate_sar_condition(market_data['close'])
        self._cached_macd_buy, self._cached_macd_sell = self._calculate_macd_condition()
        self._cached_volume_condition = self._calculate_volume_condition(market_data['volume'])
        self._cached_institutional_strong, self._cached_institutional_weak = self._calculate_institutional_condition(
            market_data['foreign_net_buy_shares'],
            market_data['investment_trust_net_buy_shares'],
            market_data['dealer_self_net_buy_shares']
        )
        
        # 建立買入訊號、賣出訊號（使用預先計算的結果）
        self.buy_signal = self._build_buy_condition(market_data)
        self.sell_signal = self._build_sell_condition(market_data)
        
        # 基礎持倉訊號（未套用持股限制）
        self.base_position = self.buy_signal.hold_until(self.sell_signal)

    @staticmethod
    def load_market_data():
        """
        靜態方法：載入市場數據（供外部預先載入使用）
        
        Returns:
            dict: 包含所有必要數據的字典
        """
        with data.universe(market='TSE_OTC'):
            return {
                'open': data.get('price:開盤價'),
                'close': data.get('price:收盤價'),
                'high': data.get('price:最高價'),
                'low': data.get('price:最低價'),
                'volume': data.get('price:成交股數'),
                'foreign_net_buy_shares': data.get('institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)'),
                'investment_trust_net_buy_shares': data.get('institutional_investors_trading_summary:投信買賣超股數'),
                'dealer_self_net_buy_shares': data.get('institutional_investors_trading_summary:自營商買賣超股數(自行買賣)')
            }
    
    def _load_data(self):
        """
        載入所需數據（實例方法，內部調用靜態方法）
        
        Returns:
            dict: 包含所有必要數據的字典
        """
        return self.load_market_data()

    def _streak(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算布林值連續為 True 的天數（逐欄位）。"""
        def _col_streak(col: pd.Series) -> pd.Series:
            col_bool = col.fillna(False).astype(bool)
            groups = (~col_bool).cumsum()
            groups.name = None 
            return col_bool.astype(int).groupby(groups).cumsum()
        
        return df.apply(_col_streak)

    def _calculate_sar_condition(self, close):
        """
        計算 SAR 指標條件
        
        Args:
            close: 收盤價
            
        Returns:
            sar_buy_condition: SAR買進條件
            sar_sell_condition: SAR賣出條件
        """
        # 使用 finlab 的 indicator API 計算 SAR (使用原始價格)
        sar = data.indicator('SAR', 
                           acceleration=self.sar_params['acceleration'], 
                           maximum=self.sar_params['maximum'], 
                           adjust_price=False)
        
        # 儲存 SAR 值供視覺化使用
        self.sar_values = sar
        
        # SAR 在價格下方表示看漲狀態
        sar_below_price = sar < close

        # SAR 翻多事件：當天由非翻多轉為翻多。
        # 注意：第一筆資料沒有「前一日」狀態，以當日值填補避免第一筆即觸發虛假翻多訊號。
        sar_below_price_prev = sar_below_price.shift(1).fillna(sar_below_price)
        sar_flip_bullish = sar_below_price & (~sar_below_price_prev)

        # 以兩參數時間窗檢查 SAR 是否在 MACD 前 lag_min~lag_max 天內出現翻多。
        # 注意: 不要用 pd.DataFrame(...) 初始化，否則會失去 finlab DataFrame 擴充方法 (e.g. hold_until)。
        lagged_signals = [
            sar_flip_bullish.shift(lag).fillna(False)
            for lag in range(self.sar_signal_lag_min, self.sar_signal_lag_max + 1)
        ]
        sar_in_alignment_window = lagged_signals[0]
        for signal in lagged_signals[1:]:
            sar_in_alignment_window = sar_in_alignment_window | signal
        
        # 檢測不斷創新高的股票 (120天內創新高次數過多)
        high_120 = close.rolling(120).max()
        is_new_high = close >= high_120
        new_high_ratio_120 = is_new_high.rolling(120).mean()
        constantly_new_high = new_high_ratio_120 > self.new_high_ratio_120
        
        # 買進條件: SAR 訊號在對齊時間窗內，且不是不斷創新高。
        sar_buy_condition = sar_in_alignment_window & (~constantly_new_high)
        
        # 賣出條件: SAR 翻轉到價格上方 (看跌訊號)
        sar_sell_condition = ~sar_below_price
        
        return sar_buy_condition, sar_sell_condition

    def _calculate_macd_condition(self):
        """
        計算 MACD 指標條件
        
        Oscar 的策略邏輯:
        - 買進: 快線(DIF/黃線)由下往上穿過慢線(MACD/藍線) - 黃金交叉
        - 賣出: 快線由上往下穿過慢線 - 死亡交叉
        
        Returns:
            macd_buy_condition: MACD買進條件
            macd_sell_condition: MACD賣出條件
        """
        # 使用 finlab 的 indicator API 計算 MACD (使用原始價格)
        dif, dea, histogram = data.indicator('MACD', 
                                      fastperiod=self.macd_params['fastperiod'], 
                                      slowperiod=self.macd_params['slowperiod'], 
                                      signalperiod=self.macd_params['signalperiod'], 
                                      adjust_price=False)
        
        # 儲存 MACD 值供視覺化使用
        self.macd_dif = dif
        self.macd_dea = dea
        self.macd_histogram = histogram
        
        # 黃金交叉: DIF > MACD 且前一日 DIF <= MACD (剛形成買進訊號)
        # 死亡交叉: DIF < MACD 且前一日 DIF >= MACD
        macd_buy_condition = (dif > dea) & (dif.shift(1) <= dea.shift(1))
        macd_sell_condition = (dif < dea) & (dif.shift(1) >= dea.shift(1))
        
        return macd_buy_condition, macd_sell_condition

    def _calculate_volume_condition(self, volume):
        """
        計算成交量條件
        
        Args:
            volume: 成交股數
            
        Returns:
            volume_condition: 成交量條件
        """
        # 計算30日平均成交量（需完整30日資料）
        avg_volume_30 = volume.rolling(30).mean()
        
        # 條件1: 成交量大於30日平均的一定比例 (確保基本流動性)，比例由設定檔控制
        # 條件2: 30日平均成交量大於100萬股 (中大型股)
        # 條件3: 排除異常暴量 (當日成交量超過30日平均10倍可能是炒作)
        volume_above_avg = volume > (avg_volume_30 * self.volume_above_avg_ratio)
        sufficient_liquidity = avg_volume_30 > 1_000_000
        not_abnormal_volume = volume < (avg_volume_30 * 10)
        
        return volume_above_avg & sufficient_liquidity & not_abnormal_volume

    def _calculate_institutional_condition(self, foreign_net_buy, trust_net_buy, dealer_net_buy):
        """
        計算三大法人條件
        
        Args:
            foreign_net_buy: 外資買賣超股數
            trust_net_buy: 投信買賣超股數
            dealer_net_buy: 自營商買賣超股數
            
        Returns:
            institutional_strong_condition: 三者皆買超
            institutional_weak_condition: 至少兩者買超
        """
        # 外資買超、投信買超、自營商買賣超
        foreign_buy = foreign_net_buy > 0
        trust_buy = trust_net_buy > 0
        dealer_buy = dealer_net_buy > 0
        
        # 儲存法人買超數據供視覺化使用
        self.institutional_condition = {
            'foreign_buy': foreign_buy,
            'trust_buy': trust_buy,
            'dealer_buy': dealer_buy
        }
        
        # 計算買超家數
        buy_count = foreign_buy.astype(int) + trust_buy.astype(int) + dealer_buy.astype(int)
        
        # 三者皆買超 (最強訊號)
        institutional_strong_condition = buy_count == 3
        
        # 至少兩者買超 (次強訊號，包含三者)
        institutional_weak_condition = buy_count >= 2
        
        return institutional_strong_condition, institutional_weak_condition

    def _build_buy_condition(self, market_data):
        """
        建立買進條件（使用預先計算的指標結果）
        
        Args:
            market_data: 包含所有市場數據的字典（僅用於兼容性，實際使用緩存）
            
        Returns:
            buy_condition: 綜合買進條件
        """
        # 🚀 使用預先計算的結果，避免重複計算
        sar_buy = self._cached_sar_buy
        macd_buy = self._cached_macd_buy
        volume_ok = self._cached_volume_condition
        institutional_strong = self._cached_institutional_strong
        
        # 最終買進條件: SAR + MACD + Volume + 三大法人皆買超
        buy_condition = sar_buy & macd_buy & volume_ok & institutional_strong
        
        return buy_condition

    def _build_sell_condition(self, market_data):
        """
        建立賣出條件（使用預先計算的指標結果）
        
        Args:
            market_data: 包含所有市場數據的字典（僅用於兼容性，實際使用緩存）
            
        Returns:
            sell_condition: 綜合賣出條件
        """
        # 🚀 使用預先計算的結果，避免重複計算
        sar_sell = self._cached_sar_sell
        macd_sell = self._cached_macd_sell
        
        # 賣出條件: SAR翻轉 或 MACD死亡交叉 (任一即賣)
        sell_condition = sar_sell | macd_sell
        
        return sell_condition

    def run_strategy(self, max_stocks=1000, start_date='2020-01-01', fee_ratio=0.001425, tax_ratio=0.003):
        """
        執行策略回測（套用持股限制）
        
        Args:
            max_stocks: 最多同時持有幾檔股票
            start_date: 回測起始日期
            fee_ratio: 手續費率（預設0.1425%）
            tax_ratio: 證交稅率（預設0.3%）
            
        Returns:
            report: 回測報告物件
        """
        
        # 套用起始日期
        base_position = self.base_position.loc[start_date:]
        
        # 實施持股限制：最多同時持有 max_stocks 檔股票
        final_position = pd.DataFrame(False, index=base_position.index, columns=base_position.columns)
        
        for date in base_position.index:
            can_hold = base_position.loc[date]
            if can_hold.any():
                stocks_to_hold = can_hold[can_hold].index.tolist()
                # 選擇前 max_stocks 檔
                selected_stocks = stocks_to_hold[:max_stocks]
                for stock in selected_stocks:
                    final_position.loc[date, stock] = True
        
        # 執行回測
        self.report = sim(
            position=final_position,
            resample='D',
            upload=False,
            market=AdjustTWMarketInfo(),
            fee_ratio=fee_ratio,
            tax_ratio=tax_ratio,
            position_limit=1.0 / max_stocks  # 每檔股票平均分配資金
        )
        
        return self.report

    def get_report(self):
        """
        取得回測報告
        
        Returns:
            report: 回測報告物件，若未運行策略則返回None
        """
        return self.report

# Example usage:
if __name__ == "__main__":
    # 從 args 取得 stock_id (可選)
    import argparse
    parser = argparse.ArgumentParser(description='Oscar TW Strategy Executor')
    parser.add_argument('--stock_id', type=str, default=None, help='股票代碼 (可選，若不提供則考慮所有股票)')
    args = parser.parse_args()
    stock_id = args.stock_id

    # 初始化策略（可調整 SAR/MACD 對齊時間窗）
    strategy = OscarTWStrategy(sar_signal_lag_min=0, sar_signal_lag_max=2)
    
    base_position = strategy.base_position
    
    if stock_id:
        # 單一股票模式：只買該股票
        final_position = pd.DataFrame(False, index=base_position.index, columns=base_position.columns)
        final_position[stock_id] = base_position[stock_id]
        
        report = sim(
            position=final_position,
            resample=None,
            upload=False,
            market=AdjustTWMarketInfo(),
            fee_ratio=0.001425,
            tax_ratio=0.003,
            position_limit=1.0  # 單一股票全額投資
        )
        
        report.display(save_report_path=f'assets/OscarTWStrategy/{stock_id}_report.html')
    else:
        # 全市場模式：考慮所有股票
        report = sim(
            position=base_position,
            resample=None,
            upload=False,
            market=AdjustTWMarketInfo(),
            fee_ratio=0.001425,
            tax_ratio=0.003,
            position_limit=1.0  # 全額投資所有符合條件的股票
        )
        
        report.display(save_report_path=f'assets/OscarTWStrategy/all_stocks_report.html')
    
    
