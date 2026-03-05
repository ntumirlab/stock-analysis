"""
00635U 布林通道策略 (Bollinger Bands Strategy for 00635U)

針對 00635U (元大台灣50正向1倍ETF) 的布林通道交易策略

核心邏輯:
- 布林通道由三條線組成：上軌、中軌(簡單移動平均線)、下軌
- 買進: 價格接近下軌（距離下軌5%範圍內）且價格向上
- 賣出: 價格接近上軌（距離上軌5%範圍內）或價格跌破中軌
- RSI輔助: 結合RSI指標避免超買超賣

策略特點:
🎯 專注單一股票 00635U (流動性充足)
📊 使用布林通道+RSI雙重確認
⚡ 適合短中期交易
🛡️ 內建止損機制 (跌破20日均線)

性能優化:
- 預先計算所有指標，避免重複計算
- 支持預載數據
"""

from finlab import data
from finlab.markets.tw import TWMarket
from finlab.backtest import sim
import pandas as pd
import numpy as np
from utils.config_loader import ConfigLoader


class AdjustTWMarketInfo(TWMarket):
    """自訂市場資訊類別，用於調整交易價格為開盤價"""
    def get_trading_price(self, name, adj=True):
        # 使用當日開盤價作為交易價格
        return self.get_price('open', adj=adj)


class BollingerBands00635UStrategy:
    """
    00635U 布林通道策略
    
    專注於單一股票 00635U 的布林通道交易策略
    
    Attributes:
        report: 回測報告物件
        position: 持倉訊號
        buy_signal: 買入訊號
        sell_signal: 賣出訊號
    """
    
    def __init__(self, 
                 stock_id='00635U',
                 sma_period=20,
                 bb_std_dev=2,
                 bb_lower_pct=0.05,
                 bb_upper_pct=0.05,
                 rsi_period=14,
                 rsi_oversold=30,
                 rsi_overbought=70,
                 config_path="config.yaml",
                 market_data=None):
        """
        初始化策略參數
        
        Args:
            stock_id: 股票代碼，預設為 '00635U'
            sma_period: 簡單移動平均線週期，預設 20 日
            bb_std_dev: 布林通道標準差倍數，預設 2
            bb_lower_pct: 認定接近下軌的距離（百分比），預設 5%
            bb_upper_pct: 認定接近上軌的距離（百分比），預設 5%
            rsi_period: RSI 計算週期，預設 14 日
            rsi_oversold: RSI 超賣點位，預設 30
            rsi_overbought: RSI 超買點位，預設 70
            config_path: 設定檔路徑
            market_data: 預先載入的市場數據
        """
        # 策略參數
        self.stock_id = stock_id
        self.sma_period = sma_period
        self.bb_std_dev = bb_std_dev
        self.bb_lower_pct = bb_lower_pct
        self.bb_upper_pct = bb_upper_pct
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        # 載入配置
        try:
            self.config_loader = ConfigLoader(config_path)
        except:
            self.config_loader = None
        
        # 回測報告
        self.report = None
        
        # 載入市場數據
        market_data = market_data if market_data is not None else self._load_data()
        self.market_data = market_data
        
        # 儲存指標供視覺化使用
        self.sma = None
        self.bb_upper = None
        self.bb_lower = None
        self.bb_middle = None
        self.rsi = None
        self.trade_price = market_data['open'][stock_id]
        
        # 計算布林通道和RSI指標
        self._calculate_bollinger_bands(market_data['close'][stock_id])
        self._calculate_rsi(market_data['close'][stock_id])
        
        # 構建買賣訊號
        self.buy_signal = self._build_buy_condition(market_data['close'][stock_id])
        self.sell_signal = self._build_sell_condition(market_data['close'][stock_id])
        
        # 基礎持倉訊號 (買進後持倉直到賣出)
        self.base_position = self.buy_signal.astype(bool).to_frame(name=stock_id)
    
    @staticmethod
    def load_market_data():
        """
        靜態方法：載入市場數據
        
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
            }
    
    def _load_data(self):
        """
        載入所需數據（實例方法）
        
        Returns:
            dict: 包含所有必要數據的字典
        """
        return self.load_market_data()
    
    def _calculate_bollinger_bands(self, close):
        """
        計算布林通道指標
        
        Args:
            close: 收盤價 (單一股票 Series)
        """
        # 計算簡單移動平均線 (中軌)
        sma = close.rolling(window=self.sma_period).mean()
        
        # 計算標準差
        std_dev = close.rolling(window=self.sma_period).std()
        
        # 計算上軌和下軌
        bb_upper = sma + (std_dev * self.bb_std_dev)
        bb_lower = sma - (std_dev * self.bb_std_dev)
        
        # 儲存值供視覺化使用
        self.sma = sma
        self.bb_upper = bb_upper
        self.bb_lower = bb_lower
        self.bb_middle = sma
    
    def _calculate_rsi(self, close):
        """
        計算 RSI 指標 (相對強弱指數)
        
        Args:
            close: 收盤價 (單一股票 Series)
        """
        # 計算價格變動
        delta = close.diff()
        
        # 分離上升和下降
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 計算平均收益和平均虧損
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        # 計算相對強度 (RS)
        rs = avg_gain / avg_loss
        
        # 計算 RSI
        rsi = 100 - (100 / (1 + rs))
        
        # 儲存值供視覺化使用
        self.rsi = rsi
    
    def _build_buy_condition(self, close):
        """
        建立買進條件
        
        買進邏輯:
        1. 價格接近下軌 (在下軌±5%範圍內)
        2. RSI 在超賣區域 (< 30)
        3. 價格開始向上反彈 (今日收盤價 > 前一日收盤價)
        4. 價格在簡單移動平均線上方
        
        Args:
            close: 收盤價
            
        Returns:
            buy_condition: 買進訊號 (布林值 Series)
        """
        # 條件1: 價格接近下軌
        lower_bound = self.bb_lower * (1 + self.bb_lower_pct)
        price_near_lower = close <= lower_bound
        price_above_lower = close >= (self.bb_lower * (1 - self.bb_lower_pct))
        price_near_lower_band = price_near_lower & price_above_lower
        
        # 條件2: RSI 超賣
        rsi_oversold = self.rsi < self.rsi_oversold
        
        # 條件3: 價格向上反彈
        price_bouncing_up = close > close.shift(1)
        
        # 條件4: 價格在中軌上方
        price_above_sma = close > self.sma
        
        # 綜合買進條件
        buy_condition = price_near_lower_band & rsi_oversold & price_bouncing_up & price_above_sma
        
        return buy_condition
    
    def _build_sell_condition(self, close):
        """
        建立賣出條件
        
        賣出邏輯:
        1. 價格接近上軌 (在上軌±5%範圍內) OR
        2. RSI 在超買區域 (> 70) OR
        3. 價格跌破20日簡單移動平均線 (止損)
        
        Args:
            close: 收盤價
            
        Returns:
            sell_condition: 賣出訊號 (布林值 Series)
        """
        # 條件1: 價格接近上軌
        upper_bound = self.bb_upper * (1 - self.bb_upper_pct)
        price_near_upper = close >= upper_bound
        price_below_upper = close <= (self.bb_upper * (1 + self.bb_upper_pct))
        price_near_upper_band = price_near_upper & price_below_upper
        
        # 條件2: RSI 超買
        rsi_overbought = self.rsi > self.rsi_overbought
        
        # 條件3: 價格跌破停損線 (20日均線)
        stop_loss_line = close.rolling(window=20).mean()
        price_break_stop_loss = close < stop_loss_line
        
        # 綜合賣出條件 (任一即賣)
        sell_condition = price_near_upper_band | rsi_overbought | price_break_stop_loss
        
        return sell_condition
    
    def run_strategy(self, start_date='2020-01-01', fee_ratio=0.001425, tax_ratio=0.003):
        """
        執行策略回測
        
        Args:
            start_date: 回測起始日期
            fee_ratio: 手續費率（預設0.1425%）
            tax_ratio: 證交稅率（預設0.3%）
            
        Returns:
            report: 回測報告物件
        """
        # 構建持倉訊號：買進後持倉直到賣出訊號出現
        position = pd.DataFrame(False, index=self.buy_signal.index, columns=[self.stock_id])
        in_position = False
        
        for date in self.buy_signal.index:
            if self.buy_signal[date] and not in_position:
                in_position = True
            
            if in_position:
                position.loc[date, self.stock_id] = True
            
            if in_position and self.sell_signal[date]:
                in_position = False
        
        # 套用起始日期
        position = position.loc[start_date:]
        
        # 執行回測
        self.report = sim(
            position=position,
            resample='D',
            upload=False,
            market=AdjustTWMarketInfo(),
            fee_ratio=fee_ratio,
            tax_ratio=tax_ratio,
            position_limit=1.0  # 100% 投入單一股票
        )
        
        return self.report
    
    def get_report(self):
        """
        取得回測報告
        
        Returns:
            report: 回測報告物件，若未運行策略則返回None
        """
        return self.report
    
    def get_indicators_data(self):
        """
        取得所有指標數據用於視覺化
        
        Returns:
            dict: 包含所有指標數據的字典
        """
        return {
            'sma': self.sma,
            'bb_upper': self.bb_upper,
            'bb_lower': self.bb_lower,
            'rsi': self.rsi,
            'close': self.market_data['close'][self.stock_id],
            'buy_signal': self.buy_signal,
            'sell_signal': self.sell_signal
        }


# Example usage:
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Bollinger Bands Strategy for 00635U')
    parser.add_argument('--stock_id', type=str, default='00635U', help='股票代碼，預設為 00635U')
    parser.add_argument('--sma_period', type=int, default=20, help='SMA 週期')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='回測起始日期')
    args = parser.parse_args()
    
    # 初始化策略
    strategy = BollingerBands00635UStrategy(
        stock_id=args.stock_id,
        sma_period=args.sma_period
    )
    
    # 運行回測
    print(f"執行 {args.stock_id} 布林通道策略回測...")
    report = strategy.run_strategy(start_date=args.start_date)
    
    # 輸出報告
    print("\n=== 回測結果 ===")
    print(f"策略收益率: {report.stats()['Annual Return']:.2%}")
    print(f"最大回撤: {report.stats()['Max Drawdown']:.2%}")
    print(f"夏普比例: {report.stats()['Sharpe Ratio']:.2f}")
    print(f"贏率: {report.stats()['Win Rate']:.2%}")
