"""
2560 均線波段策略（簡化版）
持倉期間: 5–30 個交易日

核心指標:
    - MA25:      25日價格均線（趨勢線）
    - Vol_MA5:   5日成交量均線（短期動能）
    - Vol_MA60:  60日成交量均線（基準過濾）
"""

import numpy as np
import pandas as pd
from finlab import data
from finlab.backtest import sim

class _2560TWStrategy:
    """
    2560 均線波段策略

    Attributes:
        report:      回測報告物件
        buy_signal:  買入訊號 (DataFrame[bool])
        sell_signal: 賣出訊號 (DataFrame[bool])
    """

    def __init__(
        self,
        ma25_slope_lookback: int = 3,       # MA25 向上判斷回溯天數
        pullback_tolerance: float = 0.02,   # 回踩容差 ±2%
        deviation_threshold: float = 0.15,  # 退出觸發一：乖離門檻 15%
        surge_lookback: int = 5,            # 退出觸發三：漲幅計算天數
        surge_pct: float = 0.08,            # 退出觸發三：漲幅門檻 8%
        high_lookback: int = 20,            # 退出觸發三：前高回溯天數
        stop_loss_pct: float = 0.07,        # 百分比停損 7%（5–8% 中間值）
        take_profit_pct: float = 0.125,     # 停利 12.5%（10–15% 中間值）
    ):
        self.ma25_slope_lookback  = ma25_slope_lookback
        self.pullback_tolerance   = pullback_tolerance
        self.deviation_threshold  = deviation_threshold
        self.surge_lookback       = surge_lookback
        self.surge_pct            = surge_pct
        self.high_lookback        = high_lookback
        self.stop_loss_pct        = stop_loss_pct
        self.take_profit_pct      = take_profit_pct

        self.report = None

        market_data = self._load_data()
        self._compute_indicators(market_data)

        self.buy_signal  = self._build_buy_signal()
        self.sell_signal = self._build_sell_signal()
        self.base_position = self.buy_signal.hold_until(self.sell_signal)

    def _load_data(self) -> dict:
        with data.universe(market="TSE_OTC"):
            market_data = {
                "close":  data.get("price:收盤價"),
                "high":   data.get("price:最高價"),
                "volume": data.get("price:成交股數") / 1000,  # 換算為張
            }
        return market_data

    def _compute_indicators(self, market_data: dict) -> None:
        self.market_data = market_data
        close  = market_data["close"]
        volume = market_data["volume"]

        self.ma25     = close.average(25)
        self.vol_ma5  = volume.average(5)
        self.vol_ma60 = volume.average(60)

    def _build_buy_signal(self) -> pd.DataFrame:
        close  = self.market_data["close"]
        volume = self.market_data["volume"]

        # ── 前提條件（兩者必須同時成立）──────────────────────────────────────
        # 1. 趨勢確認：MA25 向上傾斜
        ma25_rising = self.ma25.rise(self.ma25_slope_lookback)

        # 2. 動能過濾：5日均量 > 60日均量
        vol_filter = self.vol_ma5 > self.vol_ma60

        prerequisites = ma25_rising & vol_filter

        # ── 進場觸發（二選一）────────────────────────────────────────────────
        # A. 突破式進場：收盤站上 MA25 且當日成交量 > Vol_MA60（放量突破）
        breakout_entry = (
            (close > self.ma25) & (volume > self.vol_ma60)
        ).is_entry()

        # B. 回踩式進場：價格回踩至 MA25 ±2%，且成交量 < Vol_MA60（縮量）
        near_ma25      = (close - self.ma25).abs() / self.ma25.replace(0, np.nan) < self.pullback_tolerance
        low_volume     = volume < self.vol_ma60
        pullback_entry = (near_ma25 & low_volume).is_entry()

        entry_signal = breakout_entry | pullback_entry

        return prerequisites & entry_signal

    def _build_sell_signal(self) -> pd.DataFrame:
        close = self.market_data["close"]
        high  = self.market_data["high"]

        # 退出觸發一：乖離過大（收盤高於 MA25 超過 15%）
        exit_deviation = (
            (close - self.ma25) / self.ma25.replace(0, np.nan) > self.deviation_threshold
        )

        # 退出觸發二：資金動能轉弱（Vol_MA5 跌破 Vol_MA60）
        exit_vol_cross = (self.vol_ma5 < self.vol_ma60).is_entry()

        # 退出觸發三：衝高失敗
        # 近 5 日漲幅 > 8%，但收盤未突破前 20 日最高點
        surge = close / close.shift(self.surge_lookback).replace(0, np.nan) - 1
        prior_high = high.shift(1).rolling(self.high_lookback, min_periods=self.high_lookback // 2).max()
        exit_failed_surge = ((surge > self.surge_pct) & (close < prior_high)).is_entry()

        # 技術停損：收盤跌破 MA25
        exit_tech_stop = close < self.ma25

        return (
            exit_deviation
            | exit_vol_cross
            | exit_failed_surge
            | exit_tech_stop
        )

    def run_strategy(
        self,
        start_date: str = "2020-01-01",
        fee_ratio: float = 0.001425,
        tax_ratio: float = 0.003,
    ):
        base_position = self.base_position.loc[start_date:]

        if len(base_position.index) == 0:
            raise ValueError(f"start_date={start_date} 之後無可用交易日，請確認日期範圍。")

        selected_mask  = base_position.astype(bool)
        selected_count = selected_mask.sum(axis=1)

        # 等權重分配倉位（每筆佔總資金相等比例）
        final_position = selected_mask.div(
            selected_count.replace(0, np.nan), axis=0
        ).fillna(0.0)

        self.report = sim(
            position=final_position,
            resample="D",
            upload=False,
            trade_at_price="open",
            fee_ratio=fee_ratio,
            tax_ratio=tax_ratio,
            stop_loss=self.stop_loss_pct,
            take_profit=self.take_profit_pct,
            position_limit=1.0,
        )

        return self.report

    def get_report(self):
        if self.report is None:
            return "report 物件為空，請先運行策略"
        return self.report

if __name__ == "__main__":
    strategy = _2560TWStrategy(
        ma25_slope_lookback=3,
        pullback_tolerance=0.02,
        deviation_threshold=0.15,
        surge_lookback=5,
        surge_pct=0.08,
        high_lookback=20,
        stop_loss_pct=0.07,
        take_profit_pct=0.125,
    )

    report = strategy.run_strategy(start_date="2020-01-01")
    print(report)