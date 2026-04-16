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
from utils.config_loader import ConfigLoader


class _2560AndOrTWStrategy:
    """
    2560 均線波段策略

    Attributes:
        report:      回測報告物件
        buy_signal:  買入訊號 (DataFrame[bool])
        sell_signal: 賣出訊號 (DataFrame[bool])
    """

    def __init__(self, config_path="config.yaml", override_params=None):
        self.report = None
        self.config_loader = ConfigLoader(config_path)
        config_base = self.config_loader.config.get('_2560', {})
        config_andor = config_base.get('andor', {})

        # 如果有傳入實驗參數，就用實驗的；否則就讀 config.yaml 中的
        if override_params is None:
            override_params = {}

        self.config_loader = ConfigLoader(config_path)
        self.ma25_slope_lookback      = override_params.get('ma25_slope_lookback', config_andor.get('ma25_slope_lookback', 3))
        self.pullback_tolerance       = override_params.get('pullback_tolerance', config_andor.get('pullback_tolerance', 0.02))
        self.small_candle_threshold   = override_params.get('small_candle_threshold', config_andor.get('small_candle_threshold', 0.02))
        self.deviation_threshold      = override_params.get('deviation_threshold', config_andor.get('deviation_threshold', 0.15))
        self.surge_lookback           = override_params.get('surge_lookback', config_andor.get('surge_lookback', 5))
        self.surge_pct                = override_params.get('surge_pct', config_andor.get('surge_pct', 0.08))
        self.high_lookback            = override_params.get('high_lookback', config_andor.get('high_lookback', 20))
        self.stop_loss_pct            = override_params.get('stop_loss_pct', config_andor.get('stop_loss_pct', 0.07))
        self.take_profit_pct          = override_params.get('take_profit_pct', config_andor.get('take_profit_pct', 0.125))
        self.max_positions            = override_params.get('max_positions', config_andor.get('max_positions', 15))
        self.market_ma_period         = override_params.get('market_ma_period', config_andor.get('market_ma_period', 60))
        self.pullback_no_new_low_days = override_params.get('pullback_no_new_low_days', config_andor.get('pullback_no_new_low_days', 2))
        self.pullback_away_pct        = override_params.get('pullback_away_pct', config_andor.get('pullback_away_pct', 0.08))

        market_data = self._load_data()
        self._compute_indicators(market_data)

        self.buy_signal    = self._build_buy_signal()
        self.sell_signal   = self._build_sell_signal()
        self.base_position = self.buy_signal.hold_until(self.sell_signal)

    def _load_data(self) -> dict:
        with data.universe(market="TSE_OTC"):
            market_data = {
                "close":  data.get("price:收盤價"),
                "open":   data.get("price:開盤價"),
                "high":   data.get("price:最高價"),
                "low":    data.get("price:最低價"),
                "volume": data.get("price:成交股數") / 1000,  # 換算為張
            }

        # 大盤加權指數
        market_data["taiex"] = data.get("market_transaction_info:收盤指數")["TAIEX"]

        return market_data

    def _compute_indicators(self, market_data: dict) -> None:
        self.market_data = market_data
        close  = market_data["close"]
        volume = market_data["volume"]
        taiex  = market_data["taiex"]

        self.ma25     = close.average(25)
        self.vol_ma5  = volume.average(5)
        self.vol_ma60 = volume.average(60)

        # 大盤 MA60（Series）
        taiex_ma60 = taiex.rolling(
            self.market_ma_period, min_periods=self.market_ma_period // 2
        ).mean()

        # 大盤站上 MA60：True/False Series，對齊個股 index
        self.market_filter = (taiex > taiex_ma60).reindex(close.index).ffill()

    def _build_buy_signal(self) -> pd.DataFrame:
        close  = self.market_data["close"]
        open_  = self.market_data["open"]
        volume = self.market_data["volume"]

        # ── 大盤過濾：加權指數站上 MA60 才允許進場 ───────────────────────────
        # market_filter 是 Series，用 reindex 對齊後直接廣播
        market_series = self.market_filter.reindex(close.index).ffill().fillna(False)
        market_ok = close.apply(lambda _: market_series)

        # ── 前提條件（兩者必須同時成立）──────────────────────────────────────
        # 1. 趨勢確認：MA25 向上傾斜
        ma25_rising = self.ma25.rise(self.ma25_slope_lookback)

        # 2. 動能過濾：5日均量 > 60日均量
        vol_filter = self.vol_ma5 > self.vol_ma60

        prerequisites = market_ok & ma25_rising & vol_filter

        # ── 進場觸發（二選一）────────────────────────────────────────────────
        # A. 突破式進場：收盤站上 MA25 且當日成交量 > Vol_MA60
        breakout_entry = (
            (close > self.ma25)
            & (close.shift(1) <= self.ma25.shift(1))  # 昨天還在 MA25 之下
            & (volume > self.vol_ma60)
        ).fillna(False)

        # B. 回踩式進場：
        #    - 近 5 天內曾在 MA25 之上（從上方回踩，不含今天）
        #    - 價格回踩至 MA25 ±2%
        #    - 成交量縮量（< Vol_MA60）
        #    - 小星線止跌確認：當日漲跌幅 < small_candle_threshold
        #    - 不創新低：連續 N 天低點未創新低
        #    - 第一次回踩：曾遠離 MA25 超過 M%，且中間未再觸碰 MA25 附近
        low = self.market_data["low"]

        near_ma25 = (
            (close >= self.ma25)
            & ((close - self.ma25) / self.ma25.replace(0, np.nan) < self.pullback_tolerance)
        )
        low_volume = volume < self.vol_ma60

        # 小星線：以收盤相對前日收盤的漲跌幅來衡量當日波動幅度
        daily_change = (close - close.shift(1)).abs() / close.shift(1).replace(0, np.nan)
        small_candle = daily_change < self.small_candle_threshold

        # 不創新低：過去 N 天每日低點都高於前一日低點
        # rolling(N).min() 取近 N 天「low > low.shift(1)」的最小值，全為 True(1) 才通過
        no_new_low = (
            (low > low.shift(1))
            .rolling(self.pullback_no_new_low_days, min_periods=self.pullback_no_new_low_days)
            .min()
            .astype(bool)
        )

        # 第一次回踩：
        #   step1. 曾遠離 MA25（乖離 > M%），用 lookback 窗口找最近一次遠離的位置
        #   step2. 從那次遠離之後，到今天為止，中間沒有再進入 MA25 ±pullback_tolerance 範圍
        #          實作：近 30 天內曾遠離，且上次進入 near_ma25 之前先有一次遠離
        away_from_ma25 = (
            (close - self.ma25) / self.ma25.replace(0, np.nan) > self.pullback_away_pct
        )
        # 近 30 天內曾遠離過 MA25
        recently_away = (
            away_from_ma25.shift(1)
            .rolling(30, min_periods=1)
            .max()
            .astype(bool)
        )
        # 第一次接觸：昨天不在 MA25 附近，今天才是第一天進入
        # 避免在同一段回踩過程中重複進場
        first_pullback = recently_away & ~near_ma25.shift(1).fillna(False)

        pullback_entry = near_ma25 & low_volume & small_candle & no_new_low & first_pullback

        entry_signal = breakout_entry | pullback_entry
        # entry_signal = breakout_entry
        # entry_signal = pullback_entry

        return prerequisites & entry_signal

    def _build_sell_signal(self) -> pd.DataFrame:
        close = self.market_data["close"]
        high  = self.market_data["high"]

        # 退出觸發一：乖離過大（收盤高於 MA25 超過 15%）
        exit_deviation = (
            (close - self.ma25) / self.ma25.replace(0, np.nan) > self.deviation_threshold
        )

        # 退出觸發二：衝高失敗
        # 近 5 日漲幅 > 8%，但收盤未突破前 20 日最高點
        surge      = close / close.shift(self.surge_lookback).replace(0, np.nan) - 1
        prior_high = high.shift(1).rolling(
            self.high_lookback, min_periods=self.high_lookback // 2
        ).max()
        exit_failed_surge = (surge > self.surge_pct) & (close < prior_high)

        # 技術停損：收盤跌破 MA25
        exit_tech_stop = close < self.ma25

        # ※ exit_vol_cross 已移除（Vol_MA5 < Vol_MA60 與回踩進場邏輯自相矛盾）

        return (
            exit_deviation
            | exit_failed_surge
            | exit_tech_stop
        )

    def _rank_and_limit(self, selected_mask: pd.DataFrame) -> pd.DataFrame:
        """
        以量能比率（Vol_MA5 / Vol_MA60）對當日訊號股票排序，
        只保留前 max_positions 檔。
        """
        vol_ratio  = self.vol_ma5 / self.vol_ma60.replace(0, np.nan)
        score      = vol_ratio.where(selected_mask)
        ranked     = score.rank(axis=1, ascending=False, method="first")
        top_n_mask = ranked <= self.max_positions

        return top_n_mask.fillna(False).astype(bool)

    def run_strategy(
        self,
        start_date: str = "2020-01-01",
        fee_ratio: float = 0.001425,
        tax_ratio: float = 0.003,
    ):
        base_position = self.base_position.loc[start_date:]

        if len(base_position.index) == 0:
            raise ValueError(f"start_date={start_date} 之後無可用交易日，請確認日期範圍。")

        selected_mask = base_position.astype(bool)

        # 量能排序 + 持股數限制
        top_n_mask = self._rank_and_limit(selected_mask)

        # 等權重分配倉位
        top_n_count    = top_n_mask.sum(axis=1)
        final_position = top_n_mask.div(
            top_n_count.replace(0, np.nan), axis=0
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
        config_path="config.yaml",
    )

    report = strategy.run_strategy(start_date="2020-01-01")
    print(report)