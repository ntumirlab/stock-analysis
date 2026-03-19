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
- 選股與權重: 所有符合條件的股票皆納入持股，並採等權重配置
- 特別注意: 排除不斷創新高的股票，避免追高

股票範圍:
全台股市場 (TSE + OTC)
"""

from __future__ import annotations

from finlab import data
from finlab.backtest import sim
import pandas as pd
import numpy as np

from utils.config_loader import ConfigLoader


class OscarAndOrStrategy:
    """Oscar AND/OR strategy with clean base-style boolean logic."""

    def __init__(
        self,
        sar_signal_lag_min: int = 0,
        sar_signal_lag_max: int = 2,
        config_path: str = "config.yaml",
        sar_params: dict | None = None,
        macd_params: dict | None = None,
        market_data: dict | None = None,
        volume_above_avg_ratio: float | None = None,
        new_high_ratio_120: float | None = None,
        sar_max_dots: int | None = None,
        sar_reject_dots: int | None = None,
        macd_signal_lag_min: int | None = None,
        macd_signal_lag_max: int | None = None,
        min_avg_volume_30: float | None = None,
        max_volume_spike_ratio: float | None = None,
    ):
        self.config_loader = ConfigLoader(config_path)
        oscar_root = self.config_loader.config.get("oscar", {})
        if isinstance(oscar_root, dict) and any(k in oscar_root for k in ("general", "andor", "composite")):
            general_cfg = oscar_root.get("general", {}) or {}
            mode_cfg = oscar_root.get("andor", {}) or {}
            oscar_config = {**general_cfg, **mode_cfg}
        else:
            oscar_config = oscar_root

        if sar_max_dots is not None and sar_signal_lag_max == 2:
            sar_signal_lag_max = int(sar_max_dots)

        self.sar_signal_lag_min = max(0, int(sar_signal_lag_min))
        self.sar_signal_lag_max = max(self.sar_signal_lag_min, int(sar_signal_lag_max))
        self.macd_signal_lag_min = max(0, int(macd_signal_lag_min or 0))
        self.macd_signal_lag_max = max(self.macd_signal_lag_min, int(macd_signal_lag_max or 0))

        # Backward-compatible fields retained for external callers.
        self.sar_max_dots = sar_max_dots
        self.sar_reject_dots = sar_reject_dots

        self.sar_params = sar_params or {"acceleration": 0.02, "maximum": 0.2}
        self.macd_params = macd_params or {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}

        self.volume_above_avg_ratio = float(
            volume_above_avg_ratio if volume_above_avg_ratio is not None else oscar_config.get("volume_above_avg_ratio", 0.25)
        )
        self.new_high_ratio_120 = float(
            new_high_ratio_120 if new_high_ratio_120 is not None else oscar_config.get("new_high_ratio_120", 0.3)
        )
        self.min_avg_volume_30 = float(
            min_avg_volume_30 if min_avg_volume_30 is not None else oscar_config.get("min_avg_volume_30", 1_000_000)
        )
        self.max_volume_spike_ratio = float(
            max_volume_spike_ratio if max_volume_spike_ratio is not None else oscar_config.get("max_volume_spike_ratio", 10.0)
        )

        self.report = None
        market_data = market_data if market_data is not None else self._load_data()
        self.market_data = market_data

        self.trade_price = market_data["open"]
        self.sar_values = None
        self.macd_dif = None
        self.macd_dea = None
        self.macd_histogram = None
        self.institutional_condition = {
            "foreign_buy": None,
            "trust_buy": None,
            "dealer_buy": None,
        }

        self._cached_sar_buy, self._cached_sar_sell = self._calculate_sar_condition(market_data["close"])
        self._cached_macd_buy, self._cached_macd_sell = self._calculate_macd_condition(market_data["close"])
        self._cached_volume_condition = self._calculate_volume_condition(market_data["volume"])
        self._cached_institutional_strong, self._cached_institutional_weak = self._calculate_institutional_condition(
            market_data["foreign_net_buy_shares"],
            market_data["investment_trust_net_buy_shares"],
            market_data["dealer_self_net_buy_shares"],
        )

        self.buy_signal = self._build_buy_condition()
        self.sell_signal = self._build_sell_condition()
        self.base_position = self.buy_signal.hold_until(self.sell_signal)

    @classmethod
    def clear_runtime_cache(cls):
        # Backward compatibility for legacy callers.
        return None

    @staticmethod
    def load_market_data() -> dict:
        with data.universe(market="TSE_OTC"):
            return {
                "open": data.get("price:開盤價"),
                "close": data.get("price:收盤價"),
                "high": data.get("price:最高價"),
                "low": data.get("price:最低價"),
                "volume": data.get("price:成交股數"),
                "foreign_net_buy_shares": data.get(
                    "institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)"
                ),
                "investment_trust_net_buy_shares": data.get(
                    "institutional_investors_trading_summary:投信買賣超股數"
                ),
                "dealer_self_net_buy_shares": data.get(
                    "institutional_investors_trading_summary:自營商買賣超股數(自行買賣)"
                ),
            }

    def _load_data(self) -> dict:
        return self.load_market_data()

    @staticmethod
    def _expand_event_window(event_df: pd.DataFrame, lag_min: int, lag_max: int) -> pd.DataFrame:
        lagged_signals = [event_df.shift(lag).fillna(False) for lag in range(lag_min, lag_max + 1)]
        expanded = lagged_signals[0]
        for signal in lagged_signals[1:]:
            expanded = expanded | signal
        return expanded

    def _calculate_sar_condition(self, close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        sar = data.indicator(
            "SAR",
            acceleration=self.sar_params["acceleration"],
            maximum=self.sar_params["maximum"],
            adjust_price=True,
        ).reindex(index=close.index, columns=close.columns)

        self.sar_values = sar

        sar_below_price = sar < close
        sar_below_price_prev = sar_below_price.shift(1).fillna(sar_below_price)
        sar_flip_bullish = sar_below_price & (~sar_below_price_prev)

        sar_in_alignment_window = self._expand_event_window(
            sar_flip_bullish,
            self.sar_signal_lag_min,
            self.sar_signal_lag_max,
        )

        high_120 = close.rolling(120).max()
        is_new_high = close >= high_120
        new_high_ratio_120 = is_new_high.rolling(120).mean()
        constantly_new_high = new_high_ratio_120 > self.new_high_ratio_120

        sar_buy_condition = sar_in_alignment_window & (~constantly_new_high)
        sar_sell_condition = ~sar_below_price
        return sar_buy_condition, sar_sell_condition

    def _calculate_macd_condition(self, close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        dif, dea, histogram = data.indicator(
            "MACD",
            fastperiod=self.macd_params["fastperiod"],
            slowperiod=self.macd_params["slowperiod"],
            signalperiod=self.macd_params["signalperiod"],
            adjust_price=True,
        )
        dif = dif.reindex(index=close.index, columns=close.columns)
        dea = dea.reindex(index=close.index, columns=close.columns)
        histogram = histogram.reindex(index=close.index, columns=close.columns)

        self.macd_dif = dif
        self.macd_dea = dea
        self.macd_histogram = histogram

        macd_cross_bullish = (dif > dea) & (dif.shift(1) <= dea.shift(1))
        macd_buy_condition = self._expand_event_window(
            macd_cross_bullish,
            self.macd_signal_lag_min,
            self.macd_signal_lag_max,
        )
        macd_sell_condition = (dif < dea) & (dif.shift(1) >= dea.shift(1))

        return macd_buy_condition, macd_sell_condition

    def _calculate_volume_condition(self, volume: pd.DataFrame) -> pd.DataFrame:
        avg_volume_30 = volume.rolling(30).mean()

        volume_above_avg = volume > (avg_volume_30 * self.volume_above_avg_ratio)
        sufficient_liquidity = avg_volume_30 > self.min_avg_volume_30
        not_abnormal_volume = volume < (avg_volume_30 * self.max_volume_spike_ratio)

        return volume_above_avg & sufficient_liquidity & not_abnormal_volume

    def _calculate_institutional_condition(
        self,
        foreign_net_buy: pd.DataFrame,
        trust_net_buy: pd.DataFrame,
        dealer_net_buy: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        foreign_buy = foreign_net_buy > 0
        trust_buy = trust_net_buy > 0
        dealer_buy = dealer_net_buy > 0

        self.institutional_condition = {
            "foreign_buy": foreign_buy,
            "trust_buy": trust_buy,
            "dealer_buy": dealer_buy,
        }

        buy_count = foreign_buy.astype(int) + trust_buy.astype(int) + dealer_buy.astype(int)
        institutional_strong_condition = buy_count == 3
        institutional_weak_condition = buy_count >= 2

        return institutional_strong_condition, institutional_weak_condition

    def _build_buy_condition(self) -> pd.DataFrame:
        return (
            self._cached_sar_buy
            & self._cached_macd_buy
            & self._cached_volume_condition
            & self._cached_institutional_strong
        )

    def _build_sell_condition(self) -> pd.DataFrame:
        return self._cached_sar_sell | self._cached_macd_sell

    @staticmethod
    def _build_equal_weight_position(selected_mask: pd.DataFrame) -> pd.DataFrame:
        selected_count = selected_mask.sum(axis=1)
        return selected_mask.div(selected_count.replace(0, np.nan), axis=0).fillna(0.0)

    @staticmethod
    def _apply_max_stocks_mask(selected_mask: pd.DataFrame, max_stocks: int) -> pd.DataFrame:
        max_n = max(1, int(max_stocks))
        capped_mask = pd.DataFrame(False, index=selected_mask.index, columns=selected_mask.columns)
        for dt in selected_mask.index:
            row = selected_mask.loc[dt]
            chosen = row[row].index[:max_n]
            if len(chosen) > 0:
                capped_mask.loc[dt, chosen] = True
        return capped_mask

    def run_strategy(
        self,
        max_stocks: int | None = None,
        start_date: str = "2020-01-01",
        fee_ratio: float = 0.001425,
        tax_ratio: float = 0.003,
        sim_resample: str = "D",
    ):
        base_position = self.base_position.loc[start_date:]
        if len(base_position.index) == 0:
            raise ValueError("No trading days available after start_date.")

        selected_mask = base_position.astype(bool)
        if max_stocks is not None:
            selected_mask = self._apply_max_stocks_mask(selected_mask, max_stocks=max_stocks)

        final_position = self._build_equal_weight_position(selected_mask)

        self.report = sim(
            position=final_position,
            resample=sim_resample,
            upload=False,
            trade_at_price="open",
            fee_ratio=fee_ratio,
            tax_ratio=tax_ratio,
            position_limit=1.0,
        )

        return self.report

    def get_report(self):
        return self.report


if __name__ == "__main__":
    strategy = OscarAndOrStrategy(sar_signal_lag_min=0, sar_signal_lag_max=2)
    report = strategy.run_strategy(start_date="2020-01-01")
    report.display(save_report_path="assets/OscarTWStrategy/andor_report.html")
