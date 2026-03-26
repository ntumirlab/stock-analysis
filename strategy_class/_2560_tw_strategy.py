"""
2560 MA Band Strategy — 中期波段交易策略
持倉期間: 5–30 個交易日 | 股票範圍: TSE + OTC
"""

import numpy as np
import pandas as pd
from finlab import data
from finlab.markets.tw import TWMarket
from finlab.backtest import sim
from time import perf_counter
from utils.config_loader import ConfigLoader


class AdjustTWMarketInfo(TWMarket):
    """訊號使用收盤後資料，對齊至下一交易日開盤成交。"""

    def get_trading_price(self, name, adj=True):
        return self.get_price("open", adj=adj).shift(1)


class _2560TWStrategy:
    """
    2560 均線波段策略

    Attributes:
        report:         回測報告物件
        buy_signal:     買入訊號 (DataFrame[bool])
        sell_signal:    賣出訊號 (DataFrame[bool])
        base_position:  基礎持倉訊號（已套用百分比停損）
    """

    _MARKET_DATA_CACHE = None

    def __init__(
        self,
        config_path="config.yaml",
        market_data=None,
        ma_short: int = 5,
        ma_mid: int = 25,
        ma_long: int = 60,
        ma_extra_long: int = 120,
        ma25_slope_lookback: int = 3,
        ma5_slope_lookback: int = 3,
        ma5_slope_tolerance: float = 0.97,
        pullback_ma25_tolerance: float = 0.03,
        pullback_volume_ratio: float = 0.8,
        doji_threshold: float = 0.01,
        deviation_exit_threshold: float = 0.15,
        kdj_j_overbought: float = 100.0,
        kdj_j_single_day_turn: bool = True,
        kdj_j_turn_threshold: float = 80.0,
        resistance_lookback: int = 60,
        resistance_tolerance: float = 0.03,
        stop_loss_pct: float = 0.07,
        time_stop_days: int = 0,
        tech_stop_volume_ratio: float = 1.5,
        kdj_params: dict = None,
        macd_params: dict = None,
        kdj_golden_cross_window: int = 3,
        min_avg_volume_30: float = 1_000_000,
        preprocess_start_date: str = None,
        preprocess_lookback_days: int = 252,
        enable_profiling: bool = False,
        sim_fast_mode_default: bool = False,
    ):
        init_t0 = perf_counter()
        self.enable_profiling = bool(enable_profiling)
        self.profile_stats = {}

        self.config_loader = ConfigLoader(config_path)
        cfg = self.config_loader.config.get("strategy_2560", {}) or {}

        self.ma_short                 = int(cfg.get("ma_short", ma_short))
        self.ma_mid                   = int(cfg.get("ma_mid", ma_mid))
        self.ma_long                  = int(cfg.get("ma_long", ma_long))
        self.ma_extra_long            = int(cfg.get("ma_extra_long", ma_extra_long))
        self.ma25_slope_lookback      = int(cfg.get("ma25_slope_lookback", ma25_slope_lookback))
        self.ma5_slope_lookback       = int(cfg.get("ma5_slope_lookback", ma5_slope_lookback))
        self.ma5_slope_tolerance      = float(cfg.get("ma5_slope_tolerance", ma5_slope_tolerance))
        self.pullback_ma25_tolerance  = float(cfg.get("pullback_ma25_tolerance", pullback_ma25_tolerance))
        self.pullback_volume_ratio    = float(cfg.get("pullback_volume_ratio", pullback_volume_ratio))
        self.doji_threshold           = float(cfg.get("doji_threshold", doji_threshold))
        self.deviation_exit_threshold = float(cfg.get("deviation_exit_threshold", deviation_exit_threshold))
        self.kdj_j_overbought         = float(cfg.get("kdj_j_overbought", kdj_j_overbought))
        self.kdj_j_single_day_turn    = bool(cfg.get("kdj_j_single_day_turn", kdj_j_single_day_turn))
        self.kdj_j_turn_threshold     = float(cfg.get("kdj_j_turn_threshold", kdj_j_turn_threshold))
        self.resistance_lookback      = int(cfg.get("resistance_lookback", resistance_lookback))
        self.resistance_tolerance     = float(cfg.get("resistance_tolerance", resistance_tolerance))
        self.stop_loss_pct            = float(cfg.get("stop_loss_pct", stop_loss_pct))
        self.time_stop_days           = int(cfg.get("time_stop_days", time_stop_days))
        self.tech_stop_volume_ratio   = float(cfg.get("tech_stop_volume_ratio", tech_stop_volume_ratio))
        self.kdj_params               = kdj_params or {"fastk_period": 9, "signal": 3}
        self.macd_params              = macd_params or {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}
        self.kdj_golden_cross_window  = int(cfg.get("kdj_golden_cross_window", kdj_golden_cross_window))
        self.min_avg_volume_30        = float(cfg.get("min_avg_volume_30", min_avg_volume_30))
        self.preprocess_start_date    = preprocess_start_date or cfg.get("preprocess_start_date")
        self.preprocess_lookback_days = max(0, int(cfg.get("preprocess_lookback_days", preprocess_lookback_days)))
        self.sim_fast_mode_default    = bool(cfg.get("sim_fast_mode", sim_fast_mode_default))

        self.report = None

        load_t0 = perf_counter()
        market_data = market_data if market_data is not None else self._load_data()
        self._record_profile("load_market_data_sec", load_t0)

        scope_t0 = perf_counter()
        market_data = self._scope_market_data(market_data)
        self._record_profile("scope_market_data_sec", scope_t0)

        self.market_data = market_data
        self.trade_price = market_data["open"]

        indicator_t0 = perf_counter()
        self._precompute_indicators(market_data)
        self._record_profile("calculate_indicators_sec", indicator_t0)

        self.buy_signal  = self._build_buy_condition()
        self.sell_signal = self._build_sell_condition()

        _sl = self.stop_loss_pct if self.stop_loss_pct > 0 else -np.inf
        self.base_position = self.buy_signal.hold_until(
            self.sell_signal,
            stop_loss=_sl,
        )
        self._record_profile("init_total_sec", init_t0)

    # =========================================================================
    # 數據載入
    # =========================================================================

    @staticmethod
    def load_market_data() -> dict:
        if _2560TWStrategy._MARKET_DATA_CACHE is not None:
            return _2560TWStrategy._MARKET_DATA_CACHE

        with data.universe(market="TSE_OTC"):
            loaded = {
                "open":   data.get("price:開盤價"),
                "close":  data.get("price:收盤價"),
                "high":   data.get("price:最高價"),
                "volume": data.get("price:成交股數"),
            }

        _2560TWStrategy._MARKET_DATA_CACHE = loaded
        return loaded

    def _load_data(self) -> dict:
        return self.load_market_data()

    @classmethod
    def clear_cache(cls):
        cls._MARKET_DATA_CACHE = None

    # =========================================================================
    # 輔助工具
    # =========================================================================

    def _record_profile(self, key: str, start_t: float) -> None:
        if self.enable_profiling:
            self.profile_stats[key] = round(perf_counter() - start_t, 6)

    def _scope_market_data(self, market_data: dict) -> dict:
        if not self.preprocess_start_date:
            return market_data

        start_ts     = pd.Timestamp(self.preprocess_start_date)
        scoped_start = start_ts - pd.Timedelta(days=self.preprocess_lookback_days)
        scoped = {}
        for key, df in market_data.items():
            if isinstance(df, pd.DataFrame):
                sliced = df.loc[df.index >= scoped_start]
                scoped[key] = sliced if len(sliced) > 0 else df
            else:
                scoped[key] = df
        return scoped

    # =========================================================================
    # 指標計算
    # =========================================================================

    def _precompute_indicators(self, market_data: dict) -> None:
        close  = market_data["close"]
        volume = market_data["volume"]

        self.ma5   = close.average(self.ma_short)
        self.ma25  = close.average(self.ma_mid)
        self.ma60  = close.average(self.ma_long)
        self.ma120 = close.average(self.ma_extra_long)

        self.avg_volume_30 = volume.average(30)
        self.avg_volume_60 = volume.average(60)

        self.kdj_k, self.kdj_d = self._calculate_kdj(close)
        self.kdj_j = 3.0 * self.kdj_k - 2.0 * self.kdj_d

        self.macd_dif, self.macd_dea, self.macd_histogram = self._calculate_macd(close)

    def _calculate_kdj(self, close: pd.DataFrame):
        fastk_period = self.kdj_params["fastk_period"]
        signal       = self.kdj_params["signal"]

        slowk, slowd = data.indicator(
            "STOCH",
            adjust_price=False,
            fastk_period=fastk_period,
            slowk_period=signal,
            slowk_matype=0,
            slowd_period=signal,
            slowd_matype=0,
        )

        k = slowk.reindex(index=close.index, columns=close.columns)
        d = slowd.reindex(index=close.index, columns=close.columns)
        return k, d

    def _calculate_macd(self, close: pd.DataFrame):
        dif, dea, histogram = data.indicator(
            "MACD",
            adjust_price=False,
            fastperiod=self.macd_params["fastperiod"],
            slowperiod=self.macd_params["slowperiod"],
            signalperiod=self.macd_params["signalperiod"],
        )

        dif       = dif.reindex(index=close.index, columns=close.columns)
        dea       = dea.reindex(index=close.index, columns=close.columns)
        histogram = histogram.reindex(index=close.index, columns=close.columns)

        return dif, dea, histogram

    # =========================================================================
    # 買進條件
    # =========================================================================

    def _build_buy_condition(self) -> pd.DataFrame:
        close  = self.market_data["close"]
        open_  = self.market_data["open"]
        volume = self.market_data["volume"]

        # 前提條件：25MA 向上傾斜 & 5MA > 60MA
        prerequisites = (
            self.ma25.rise(self.ma25_slope_lookback)
            & (self.ma5 > self.ma60)
        )

        # 突破式進場
        breakout_entry = (close > self.ma25).is_entry()

        # 回踩式進場：接近 MA25（±3%）+ 縮量 + 星線
        close_to_ma25  = (close - self.ma25).abs() / self.ma25.replace(0, np.nan) < self.pullback_ma25_tolerance
        low_volume     = volume < (self.avg_volume_60 * self.pullback_volume_ratio)
        doji_candle    = (close - open_).abs() / close.replace(0, np.nan) < self.doji_threshold
        pullback_entry = close_to_ma25 & low_volume & doji_candle

        entry_signal = breakout_entry | pullback_entry

        # Filter 1：5MA 不急速下降
        ma5_not_plunging = (
            self.ma5 >= self.ma5.shift(self.ma5_slope_lookback) * self.ma5_slope_tolerance
        )

        # Filter 2：KDJ 金叉（窗口內至少發生一次）
        kdj_golden_cross = (
            (self.kdj_k > self.kdj_d)
            .is_entry()
            .sustain(self.kdj_golden_cross_window, nsatisfy=1)
        )

        # Filter 3：MACD Histogram 上升（動能轉強）
        macd_momentum_positive = self.macd_histogram.rise(1)

        # 流動性過濾
        sufficient_liquidity = self.avg_volume_30 > self.min_avg_volume_30

        buy_condition = (
            prerequisites
            & entry_signal
            & ma5_not_plunging
            & kdj_golden_cross
            & macd_momentum_positive
            & sufficient_liquidity
        )

        return buy_condition

    # =========================================================================
    # 賣出條件
    # =========================================================================

    def _build_sell_condition(self) -> pd.DataFrame:
        close  = self.market_data["close"]
        open_  = self.market_data["open"]
        high   = self.market_data["high"]
        volume = self.market_data["volume"]

        # 退出一：乖離過大
        exit_deviation = (
            (close - self.ma25) / self.ma25.replace(0, np.nan) > self.deviation_exit_threshold
        )

        # 退出二：KDJ 超買 / J 線從高位轉折
        j_overbought = self.kdj_j > self.kdj_j_overbought
        if self.kdj_j_single_day_turn:
            j_turning_down = (
                self.kdj_j.fall(1)
                & (self.kdj_j.shift(1) > self.kdj_j_turn_threshold)
            )
        else:
            j_turning_down = (
                self.kdj_j.fall(1).sustain(2)
                & (self.kdj_j.shift(2) > self.kdj_j_turn_threshold)
            )
        exit_kdj = j_overbought | j_turning_down

        # 退出三：接近前高阻力位（但尚未突破）
        prior_resistance = high.shift(1).rolling(
            self.resistance_lookback, min_periods=self.resistance_lookback // 2
        ).max()
        _resist = prior_resistance.replace(0, np.nan)
        exit_resistance = (
            (close / _resist) >= (1 - self.resistance_tolerance)
        ) & (close <= _resist)

        # 退出四：衝高失敗（近 10 日高點低於前 10 日高點，且收黑）
        recent_high      = high.rolling(10, min_periods=5).max()
        prior_high       = high.shift(10).rolling(10, min_periods=5).max()
        exit_failed_high = (recent_high < prior_high) & (close < open_)

        # 退出五：5MA 跌破 60MA 或 120MA（穿越事件）
        exit_ma_cross = (
            (self.ma5 < self.ma60).is_entry()
            | (self.ma5 < self.ma120).is_entry()
        )

        # 退出六：技術停損（高量跌破 MA25）
        exit_tech_stop = (
            (close < self.ma25)
            & (volume > self.avg_volume_60 * self.tech_stop_volume_ratio)
        )

        # 退出七：時間停損（預設停用）
        if self.time_stop_days > 0:
            exit_time_stop = ~close.rise(self.time_stop_days)
        else:
            exit_time_stop = pd.DataFrame(False, index=close.index, columns=close.columns)

        sell_condition = (
            exit_deviation
            | exit_kdj
            | exit_resistance
            | exit_failed_high
            | exit_ma_cross
            | exit_tech_stop
            | exit_time_stop
        )

        return sell_condition

    # =========================================================================
    # 執行策略回測
    # =========================================================================

    def run_strategy(
        self,
        start_date: str = "2020-01-01",
        fee_ratio: float = 0.001425,
        tax_ratio: float = 0.003,
        sim_fast_mode: bool = None,
        sim_resample: str = "D",
    ):
        run_t0 = perf_counter()

        base_position = self.base_position.loc[start_date:]
        if len(base_position.index) == 0:
            raise ValueError(f"start_date={start_date} 之後無可用交易日，請確認日期範圍。")

        selected_mask  = base_position.astype(bool)
        selected_count = selected_mask.sum(axis=1)
        final_position = selected_mask.div(
            selected_count.replace(0, np.nan), axis=0
        ).fillna(0.0)
        self._record_profile("build_final_position_sec", run_t0)

        self.report = sim(
            position=final_position,
            resample=sim_resample,
            upload=False,
            market=AdjustTWMarketInfo(),
            fee_ratio=fee_ratio,
            tax_ratio=tax_ratio,
            fast_mode=(
                self.sim_fast_mode_default if sim_fast_mode is None else bool(sim_fast_mode)
            ),
            position_limit=1.0,
        )
        self._record_profile("run_strategy_total_sec", run_t0)

        return self.report

    def get_report(self):
        return self.report if self.report is not None else "report物件為空，請先運行策略"


if __name__ == "__main__":
    strategy = _2560TWStrategy(
        ma_short=5,
        ma_mid=25,
        ma_long=60,
        pullback_ma25_tolerance=0.03,
        pullback_volume_ratio=0.8,
        doji_threshold=0.01,
        deviation_exit_threshold=0.15,
        kdj_j_overbought=100.0,
        kdj_j_single_day_turn=True,
        kdj_j_turn_threshold=80.0,
        resistance_lookback=60,
        resistance_tolerance=0.03,
        stop_loss_pct=0.07,
        time_stop_days=0,
        tech_stop_volume_ratio=1.5,
    )

    report = strategy.run_strategy(start_date="2020-01-01")