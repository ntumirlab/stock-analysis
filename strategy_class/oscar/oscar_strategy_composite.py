"""
Oscar 四大指標策略 (Oscar Four Key Indicators Strategy)

根據 Oscar 的交易邏輯，結合四個核心指標:
1. SAR (Stop and Reverse) - 拋物線指標
2. MACD (Moving Average Convergence Divergence) - 指數平滑異同移動平均線
3. Volume (VOL) - 買賣聲量
4. Three Major Institutional Investors - 三大法人買賣超

策略核心邏輯:
- 買進: SAR翻多訊號與MACD黃金交叉在可接受時間窗內 + 適當成交量 + 法人買超支持
- 賣出: SAR 反轉到上方 或 MACD 死亡交叉
- 選股與權重: 所有符合條件的股票皆納入持股，並採等權重配置
- 特別注意: 排除不斷創新高的股票，避免追高

股票範圍:
全台股市場 (TSE + OTC)
"""

from __future__ import annotations

import os
from finlab import data
from finlab.markets.tw import TWMarket
from finlab.backtest import sim
import pandas as pd
import numpy as np
from .oscar_strategy_composite_params import OscarCompositeParams
from utils.config_loader import ConfigLoader

class OscarCompositeStrategy:
    """Oscar composite strategy with lazy run-time signal computation."""

    def __init__(
        self,
        config_path="config.yaml",
        market_data=None,
    ):
        # 載入配置文件並初始化市場數據緩存。實際的信號計算將在 run_strategy 時根據配置動態執行，以支持參數優化和測試。
        self.config_loader = ConfigLoader(config_path)
        self._market_data_cache = market_data
        self._active_market_data = None

        self.report = None
        self.trade_price = None
        self.sar_values = None
        self.macd_dif = None
        self.macd_dea = None
        self.macd_histogram = None
        self.institutional_condition = {
            "foreign_buy": None,
            "trust_buy": None,
            "dealer_buy": None,
        }
        self.signal_power = {
            "sar": None,
            "macd": None,
            "volume": None,
            "institutional": None,
            "composite": None,
        }

        self._buy_signal = None
        self._sell_signal = None
        self._base_position = None
        self._last_run_config = None

    def _load_market_data(self) -> dict:
        '''
        載入市場數據:
        - 為了效率，市場數據將在第一次請求時載入並緩存在實例中，後續請求將直接使用緩存數據。
        '''
        if self._market_data_cache is None:
            with data.universe(market="TSE_OTC"):
                self._market_data_cache = {
                    "open": data.get("price:開盤價"),
                    "close": data.get("price:收盤價"),
                    "high": data.get("price:最高價"),
                    "low": data.get("price:最低價"),
                    "volume": data.get("price:成交股數"),
                    "foreign_net_buy_shares":
                        data.get("institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)")
                        + data.get("institutional_investors_trading_summary:外資自營商買賣超股數"),
                    "investment_trust_net_buy_shares":
                        data.get("institutional_investors_trading_summary:投信買賣超股數"),
                    "dealer_self_net_buy_shares":
                        data.get("institutional_investors_trading_summary:自營商買賣超股數(自行買賣)")
                        + data.get("institutional_investors_trading_summary:自營商買賣超股數(避險)"),
                }
        return self._market_data_cache

    def _load_params_from_config(self) -> OscarCompositeParams:
        '''
        從配置文件中載入策略參數，並構建 OscarCompositeParams 實例。
        '''
        try:
            oscar_root = self.config_loader.config.get("oscar", {})
            general_cfg = oscar_root.get("general", {}) or {}
            composite_cfg = oscar_root.get("composite", {}) or {}
            return OscarCompositeParams(**{**general_cfg, **composite_cfg})
        except Exception as e:
            raise ValueError(f"Error loading Oscar configuration: {e}")

    def _estimate_required_warmup_bars(self, params: OscarCompositeParams) -> int:
        # Keep enough history for the 120-day breakout filter, 30-day volume mean,
        # MACD stabilization, and bounded-history lookbacks before start_date.
        macd_warmup = int(params.macd_params.slowperiod or 26) + int(params.macd_params.signalperiod or 9) + 10
        bounded_history = max(
            int(params.sar_history_lookback or 1),
            int(params.macd_history_lookback or 1),
        )
        return max(260, 30, macd_warmup, bounded_history)

    def _truncate_market_data_for_start_date(
        self,
        market_data: dict,
        params: OscarCompositeParams,
        start_date: str,
    ) -> dict:
        # Trim early history to reduce indicator work while preserving enough warmup bars.
        close = market_data["close"]
        if close.empty:
            return market_data

        start_ts = pd.Timestamp(start_date)
        first_live_pos = int(close.index.searchsorted(start_ts))
        if first_live_pos <= 0:
            return market_data

        warmup_bars = self._estimate_required_warmup_bars(params)
        slice_pos = max(0, first_live_pos - warmup_bars)
        if slice_pos == 0:
            return market_data

        truncated_index = close.index[slice_pos:]
        return {
            key: frame.reindex(truncated_index)
            for key, frame in market_data.items()
        }

    def _quantize_signal(self, signal_df: pd.DataFrame, bins):
        '''
        將連續信號量化為離散分位數等級，以減少噪音影響並強調相對強弱。

        Args:
            signal_df: 訊號 DataFrame [0, 1]
            bins: 分位數數量
        '''
        if bins is None:
            return signal_df
        bins_int = max(2, int(bins))
        ranks = signal_df.rank(axis=1, pct=True)
        bucket = np.ceil((ranks * bins_int).clip(upper=bins_int)).astype(float)
        quantized = (bucket - 1.0) / float(bins_int - 1)
        return quantized.where(signal_df.notna())

    def _build_decay(self, signal_df: pd.DataFrame, lag_max: int, alpha: float) -> pd.DataFrame:
        '''
        建立衰減信號: 根據目標 DataFrame 中的訊號，對過去 lag_max 期間內的信號進行指數衰減加權求和。

        Args:
            signal_df: 訊號 DataFrame [0, 1]
            lag_max: 最大回溯期間
            alpha: 衰減速率 (alpha > 1, 越大衰減越快)
        
        Returns:
            衰減信號 DataFrame [0, 1]
        '''
        alpha_safe = max(1.0001, float(alpha))
        arr = signal_df.fillna(0.0).values.astype(np.float64)
        t_len, n_len = arr.shape
        frame_cls = signal_df.__class__

        weighted_sum = np.zeros((t_len, n_len), dtype=np.float64)
        weight_total = 0.0

        # Weighted moving average over current and lagged values: sum(signal[t-k] * alpha^-k).
        for k in range(0, lag_max + 1):
            weight = float(alpha_safe ** (-k))
            if k == 0:
                shifted = arr
            else:
                shifted = np.empty_like(arr)
                shifted[:k] = 0.0
                shifted[k:] = arr[:-k]
            weighted_sum += shifted * weight
            weight_total += weight

        if weight_total > 0:
            weighted_sum /= weight_total
        np.clip(weighted_sum, 0.0, 1.0, out=weighted_sum)

        return frame_cls(weighted_sum, index=signal_df.index, columns=signal_df.columns)

    @staticmethod
    def _safe_positive(value: float | None, fallback: float, minimum: float = 1e-9) -> float:
        if value is None:
            return max(minimum, float(fallback))
        return max(minimum, float(value))

    @staticmethod
    def _to_unit_mix_weight(raw_weight: float | None, default_raw: float) -> float:
        # Map any non-negative raw weight to [0, 1) without hard clipping.
        w = max(0.0, float(default_raw if raw_weight is None else raw_weight))
        return w / (1.0 + w)

    @staticmethod
    def _stable_sigmoid(z_df: pd.DataFrame) -> pd.DataFrame:
        # Numerically stable sigmoid to avoid overflow for large |z|.
        frame_cls = z_df.__class__
        z = z_df.to_numpy(dtype=np.float64, copy=False)
        out = np.empty_like(z)
        non_negative = z >= 0
        out[non_negative] = 1.0 / (1.0 + np.exp(-z[non_negative]))
        exp_z = np.exp(z[~non_negative])
        out[~non_negative] = exp_z / (1.0 + exp_z)
        return frame_cls(out, index=z_df.index, columns=z_df.columns)

    @staticmethod
    def _sigmoid_decreasing(distance_norm: pd.DataFrame, slope: float) -> pd.DataFrame:
        # near=1 when distance is tiny; near drops smoothly as distance grows.
        slope_safe = max(1e-6, float(slope))
        z = -slope_safe * (distance_norm - 1.0)
        return OscarCompositeStrategy._stable_sigmoid(z)

    @staticmethod
    def _sigmoid_increasing(distance_norm: pd.DataFrame, slope: float) -> pd.DataFrame:
        slope_safe = max(1e-6, float(slope))
        z = slope_safe * (distance_norm - 1.0)
        return OscarCompositeStrategy._stable_sigmoid(z)

    def _build_exponential_complement_history(
        self,
        near_signal: pd.DataFrame,
        lookback: int,
        alpha: float,
    ) -> pd.DataFrame:
        # h_t = 1 - Π(1 - w_i * near_{t-i}), bounded in [0, 1] by construction.
        lookback_safe = max(1, int(lookback))
        alpha_safe = max(1.0001, float(alpha))
        frame_cls = near_signal.__class__

        raw_weights = np.array([alpha_safe ** (-k) for k in range(lookback_safe)], dtype=np.float64)
        weight_sum = float(raw_weights.sum())
        norm_weights = raw_weights / weight_sum if weight_sum > 0 else np.ones_like(raw_weights) / lookback_safe

        near = near_signal.fillna(0.0).astype(float)
        product_term = frame_cls(1.0, index=near.index, columns=near.columns)
        for k, weight in enumerate(norm_weights):
            shifted = near.shift(k).fillna(0.0)
            product_term = product_term * (1.0 - shifted * float(weight))
        result = 1.0 - product_term
        return frame_cls(result.to_numpy(dtype=np.float64), index=result.index, columns=result.columns)

    @staticmethod
    def _bounded_or_blend(event_term: pd.DataFrame, history_term: pd.DataFrame) -> pd.DataFrame:
        # Bounded OR: 1 - (1-e)(1-h), naturally in [0, 1].
        frame_cls = event_term.__class__
        result = 1.0 - ((1.0 - event_term) * (1.0 - history_term))
        return frame_cls(result.to_numpy(dtype=np.float64), index=result.index, columns=result.columns)

    def _calculate_sar_condition(self, params: OscarCompositeParams) -> None:
        '''
        SAR 信號：
        - 主要依據 SAR 指標與收盤價的相對位置判斷多空翻轉訊號。
        - 進一步對翻轉訊號進行時間衰減處理，並排除持續創新高的股票以降低追高風險。
        '''
        close = self._active_market_data["close"]
        sar = data.indicator(
            "SAR",
            acceleration=params.sar_params.acceleration,
            maximum=params.sar_params.maximum,
            adjust_price=False,
        ).reindex(index=close.index, columns=close.columns)
        self.sar_values = sar

        sar_below_price = sar < close
        sar_below_price_prev = sar_below_price.shift(1).fillna(sar_below_price)
        sar_flip_bullish = sar_below_price & (~sar_below_price_prev)

        sar_gap_ratio = (close - sar).abs() / close.replace(0, np.nan).abs().replace(0, np.nan)
        sar_gap_ratio = sar_gap_ratio.fillna(0.0)
        sar_scale = self._safe_positive(params.sar_near_distance_scale, fallback=0.02)
        sar_distance_norm = sar_gap_ratio / sar_scale
        sar_near = self._sigmoid_decreasing(
            sar_distance_norm,
            slope=self._safe_positive(params.sar_near_sigmoid_slope, fallback=8.0),
        )

        sar_event_scale = self._safe_positive(params.sar_event_distance_scale, fallback=sar_scale)
        sar_cross_depth_norm = ((close - sar) / close.replace(0, np.nan).abs().replace(0, np.nan)).fillna(0.0) / sar_event_scale
        sar_event_magnitude = self._sigmoid_increasing(
            sar_cross_depth_norm,
            slope=self._safe_positive(params.sar_event_sigmoid_slope, fallback=10.0),
        )
        sar_event = sar_event_magnitude * sar_flip_bullish.astype(float)

        sar_history_near = self._build_decay(
            sar_near,
            lag_max=max(1, int(params.sar_history_lookback)),
            alpha=self._safe_positive(params.sar_history_decay_alpha, fallback=1.8, minimum=1.0001),
        )

        sar_near_mix = self._to_unit_mix_weight(params.sar_proximity_weight, default_raw=0.4)
        sar_event_mix = self._to_unit_mix_weight(params.sar_event_weight, default_raw=1.0)
        sar_history_term = sar_history_near * sar_near_mix
        sar_event_term = sar_event * sar_event_mix
        sar_signal_power = self._bounded_or_blend(sar_event_term, sar_history_term)

        # 排除不斷創新高的股票，避免追高風險
        high_120 = close.rolling(120).max()
        is_new_high = close >= high_120
        new_high_ratio_120 = is_new_high.rolling(120).mean()
        constantly_new_high = new_high_ratio_120 > params.new_high_ratio_120

        sar_signal_power = sar_signal_power * (~constantly_new_high).astype(float)

        sar_signal_power = self._quantize_signal(
            sar_signal_power,
            params.signal_quantile_bins.sar
        )
        self.signal_power["sar"] = sar_signal_power.fillna(0.0)

    def _calculate_macd_condition(self, params: OscarCompositeParams) -> None:
        '''
        MACD 信號：
        - 主要依據 MACD 的 DIF 與 DEA 的黃金交叉判斷多頭翻轉訊號。
        - 進一步對翻轉訊號進行時間衰減處理，以強調近期訊號並減弱過去訊號的影響。
        '''
        close = self._active_market_data["close"]
        dif, dea, histogram = data.indicator(
            "MACD",
            fastperiod=params.macd_params.fastperiod,
            slowperiod=params.macd_params.slowperiod,
            signalperiod=params.macd_params.signalperiod,
            adjust_price=False,
        )

        dif = dif.reindex(index=close.index, columns=close.columns)
        dea = dea.reindex(index=close.index, columns=close.columns)
        histogram = histogram.reindex(index=close.index, columns=close.columns)

        self.macd_dif = dif
        self.macd_dea = dea
        self.macd_histogram = histogram

        macd_cross_bullish = (dif > dea) & (dif.shift(1) <= dea.shift(1))

        macd_gap = (dif - dea).abs()
        macd_scale = self._safe_positive(params.macd_near_distance_scale, fallback=1.0)
        macd_distance_norm = macd_gap / macd_scale
        macd_near = self._sigmoid_decreasing(
            macd_distance_norm,
            slope=self._safe_positive(params.macd_near_sigmoid_slope, fallback=8.0),
        )

        macd_event_scale = self._safe_positive(params.macd_event_distance_scale, fallback=macd_scale)
        macd_cross_depth_norm = ((dif - dea).clip(lower=0.0)) / macd_event_scale
        macd_event_magnitude = self._sigmoid_increasing(
            macd_cross_depth_norm,
            slope=self._safe_positive(params.macd_event_sigmoid_slope, fallback=10.0),
        )
        macd_event = macd_event_magnitude * macd_cross_bullish.astype(float)

        macd_history_near = self._build_decay(
            macd_near,
            lag_max=max(1, int(params.macd_history_lookback)),
            alpha=self._safe_positive(params.macd_history_decay_alpha, fallback=1.8, minimum=1.0001),
        )

        macd_near_mix = self._to_unit_mix_weight(params.macd_proximity_weight, default_raw=0.4)
        macd_event_mix = self._to_unit_mix_weight(params.macd_event_weight, default_raw=1.0)
        macd_history_term = macd_history_near * macd_near_mix
        macd_event_term = macd_event * macd_event_mix
        macd_signal_power = self._bounded_or_blend(macd_event_term, macd_history_term)

        macd_signal_power = self._quantize_signal(
            macd_signal_power,
            params.signal_quantile_bins.macd
        )
        self.signal_power["macd"] = macd_signal_power.fillna(0.0)

    def _calculate_volume_condition(self, volume, params: OscarCompositeParams) -> None:
        '''
        成交量信號:
        - 主要依據當前成交量與過去平均成交量的比值判斷是否存在成交量異常放大，並結合流動性條件過濾掉流動性不足的股票。
        - 進一步對成交量放大程度進行分位數量化，以減少噪音影響並強調相對強弱。
        '''
        avg_volume_30 = volume.rolling(30).mean()
        sufficient_liquidity = avg_volume_30 > params.min_avg_volume_30

        volume_ratio = (volume / avg_volume_30.replace(0, np.nan)).fillna(0.0)
        ratio_floor = max(params.volume_above_avg_ratio, 1e-6)
        ratio_ceiling = max(ratio_floor + 1e-6, params.max_volume_spike_ratio)
        volume_signal_power = (
            (volume_ratio - ratio_floor) / (ratio_ceiling - ratio_floor)
        ).clip(lower=0.0, upper=1.0)
        volume_signal_power = volume_signal_power * sufficient_liquidity.astype(float)
        volume_signal_power = self._quantize_signal(
            volume_signal_power,
            params.signal_quantile_bins.volume
        )
        self.signal_power["volume"] = volume_signal_power.fillna(0.0)

    def _calculate_institutional_condition(
        self,
        foreign_net_buy,
        trust_net_buy,
        dealer_net_buy,
        params: OscarCompositeParams,
    ) -> None:
        '''
        法人買賣超信號:
        - 主要依據三大法人買賣超的正負判斷買賣方向，並計算買超比例作為信號強度。
        - 進一步對買超比例進行分位數量化，以減少噪音影響並強調相對強弱。
        '''
        foreign_buy = foreign_net_buy > 0
        trust_buy = trust_net_buy > 0
        dealer_buy = dealer_net_buy > 0

        self.institutional_condition = {
            "foreign_buy": foreign_buy,
            "trust_buy": trust_buy,
            "dealer_buy": dealer_buy,
        }

        buy_count = (
            foreign_buy.astype(int) + trust_buy.astype(int) + dealer_buy.astype(int)
        )
        institutional_signal_power = (buy_count / 3.0).clip(lower=0.0, upper=1.0)
        institutional_signal_power = self._quantize_signal(
            institutional_signal_power,
            params.signal_quantile_bins.institutional
        )
        self.signal_power["institutional"] = institutional_signal_power.fillna(0.0)


    def _calculate_buy_score(self, params: OscarCompositeParams):
        '''
        買進分數計算:
        - 根據 SAR、MACD、成交量和法人買超四個核心指標的信號強度，結合配置的權重計算綜合買進分數。
        - 分數計算方式為加權平均，並根據配置的權重比例對各指標進行調整，以反映其相對重要性。
        - 最終的綜合買進分數將用於判斷是否達到買進門檻，並作為選股和權重配置的依據。

        Returns:
            composite_score: 綜合買進分數 DataFrame [0, 1]
        '''
        weight_sar = params.signal_weights.sar
        weight_macd = params.signal_weights.macd
        weight_volume = params.signal_weights.volume
        weight_inst = params.signal_weights.institutional
        weight_total = weight_sar + weight_macd + weight_volume + weight_inst
        if weight_total <= 0:
            raise ValueError("Total signal weight must be greater than zero.")

        composite = (
            self.signal_power["sar"] * weight_sar
            + self.signal_power["macd"] * weight_macd
            + self.signal_power["volume"] * weight_volume
            + self.signal_power["institutional"] * weight_inst
        ) / weight_total
        self.signal_power["composite"] = composite.fillna(0.0)

    def _build_buy_condition(self, params: OscarCompositeParams):
        '''
        買進條件:
        - 綜合買進分數達到配置的買進門檻。
        '''
        if self.signal_power["composite"] is None:
            self._calculate_buy_score(params)
        return self.signal_power["composite"] >= params.buy_score_threshold

    def _build_sell_condition(self, params: OscarCompositeParams):
        '''
        賣出條件:
        - 綜合買進分數達到配置的賣出門檻。
        '''
        if self.signal_power["composite"] is None:
            self._calculate_buy_score(params)
        return self.signal_power["composite"] <= params.sell_score_threshold

    def _compute_signals(self, params: OscarCompositeParams, market_data_override=None):
        '''
        計算買賣訊號:
        - 根據提供的參數計算 SAR、MACD、成交量和法人買超的信號強度，並進一步計算綜合買進分數。
        - 最後根據綜合買進分數建立買進和賣出條件，並生成基礎持倉信號 DataFrame。
        '''
        base_data = market_data_override if market_data_override is not None else self._load_market_data()
        market_data = base_data

        # 輔助 K 線圖視覺化與報告分析用，實際交易價格為下一根 K 線的開盤價
        self.trade_price = market_data["open"]

        # Reset per-run computed scores to avoid stale values leaking between runs.
        self.signal_power = {
            "sar": None,
            "macd": None,
            "volume": None,
            "institutional": None,
            "composite": None,
        }

        self._active_market_data = market_data

        self._calculate_sar_condition(params)
        self._calculate_macd_condition(params)
        self._calculate_volume_condition(
            market_data["volume"],
            params,
        )
        self._calculate_institutional_condition(
            market_data["foreign_net_buy_shares"],
            market_data["investment_trust_net_buy_shares"],
            market_data["dealer_self_net_buy_shares"],
            params,
        )

        self._buy_signal = self._build_buy_condition(params)
        self._sell_signal = self._build_sell_condition(params)
        self._base_position = self._buy_signal.hold_until(self._sell_signal)
        self._last_run_config = params

        return self._base_position

    @property
    def buy_signal(self):
        if self._buy_signal is None:
            params = self._last_run_config or self._load_params_from_config()
            self._compute_signals(params)
        return self._buy_signal

    @property
    def sell_signal(self):
        if self._sell_signal is None:
            params = self._last_run_config or self._load_params_from_config()
            self._compute_signals(params)
        return self._sell_signal

    @property
    def base_position(self):
        if self._base_position is None:
            params = self._last_run_config or self._load_params_from_config()
            self._compute_signals(params)
        return self._base_position

    def run_strategy(
        self,
        params: OscarCompositeParams | None = None,
        start_date="2020-01-01",
        fee_ratio=0.001425,
        tax_ratio=0.003,
        sim_resample="D",
    ):
        # 如果參數在執行時未提供，則從配置文件中加載參數並計算信號。
        if params is None:
            params = self._load_params_from_config()

        full_market_data = self._load_market_data()
        truncated_market_data = self._truncate_market_data_for_start_date(
            full_market_data,
            params,
            start_date,
        )

        base_position = self._compute_signals(
            params,
            market_data_override=truncated_market_data,
        )
        base_position = self._base_position.loc[start_date:]
        if len(base_position.index) == 0:
            raise ValueError("No trading days available after start_date.")

        # 選股與權重: 所有符合條件的股票皆納入持股，並採等權重配置。
        selected_mask = base_position.astype(bool)
        selected_count = selected_mask.sum(axis=1)
        final_position = selected_mask.div(
            selected_count.replace(0, np.nan),
            axis=0,
        ).fillna(0.0)

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
    strategy = OscarCompositeStrategy()
    report = strategy.run_strategy(start_date="2020-01-01")
    os.makedirs("./assets/OscarCompositeStrategy", exist_ok=True)
    report.display(save_report_path="./assets/OscarCompositeStrategy/oscar_composite_report.html")
