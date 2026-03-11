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

性能優化 (Performance Optimizations):
🚀 智能緩存: 每個指標只計算一次，避免重複計算 (2x speedup)
🚀 支持預載數據: 可傳入預先載入的market_data，避免重複載入
🚀 並行優化友好: 適合大規模參數優化和多股票回測
"""

from finlab import data
from finlab.markets.tw import TWMarket
from finlab.backtest import sim
import pandas as pd
import numpy as np
from time import perf_counter
from utils.config_loader import ConfigLoader


class AdjustTWMarketInfo(TWMarket):
    """自訂市場資訊類別，用於調整交易價格為開盤價"""

    def get_trading_price(self, name, adj=True):
        # 訊號使用當日收盤後資料，實際成交需對齊至下一交易日開盤。
        return self.get_price("open", adj=adj).shift(1)


class OscarCompositeStrategy:
    """
    Oscar 四大指標策略

    Attributes:
        report: 回測報告物件
        position: 持倉訊號
        buy_signal: 買入訊號
        sell_signal: 賣出訊號
    """

    _MARKET_DATA_CACHE = None
    _INDICATOR_CACHE = {}
    _MAX_INDICATOR_CACHE_SIZE = 32

    def __init__(
        self,
        sar_signal_lag_min=0,
        sar_signal_lag_max=2,
        config_path="config.yaml",
        sar_params=None,
        macd_params=None,
        market_data=None,
        volume_above_avg_ratio=None,
        new_high_ratio_120=None,
        sar_max_dots=None,
        sar_reject_dots=None,
        macd_signal_lag_min=None,
        macd_signal_lag_max=None,
        min_avg_volume_30=None,
        max_volume_spike_ratio=None,
        signal_quantile_bins=None,
        signal_weights=None,
        buy_score_threshold=None,
        sell_score_threshold=None,
        preprocess_start_date=None,
        preprocess_lookback_days=None,
        enable_profiling=False,
        prefilter_enabled=None,
        prefilter_min_avg_volume_30=None,
        prefilter_min_history_days=None,
        sim_fast_mode_default=None,
    ):
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
            preprocess_start_date: 指標預處理起始日期（可選，預設沿用全資料）
            preprocess_lookback_days: 預處理回看天數（可選，預設使用 252）
            enable_profiling: 是否記錄策略內部各階段耗時
            prefilter_enabled: 是否在預處理階段先做股票池預篩
            prefilter_min_avg_volume_30: 預篩的 30 日均量門檻
            prefilter_min_history_days: 預篩所需最少有效交易日數
            sim_fast_mode_default: run_strategy 預設 fast_mode（可覆寫）
        """
        init_t0 = perf_counter()

        # 載入配置
        self.config_loader = ConfigLoader(config_path)
        oscar_root = self.config_loader.config.get("oscar", {})
        if isinstance(oscar_root, dict) and any(
            k in oscar_root for k in ("general", "andor", "composite")
        ):
            general_cfg = oscar_root.get("general", {}) or {}
            mode_cfg = oscar_root.get("composite", {}) or {}
            oscar_config = {**general_cfg, **mode_cfg}
        else:
            # Backward compatibility for legacy flat `oscar` config.
            oscar_config = oscar_root

        self.enable_profiling = bool(enable_profiling)
        self.profile_stats = {}

        # 新參數：SAR 與 MACD 訊號允許時間窗
        # 向後相容：若仍傳入舊參數 sar_max_dots，映射到 lag_max。
        if sar_max_dots is not None and sar_signal_lag_max == 2:
            sar_signal_lag_max = int(sar_max_dots)

        self.sar_signal_lag_min = max(0, int(sar_signal_lag_min))
        self.sar_signal_lag_max = max(self.sar_signal_lag_min, int(sar_signal_lag_max))
        self.macd_signal_lag_min = max(0, int(macd_signal_lag_min or 0))
        self.macd_signal_lag_max = max(
            self.macd_signal_lag_min, int(macd_signal_lag_max or 0)
        )

        # 保留舊欄位名稱供外部程式存取（deprecated, no longer used in logic）
        self.sar_max_dots = sar_max_dots
        self.sar_reject_dots = sar_reject_dots

        # SAR 和 MACD 指標參數
        self.sar_params = sar_params or {"acceleration": 0.02, "maximum": 0.2}
        self.macd_params = macd_params or {
            "fastperiod": 12,
            "slowperiod": 26,
            "signalperiod": 9,
        }

        # 成交量大於平均的比例，以及 120 日創新高比例，從設定檔讀取，可覆寫預設值
        # 預設值保留現有行為：volume_above_avg_ratio=0.25, new_high_ratio_120=0.3
        default_volume_ratio = float(oscar_config.get("volume_above_avg_ratio", 0.25))
        default_new_high_ratio = float(oscar_config.get("new_high_ratio_120", 0.3))
        self.default_max_stocks = max(1, int(oscar_config.get("max_stocks", 4)))
        self.min_avg_volume_30 = float(oscar_config.get("min_avg_volume_30", 1_000_000))
        if min_avg_volume_30 is not None:
            self.min_avg_volume_30 = float(min_avg_volume_30)

        self.max_volume_spike_ratio = float(
            oscar_config.get("max_volume_spike_ratio", 10.0)
        )
        if max_volume_spike_ratio is not None:
            self.max_volume_spike_ratio = float(max_volume_spike_ratio)

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

        # Pure composite strategy: composite scoring is intentionally enabled.
        self.use_composite_scoring = True

        self.signal_quantile_bins = oscar_config.get("signal_quantile_bins")
        if signal_quantile_bins is not None:
            self.signal_quantile_bins = signal_quantile_bins

        default_signal_weights = oscar_config.get(
            "signal_weights",
            {
                "sar": 0.35,
                "macd": 0.35,
                "volume": 0.15,
                "institutional": 0.15,
            },
        )
        self.signal_weights = signal_weights or default_signal_weights
        self.buy_score_threshold = float(oscar_config.get("buy_score_threshold", 0.60))
        self.sell_score_threshold = float(
            oscar_config.get("sell_score_threshold", 0.40)
        )
        if buy_score_threshold is not None:
            self.buy_score_threshold = float(buy_score_threshold)
        if sell_score_threshold is not None:
            self.sell_score_threshold = float(sell_score_threshold)

        config_preprocess_start = oscar_config.get("preprocess_start_date")
        self.preprocess_start_date = (
            preprocess_start_date
            if preprocess_start_date is not None
            else config_preprocess_start
        )

        config_lookback = oscar_config.get("preprocess_lookback_days", 252)
        if preprocess_lookback_days is not None:
            self.preprocess_lookback_days = max(0, int(preprocess_lookback_days))
        else:
            self.preprocess_lookback_days = max(0, int(config_lookback))

        config_prefilter_enabled = oscar_config.get("prefilter_enabled", False)
        self.prefilter_enabled = (
            bool(config_prefilter_enabled)
            if prefilter_enabled is None
            else bool(prefilter_enabled)
        )

        config_prefilter_liq = oscar_config.get(
            "prefilter_min_avg_volume_30", self.min_avg_volume_30
        )
        self.prefilter_min_avg_volume_30 = float(
            config_prefilter_liq
            if prefilter_min_avg_volume_30 is None
            else prefilter_min_avg_volume_30
        )

        config_prefilter_history_days = oscar_config.get(
            "prefilter_min_history_days", 120
        )
        self.prefilter_min_history_days = max(
            20,
            int(
                config_prefilter_history_days
                if prefilter_min_history_days is None
                else prefilter_min_history_days
            ),
        )

        config_sim_fast_mode = oscar_config.get("sim_fast_mode", False)
        self.sim_fast_mode_default = (
            bool(config_sim_fast_mode)
            if sim_fast_mode_default is None
            else bool(sim_fast_mode_default)
        )

        # 回測報告
        self.report = None

        # 載入市場數據（若未提供則載入）
        load_t0 = perf_counter()
        market_data = market_data if market_data is not None else self._load_data()
        self._record_profile("load_market_data_sec", load_t0)

        scope_t0 = perf_counter()
        market_data = self._scope_market_data_for_preprocessing(market_data)
        self._record_profile("scope_market_data_sec", scope_t0)

        prefilter_t0 = perf_counter()
        market_data = self._prefilter_market_universe(market_data)
        self._record_profile("prefilter_market_universe_sec", prefilter_t0)

        # 儲存市場數據供視覺化使用
        self.market_data = market_data

        # 儲存指標數據供視覺化使用
        self.sar_values = None
        self.macd_dif = None
        self.macd_dea = None
        self.macd_histogram = None
        self.signal_power = {
            "sar": None,
            "macd": None,
            "volume": None,
            "institutional": None,
            "composite": None,
        }

        # 儲存法人買賣超數據供視覺化使用
        self.institutional_condition = {
            "foreign_buy": None,
            "trust_buy": None,
            "dealer_buy": None,
        }

        # 儲存交易價格（開盤價）供視覺化使用
        self.trade_price = market_data["open"]

        # 🚀 優化：預先計算所有指標（只計算一次，避免重複）
        indicator_t0 = perf_counter()
        self._cached_sar_buy, self._cached_sar_sell = self._calculate_sar_condition(
            market_data["close"]
        )
        self._cached_macd_buy, self._cached_macd_sell = self._calculate_macd_condition(
            market_data["close"]
        )
        self._cached_volume_condition = self._calculate_volume_condition(
            market_data["volume"]
        )
        self._cached_institutional_strong, self._cached_institutional_weak = (
            self._calculate_institutional_condition(
                market_data["foreign_net_buy_shares"],
                market_data["investment_trust_net_buy_shares"],
                market_data["dealer_self_net_buy_shares"],
            )
        )
        self._record_profile("calculate_indicators_sec", indicator_t0)

        # 建立買入訊號、賣出訊號（使用預先計算的結果）
        self.buy_signal = self._build_buy_condition(market_data)
        self.sell_signal = self._build_sell_condition(market_data)

        # 基礎持倉訊號（未套用持股限制）
        self.base_position = self.buy_signal.hold_until(self.sell_signal)
        self._record_profile("init_total_sec", init_t0)

    @classmethod
    def clear_runtime_cache(cls):
        """Clear in-process caches for market data and indicators."""
        cls._MARKET_DATA_CACHE = None
        cls._INDICATOR_CACHE.clear()

    def _record_profile(self, key: str, start_t: float) -> None:
        if self.enable_profiling:
            self.profile_stats[key] = round(perf_counter() - start_t, 6)

    def _scope_market_data_for_preprocessing(self, market_data):
        """Optionally narrow preprocessing date range while keeping enough lookback history."""
        if not self.preprocess_start_date:
            return market_data

        scoped = {}
        start_ts = pd.Timestamp(self.preprocess_start_date)
        scoped_start = start_ts - pd.Timedelta(days=self.preprocess_lookback_days)

        for key, df in market_data.items():
            if isinstance(df, pd.DataFrame):
                scoped_df = df.loc[df.index >= scoped_start]
                # Guard: if scoped window is too tight or empty, fallback to original data.
                scoped[key] = scoped_df if len(scoped_df.index) > 0 else df
            else:
                scoped[key] = df

        return scoped

    def _prefilter_market_universe(self, market_data):
        """Cheap symbol prefilter to reduce downstream matrix width before heavy ops."""
        if not self.prefilter_enabled:
            return market_data

        close = market_data.get("close")
        volume = market_data.get("volume")
        if close is None or volume is None:
            return market_data

        if close.empty or volume.empty:
            return market_data

        filter_start = close.index.min()
        if self.preprocess_start_date:
            filter_start = pd.Timestamp(self.preprocess_start_date)

        close_recent = close.loc[close.index >= filter_start]
        volume_recent = volume.loc[volume.index >= filter_start]

        valid_days = close_recent.notna().sum(axis=0)
        enough_history = valid_days >= self.prefilter_min_history_days

        avg_volume_30 = volume_recent.rolling(30, min_periods=20).mean()
        liquid_enough = (avg_volume_30 >= self.prefilter_min_avg_volume_30).any(axis=0)

        keep_cols = close.columns[enough_history & liquid_enough]
        if len(keep_cols) == 0:
            return market_data

        filtered = {}
        for key, df in market_data.items():
            if isinstance(df, pd.DataFrame):
                common_cols = df.columns.intersection(keep_cols)
                filtered[key] = df.loc[:, common_cols]
            else:
                filtered[key] = df

        if self.enable_profiling:
            self.profile_stats["prefilter_kept_symbols"] = int(len(keep_cols))
        return filtered

    def _get_cached_indicator(self, name, **kwargs):
        """Cache finlab indicator results in-process to avoid repeated heavy calls."""
        key = (name, tuple(sorted(kwargs.items())))
        if key not in OscarCompositeStrategy._INDICATOR_CACHE:
            if len(OscarCompositeStrategy._INDICATOR_CACHE) >= self._MAX_INDICATOR_CACHE_SIZE:
                OscarCompositeStrategy._INDICATOR_CACHE.clear()
            OscarCompositeStrategy._INDICATOR_CACHE[key] = data.indicator(
                name, **kwargs
            )
        return OscarCompositeStrategy._INDICATOR_CACHE[key]

    @staticmethod
    def load_market_data():
        """
        靜態方法：載入市場數據（供外部預先載入使用）

        Returns:
            dict: 包含所有必要數據的字典
        """
        if OscarCompositeStrategy._MARKET_DATA_CACHE is not None:
            return OscarCompositeStrategy._MARKET_DATA_CACHE

        with data.universe(market="TSE_OTC"):
            loaded = {
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

        OscarCompositeStrategy._MARKET_DATA_CACHE = loaded
        return loaded

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

    def _expand_event_window(
        self, event_df: pd.DataFrame, lag_min: int, lag_max: int
    ) -> pd.DataFrame:
        """將事件訊號展開到指定 lag 區間。"""
        lagged_signals = [
            event_df.shift(lag).fillna(False) for lag in range(lag_min, lag_max + 1)
        ]
        expanded = lagged_signals[0]
        for signal in lagged_signals[1:]:
            expanded = expanded | signal
        return expanded

    def _quantize_signal(self, signal_df: pd.DataFrame, bins) -> pd.DataFrame:
        """使用分位數分箱將訊號離散化到 [0, 1]。"""
        if bins is None:
            return signal_df

        bins_int = max(2, int(bins))
        # 使用同日橫截面排名，避免沿時間軸排名造成 lookahead。
        ranks = signal_df.rank(axis=1, pct=True)
        bucket = np.ceil((ranks * bins_int).clip(upper=bins_int)).astype(float)
        quantized = (bucket - 1.0) / float(bins_int - 1)
        return quantized.where(signal_df.notna())

    def _resolve_bins(self, signal_name: str):
        """取得指定訊號的分箱數設定。"""
        if isinstance(self.signal_quantile_bins, dict):
            return self.signal_quantile_bins.get(signal_name)
        return self.signal_quantile_bins

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
        sar = self._get_cached_indicator(
            "SAR",
            acceleration=self.sar_params["acceleration"],
            maximum=self.sar_params["maximum"],
            adjust_price=False,
        ).reindex(index=close.index, columns=close.columns)

        # 儲存 SAR 值供視覺化使用
        self.sar_values = sar

        # SAR 在價格下方表示看漲狀態
        sar_below_price = sar < close

        # SAR 翻多事件：當天由非翻多轉為翻多。
        # 注意：第一筆資料沒有「前一日」狀態，以當日值填補避免第一筆即觸發虛假翻多訊號。
        sar_below_price_prev = sar_below_price.shift(1).fillna(sar_below_price)
        sar_flip_bullish = sar_below_price & (~sar_below_price_prev)

        # SAR 獨立 lag 視窗，可與 MACD 視窗不同步。
        sar_in_alignment_window = self._expand_event_window(
            sar_flip_bullish,
            self.sar_signal_lag_min,
            self.sar_signal_lag_max,
        )

        # 檢測不斷創新高的股票 (120天內創新高次數過多)
        high_120 = close.rolling(120).max()
        is_new_high = close >= high_120
        new_high_ratio_120 = is_new_high.rolling(120).mean()
        constantly_new_high = new_high_ratio_120 > self.new_high_ratio_120

        # SAR 訊號強度（0~1）：價格高於 SAR 的距離 + 翻多事件視窗。
        sar_distance = (close - sar) / close.replace(0, np.nan)
        sar_distance = sar_distance.clip(lower=0.0, upper=1.0).fillna(0.0)
        sar_event_strength = sar_in_alignment_window.astype(float)
        sar_signal_power = (0.7 * sar_distance + 0.3 * sar_event_strength) * (
            ~constantly_new_high
        ).astype(float)
        sar_signal_power = self._quantize_signal(
            sar_signal_power, self._resolve_bins("sar")
        )
        self.signal_power["sar"] = sar_signal_power.fillna(0.0)

        # 買進條件: SAR 訊號在對齊時間窗內，且不是不斷創新高。
        sar_buy_condition = sar_in_alignment_window & (~constantly_new_high)

        # 賣出條件: SAR 翻轉到價格上方 (看跌訊號)
        sar_sell_condition = ~sar_below_price

        return sar_buy_condition, sar_sell_condition

    def _calculate_macd_condition(self, close):
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
        dif, dea, histogram = self._get_cached_indicator(
            "MACD",
            fastperiod=self.macd_params["fastperiod"],
            slowperiod=self.macd_params["slowperiod"],
            signalperiod=self.macd_params["signalperiod"],
            adjust_price=False,
        )
        dif = dif.reindex(index=close.index, columns=close.columns)
        dea = dea.reindex(index=close.index, columns=close.columns)
        histogram = histogram.reindex(index=close.index, columns=close.columns)

        # 儲存 MACD 值供視覺化使用
        self.macd_dif = dif
        self.macd_dea = dea
        self.macd_histogram = histogram

        # 黃金交叉: DIF > MACD 且前一日 DIF <= MACD (剛形成買進訊號)
        # 死亡交叉: DIF < MACD 且前一日 DIF >= MACD
        macd_cross_bullish = (dif > dea) & (dif.shift(1) <= dea.shift(1))
        macd_buy_condition = self._expand_event_window(
            macd_cross_bullish,
            self.macd_signal_lag_min,
            self.macd_signal_lag_max,
        )
        macd_sell_condition = (dif < dea) & (dif.shift(1) >= dea.shift(1))

        # MACD 訊號強度（0~1）：DIF-DEA 相對尺度 + 黃金交叉事件視窗。
        macd_spread = dif - dea
        macd_scale = (
            macd_spread.abs().rolling(60, min_periods=20).max().replace(0, np.nan)
        )
        macd_norm = (macd_spread / macd_scale).clip(lower=-1.0, upper=1.0).fillna(0.0)
        macd_signal_power = 0.7 * (
            (macd_norm + 1.0) / 2.0
        ) + 0.3 * macd_buy_condition.astype(float)
        macd_signal_power = self._quantize_signal(
            macd_signal_power, self._resolve_bins("macd")
        )
        self.signal_power["macd"] = macd_signal_power.fillna(0.0)

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
        sufficient_liquidity = avg_volume_30 > self.min_avg_volume_30
        not_abnormal_volume = volume < (avg_volume_30 * 10)

        if self.max_volume_spike_ratio > 0:
            not_abnormal_volume = volume < (avg_volume_30 * self.max_volume_spike_ratio)

        # 成交量訊號強度（0~1）
        volume_ratio = (volume / avg_volume_30.replace(0, np.nan)).fillna(0.0)
        ratio_floor = max(self.volume_above_avg_ratio, 1e-6)
        ratio_ceiling = max(ratio_floor + 1e-6, self.max_volume_spike_ratio)
        volume_signal_power = (
            (volume_ratio - ratio_floor) / (ratio_ceiling - ratio_floor)
        ).clip(lower=0.0, upper=1.0)
        volume_signal_power = volume_signal_power * sufficient_liquidity.astype(float)
        volume_signal_power = self._quantize_signal(
            volume_signal_power, self._resolve_bins("volume")
        )
        self.signal_power["volume"] = volume_signal_power.fillna(0.0)

        return volume_above_avg & sufficient_liquidity & not_abnormal_volume

    def _calculate_institutional_condition(
        self, foreign_net_buy, trust_net_buy, dealer_net_buy
    ):
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
            "foreign_buy": foreign_buy,
            "trust_buy": trust_buy,
            "dealer_buy": dealer_buy,
        }

        # 計算買超家數
        buy_count = (
            foreign_buy.astype(int) + trust_buy.astype(int) + dealer_buy.astype(int)
        )

        # 三者皆買超 (最強訊號)
        institutional_strong_condition = buy_count == 3

        # 至少兩者買超 (次強訊號，包含三者)
        institutional_weak_condition = buy_count >= 2

        institutional_signal_power = (buy_count / 3.0).clip(lower=0.0, upper=1.0)
        institutional_signal_power = self._quantize_signal(
            institutional_signal_power,
            self._resolve_bins("institutional"),
        )
        self.signal_power["institutional"] = institutional_signal_power.fillna(0.0)

        return institutional_strong_condition, institutional_weak_condition

    def _calculate_composite_score(self) -> pd.DataFrame:
        """根據各訊號權重計算綜合分數。"""
        weight_sar = float(self.signal_weights.get("sar", 0.0))
        weight_macd = float(self.signal_weights.get("macd", 0.0))
        weight_volume = float(self.signal_weights.get("volume", 0.0))
        weight_inst = float(self.signal_weights.get("institutional", 0.0))
        weight_total = weight_sar + weight_macd + weight_volume + weight_inst
        if weight_total <= 0:
            weight_total = 1.0

        composite = (
            self.signal_power["sar"] * weight_sar
            + self.signal_power["macd"] * weight_macd
            + self.signal_power["volume"] * weight_volume
            + self.signal_power["institutional"] * weight_inst
        ) / weight_total

        self.signal_power["composite"] = composite.fillna(0.0)
        return self.signal_power["composite"]

    def _build_buy_condition(self, market_data):
        """
        建立買進條件（使用預先計算的指標結果）

        Args:
            market_data: 包含所有市場數據的字典（僅用於兼容性，實際使用緩存）

        Returns:
            buy_condition: 綜合買進條件
        """
        composite_score = self._calculate_composite_score()
        buy_condition = composite_score >= self.buy_score_threshold
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

        composite_score = self.signal_power["composite"]
        if composite_score is None:
            composite_score = self._calculate_composite_score()
        sell_condition = composite_score <= self.sell_score_threshold
        return sell_condition

    def run_strategy(
        self,
        max_stocks=None,
        start_date="2020-01-01",
        fee_ratio=0.001425,
        tax_ratio=0.003,
        sim_fast_mode=None,
        sim_resample="D",
    ):
        """
        執行策略回測（直接沿用 base_position）

        Args:
            max_stocks: 保留相容參數（此策略已不使用持股上限篩選）
            start_date: 回測起始日期
            fee_ratio: 手續費率（預設0.1425%）
            tax_ratio: 證交稅率（預設0.3%）
            sim_fast_mode: 是否啟用 finlab fast_mode
            sim_resample: 回測再平衡週期（預設 D）

        Returns:
            report: 回測報告物件
        """

        run_t0 = perf_counter()

        # 套用起始日期
        base_position = self.base_position.loc[start_date:]
        if len(base_position.index) == 0:
            raise ValueError("No trading days available after start_date.")

        # 直接沿用 base_position，不再施加 max_stocks 篩選。
        selected_mask = base_position.astype(bool)
        selected_count = selected_mask.sum(axis=1)
        final_position = selected_mask.div(
            selected_count.replace(0, np.nan), axis=0
        ).fillna(0.0)
        self._record_profile("build_final_position_sec", run_t0)

        # 執行回測
        self.report = sim(
            position=final_position,
            resample=sim_resample,
            upload=False,
            market=AdjustTWMarketInfo(),
            fee_ratio=fee_ratio,
            tax_ratio=tax_ratio,
            fast_mode=self.sim_fast_mode_default
            if sim_fast_mode is None
            else bool(sim_fast_mode),
            position_limit=1.0,  # 權重已在 final_position 內分配
        )
        self._record_profile("run_strategy_total_sec", run_t0)

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
    # 初始化策略（可調整 SAR/MACD 對齊時間窗）
    strategy = OscarCompositeStrategy(sar_signal_lag_min=0, sar_signal_lag_max=2)

    # 執行回測
    report = strategy.run_strategy(start_date="2020-01-01")

    # 取得回測報告
    final_report = strategy.get_report()

    # 輸出回測結果摘要
    final_report.display(save_path=f"oscar_tw_strategy_report.html")
