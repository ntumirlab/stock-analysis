from dataclasses import asdict, dataclass


@dataclass
class SarParams:
    acceleration: float | None = None
    maximum: float | None = None


@dataclass
class MacdParams:
    fastperiod: int | None = None
    slowperiod: int | None = None
    signalperiod: int | None = None


@dataclass
class SignalWeights:
    sar: float | None = None
    macd: float | None = None
    volume: float | None = None
    institutional: float | None = None


@dataclass
class SignalQuantileBins:
    sar: int | None = None
    macd: int | None = None
    volume: int | None = None
    institutional: int | None = None


@dataclass
class OscarCompositeParams:
    sar_params: SarParams | dict | None = None
    macd_params: MacdParams | dict | None = None
    volume_above_avg_ratio: float | None = None
    min_avg_volume_30: float | None = None
    new_high_ratio_120: float | None = None
    sar_signal_lag_max: int | None = None
    sar_event_decay_alpha: float | None = None
    macd_signal_lag_max: int | None = None
    macd_event_decay_alpha: float | None = None
    max_volume_spike_ratio: float | None = None
    signal_weights: SignalWeights | dict | None = None
    buy_score_threshold: float | None = None
    sell_score_threshold: float | None = None
    signal_quantile_bins: SignalQuantileBins | dict | int | None = None
    sar_proximity_weight: float | None = None
    sar_event_weight: float | None = None
    macd_proximity_weight: float | None = None
    macd_event_weight: float | None = None
    sar_near_sigmoid_slope: float | None = None
    sar_near_distance_scale: float | None = None
    sar_event_sigmoid_slope: float | None = None
    sar_event_distance_scale: float | None = None
    sar_history_lookback: int | None = None
    sar_history_decay_alpha: float | None = None
    macd_near_sigmoid_slope: float | None = None
    macd_near_distance_scale: float | None = None
    macd_event_sigmoid_slope: float | None = None
    macd_event_distance_scale: float | None = None
    macd_history_lookback: int | None = None
    macd_history_decay_alpha: float | None = None

    def __post_init__(self):
        if isinstance(self.sar_params, dict):
            self.sar_params = SarParams(**self.sar_params)
        elif self.sar_params is None:
            self.sar_params = SarParams()

        if isinstance(self.macd_params, dict):
            self.macd_params = MacdParams(**self.macd_params)
        elif self.macd_params is None:
            self.macd_params = MacdParams()

        if isinstance(self.signal_weights, dict):
            self.signal_weights = SignalWeights(**self.signal_weights)
        elif self.signal_weights is None:
            self.signal_weights = SignalWeights()

        if isinstance(self.signal_quantile_bins, dict):
            self.signal_quantile_bins = SignalQuantileBins(**self.signal_quantile_bins)
        elif isinstance(self.signal_quantile_bins, int):
            self.signal_quantile_bins = SignalQuantileBins(
                sar=self.signal_quantile_bins,
                macd=self.signal_quantile_bins,
                volume=self.signal_quantile_bins,
                institutional=self.signal_quantile_bins,
            )
        elif self.signal_quantile_bins is None:
            self.signal_quantile_bins = SignalQuantileBins()

        # Defaults for bounded near/event fusion math.
        self.sar_proximity_weight = 0.4 if self.sar_proximity_weight is None else self.sar_proximity_weight
        self.sar_event_weight = 1.0 if self.sar_event_weight is None else self.sar_event_weight
        self.macd_proximity_weight = 0.4 if self.macd_proximity_weight is None else self.macd_proximity_weight
        self.macd_event_weight = 1.0 if self.macd_event_weight is None else self.macd_event_weight

        self.sar_near_sigmoid_slope = 8.0 if self.sar_near_sigmoid_slope is None else self.sar_near_sigmoid_slope
        self.sar_near_distance_scale = 0.02 if self.sar_near_distance_scale is None else self.sar_near_distance_scale
        self.sar_event_sigmoid_slope = 10.0 if self.sar_event_sigmoid_slope is None else self.sar_event_sigmoid_slope
        self.sar_event_distance_scale = 0.02 if self.sar_event_distance_scale is None else self.sar_event_distance_scale
        self.sar_history_lookback = 4 if self.sar_history_lookback is None else self.sar_history_lookback
        self.sar_history_decay_alpha = 1.8 if self.sar_history_decay_alpha is None else self.sar_history_decay_alpha

        self.macd_near_sigmoid_slope = 8.0 if self.macd_near_sigmoid_slope is None else self.macd_near_sigmoid_slope
        self.macd_near_distance_scale = 1.0 if self.macd_near_distance_scale is None else self.macd_near_distance_scale
        self.macd_event_sigmoid_slope = 10.0 if self.macd_event_sigmoid_slope is None else self.macd_event_sigmoid_slope
        self.macd_event_distance_scale = 1.0 if self.macd_event_distance_scale is None else self.macd_event_distance_scale
        self.macd_history_lookback = 4 if self.macd_history_lookback is None else self.macd_history_lookback
        self.macd_history_decay_alpha = 1.8 if self.macd_history_decay_alpha is None else self.macd_history_decay_alpha

    def to_dict(self) -> dict:
        return asdict(self)
