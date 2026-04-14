import logging
import pandas as pd
from finlab.markets.tw import TWMarket

logger = logging.getLogger(__name__)


class CustomPriceTWMarket(TWMarket):
    """TWMarket extended to support per-cell buy/sell price differentiation.

    Parameters
    ----------
    position : pd.DataFrame
        Precomputed position signals (date × stock), values in {0, 1} or float.
    buy_price : pd.DataFrame
        Price to use when a stock is entered (position[n] > position[n-1]).
    sell_price : pd.DataFrame
        Price to use when a stock is exited (position[n] < position[n-1]).

    Supported trade_at_price strings
    ---------------------------------
    'custom' : blended DataFrame — buy_price on entry days, sell_price on exit days,
                 NaN (falls back to close) on hold/flat days.
    Any standard string ('open', 'close', 'high', 'low', etc.) raises ValueError since those are not supported by this custom market.
    """

    def __init__(
        self,
        position: pd.DataFrame,
        buy_price: pd.DataFrame,
        sell_price: pd.DataFrame,
    ):
        super().__init__()
        self._position = position.astype(float).fillna(0)
        self._buy_price = buy_price
        self._sell_price = sell_price

        # Cache so we don't rebuild on repeated calls
        self._cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_trading_price(self, name: str, adj: bool = True) -> pd.DataFrame:
        if name == 'custom':
            return self._get_or_build('custom', self._build_custom)
        raise ValueError(f"Unsupported trade_at_price name in CustomPriceTWMarket: {name}")

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _get_or_build(self, key: str, builder) -> pd.DataFrame:
        if key not in self._cache:
            self._cache[key] = builder()
        return self._cache[key]

    def _entry_exit_masks(self):
        """Return (entering, exiting) boolean DataFrames aligned to buy_price index."""
        ref_index = self._buy_price.index
        ref_cols = self._buy_price.columns

        pos = self._position.reindex(index=ref_index, columns=ref_cols, fill_value=0)
        diff = pos.diff()  # +value = entering, -value = exiting

        entering = diff > 0
        exiting = diff < 0
        return entering, exiting

    def _build_custom(self) -> pd.DataFrame:
        """buy_price on entry, sell_price on exit, NaN elsewhere (hold/flat days)."""
        entering, exiting = self._entry_exit_masks()

        blended = pd.DataFrame(
            float("nan"),
            index=self._buy_price.index,
            columns=self._buy_price.columns,
        )
        blended[entering] = self._buy_price[entering]
        blended[exiting] = self._sell_price[exiting]
        return blended
