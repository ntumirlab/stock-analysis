"""多空轉折模型：指標計算與總分組合（需要 ta-lib 與 strategy_class，於容器內執行）。"""

import numpy as np
import pandas as pd

from alan_dashboard.market_regime_model import breadth_score, compute_components


def test_compute_components_total_range_and_breadth_added_last():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range('2024-01-01', periods=300)
    close = pd.Series(10000 + rng.normal(0, 100, len(dates)).cumsum(), index=dates)
    ohlc = pd.DataFrame({
        'open': close, 'high': close + 50, 'low': close - 50, 'close': close,
    })
    breadth = pd.Series(rng.integers(-60, 61, len(dates)), index=dates)

    comp = compute_components(ohlc, 35, 21, 18, breadth_diff=breadth)
    base = compute_components(ohlc, 35, 21, 18)

    assert comp['total'].between(-10, 10).all()
    assert base['total'].between(-9, 9).all()
    # 現貨分只加在最後：其餘欄位不受影響
    for col in ('ma_score', 'di_score', 'subtotal', 'dif_score', 'macd_score', 'kd_score'):
        pd.testing.assert_series_equal(comp[col], base[col])
    pd.testing.assert_series_equal(comp['total'], base['total'] + comp['breadth_score'],
                                   check_names=False)
    assert (comp['breadth_score'] == breadth_score(breadth)).all()
