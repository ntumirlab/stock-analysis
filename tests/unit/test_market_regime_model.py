"""多空轉折模型：市值前 150 檔成分、現貨檔數差與分數組成（不需 finlab）。"""

import numpy as np
import pandas as pd
import pytest

from alan_dashboard.market_regime_model import (
    breadth_score, listed_common_stocks, ma_direction_breadth, top_n_membership,
)


def test_listed_common_stocks_filters_market_and_ids():
    cat = pd.DataFrame({
        'stock_id': ['2330', '0050', '2881A', '910861', '7610', '6488', '00631L'],
        'name':     ['台積電', '元大台灣50', '富邦特', '神隆', '聯友金屬-創', '環球晶', '元大台灣50正2'],
        'market':   ['sii', 'etf', 'sii', 'sii', 'sii', 'otc', 'etf'],
    })
    assert listed_common_stocks(cat) == ['2330']


def test_top_n_membership_reranks_at_quarter_end_effective_next_day():
    dates = pd.bdate_range('2025-06-25', '2025-07-04')  # 跨 6/30 季末
    mv = pd.DataFrame(index=dates, columns=['A', 'B', 'C'], dtype=float)
    mv[:] = [[3, 2, 1]] * len(dates)
    mv.loc['2025-06-30':] = [1, 2, 3]  # 季末當天 C 的市值超過 A
    member = top_n_membership(mv, ['A', 'B', 'C', 'Z'], n=2)

    assert member.loc['2025-06-27'].tolist() == [False, False, False]  # 尚無季末排名
    assert member.loc['2025-06-30'].tolist() == [False, False, False]  # 季末當天尚未生效
    assert member.loc['2025-07-01'].tolist() == [False, True, True]     # 次一交易日起用新排名
    assert member.loc['2025-07-04'].tolist() == [False, True, True]     # 整季鎖定
    assert list(member.columns) == ['A', 'B', 'C']  # 不在市值資料裡的代號被忽略


def test_ma_direction_breadth_counts_members_only():
    dates = pd.bdate_range('2025-01-01', periods=30)
    n = len(dates)
    close = pd.DataFrame({
        'UP':   np.arange(1, n + 1, dtype=float),          # 三條均線持續向上
        'DOWN': np.arange(n, 0, -1, dtype=float),          # 三條均線持續向下
        'FLAT': np.full(n, 10.0),                          # 均線持平：不計
        'OUT':  np.arange(1, n + 1, dtype=float),          # 向上但不是成分股
    }, index=dates)
    member = pd.DataFrame(True, index=dates, columns=['UP', 'DOWN', 'FLAT'])
    member['OUT'] = False
    diff = ma_direction_breadth(close, member)

    assert diff.iloc[-1] == 0            # UP(+1) 與 DOWN(−1) 抵銷，FLAT 與 OUT 不計
    member['DOWN'] = False
    assert ma_direction_breadth(close, member).iloc[-1] == 1
    assert ma_direction_breadth(close, member).iloc[:20].eq(0).all()  # MA20 暖身期間無值
    # 成分名單資料日比收盤價早（市值晚更新）：最後幾天沿用最近一次名單
    assert ma_direction_breadth(close, member.iloc[:-3]).iloc[-1] == 1
    # 當日個股收盤價尚未更新（全為 NaN）：無值，而不是 0
    close.iloc[-1] = np.nan
    assert np.isnan(ma_direction_breadth(close, member).iloc[-1])


def test_breadth_score_threshold():
    diff = pd.Series([26, 25, 0, -25, -26])
    assert breadth_score(diff).tolist() == [1, 0, 0, 0, -1]


def test_compute_components_total_range_and_breadth_added_last():
    talib = pytest.importorskip('talib')  # CI 無 ta-lib，只在本機／容器內執行
    from alan_dashboard.market_regime_model import compute_components

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
