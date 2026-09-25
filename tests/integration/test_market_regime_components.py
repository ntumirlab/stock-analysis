"""多空轉折模型：指標計算與總分組合（需要 ta-lib 與 strategy_class，於容器內執行）。"""

import numpy as np
import pandas as pd

from alan_dashboard.market_regime_model import breadth_score, compute_components, options_score


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

    assert comp['total'].between(-9, 9).all()
    assert base['total'].between(-8, 8).all()
    # 均線分 ±1 需 KD 同向：均線分為 ±1 的日期 KD 方向必同號；KD 不同向或反向時均線分不會是 ±1
    one_point = comp['ma_score'].abs() == 1
    assert (comp.loc[one_point, 'kd_dir'] == comp.loc[one_point, 'ma_score']).all()
    assert one_point.any()
    # 現貨分只加在最後：其餘欄位不受影響
    for col in ('ma_score', 'di_score', 'subtotal', 'dif_score', 'macd_score', 'kd_dir'):
        pd.testing.assert_series_equal(comp[col], base[col])
    pd.testing.assert_series_equal(comp['total'], base['total'] + comp['breadth_score'],
                                   check_names=False)
    assert (comp['breadth_score'] == breadth_score(breadth)).all()


def test_compute_components_aligns_pcr_to_ohlc_dates_and_adds_options_score():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range('2024-01-01', periods=300)
    close = pd.Series(10000 + rng.normal(0, 100, len(dates)).cumsum(), index=dates)
    ohlc = pd.DataFrame({'open': close, 'high': close + 50, 'low': close - 50, 'close': close})
    # P/C 只涵蓋後半段、且日期多一個非交易日（週六）：模擬資料起始較晚與日曆不完全一致
    pcr_dates = dates[150:].union([dates[200] + pd.Timedelta(days=5)])
    pcr = pd.DataFrame({
        'near_month': '202406',
        'near_pcr': rng.uniform(70, 160, len(pcr_dates)),
        'far_pcr': rng.uniform(70, 160, len(pcr_dates)),
    }, index=pcr_dates)

    comp = compute_components(ohlc, 35, 21, 18, pcr=pcr)
    base = compute_components(ohlc, 35, 21, 18)

    assert comp.index.equals(dates)
    # 前半段無 P/C：值為 NaN、選擇權分 0；後半段對齊到交易日、與直接算的分數一致
    assert comp['pcr_near'].iloc[:150].isna().all() and (comp['options_score'].iloc[:150] == 0).all()
    pd.testing.assert_series_equal(comp['pcr_near'].iloc[150:], pcr['near_pcr'].reindex(dates[150:]),
                                   check_names=False)
    expected = options_score(pcr['near_pcr'].reindex(dates), pcr['far_pcr'].reindex(dates))
    assert (comp['options_score'] == expected).all()
    assert comp['options_score'].abs().max() >= 1  # 隨機資料要真的觸發計分，否則下面的等式沒有意義
    # 選擇權分只加在最後：其餘欄位不變，總分 = 原總分 + 選擇權分
    for col in ('ma_score', 'di_score', 'subtotal', 'dif_score', 'macd_score', 'kd_dir', 'breadth_score'):
        pd.testing.assert_series_equal(comp[col], base[col])
    pd.testing.assert_series_equal(comp['total'], base['total'] + comp['options_score'], check_names=False)
    assert comp['total'].between(-11, 11).all()
