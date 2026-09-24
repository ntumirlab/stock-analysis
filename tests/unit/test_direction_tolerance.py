"""Alan 策略基底的「上揚／下彎」判斷：相對容忍值濾除浮點誤差。

CI 的 requirements-dev.txt 不安裝 finlab，故以 sys.modules 注入假模組
（同 tests/unit/test_rank_subsets.py 的手法）。
"""

import sys
import types

import pandas as pd
import pytest


@pytest.fixture
def base_cls(monkeypatch):
    fake_finlab = types.ModuleType('finlab')
    fake_finlab.data = types.ModuleType('finlab.data')
    fake_backtest = types.ModuleType('finlab.backtest')
    fake_backtest.sim = lambda *a, **k: None
    fake_dataframe = types.ModuleType('finlab.dataframe')
    fake_dataframe.FinlabDataFrame = pd.DataFrame
    fake_markets = types.ModuleType('finlab.markets')
    fake_tw = types.ModuleType('finlab.markets.tw')
    fake_tw.TWMarket = type('TWMarket', (), {'__init__': lambda self: None})
    fake_markets.tw = fake_tw
    for name, mod in [('finlab', fake_finlab), ('finlab.data', fake_finlab.data),
                      ('finlab.backtest', fake_backtest), ('finlab.dataframe', fake_dataframe),
                      ('finlab.markets', fake_markets), ('finlab.markets.tw', fake_tw)]:
        monkeypatch.setitem(sys.modules, name, mod)
    from strategy_class.alan_tw_strategy_base import AlanTWStrategyBase
    return AlanTWStrategyBase


def _obj(cls, tol=None):
    obj = cls.__new__(cls)  # 不跑 __init__（會拉 finlab 資料）
    if tol is not None:
        obj.direction_tol = tol
    return obj


def test_default_tolerance_is_enabled(base_cls):
    assert base_cls.direction_tol == 1e-9


def test_floating_noise_is_flat_but_real_moves_keep_direction(base_cls):
    dates = pd.bdate_range('2025-01-01', periods=4)
    ma = pd.DataFrame({
        'NOISE_UP': [100.0, 100.0, 100.0, 100.0 + 1e-11],   # 變動 1e-13 倍：浮點誤差
        'NOISE_DN': [100.0, 100.0, 100.0, 100.0 - 1e-11],
        'UP':       [100.0, 100.0, 100.0, 100.01],           # 真的漲一檔
        'DN':       [100.0, 100.0, 100.0, 99.99],
    }, index=dates)
    obj = _obj(base_cls)
    rising, falling = obj._rising(ma), obj._falling(ma)
    assert rising.iloc[-1].tolist() == [False, False, True, False]
    assert falling.iloc[-1].tolist() == [False, False, False, True]
    assert not rising.iloc[0].any() and not falling.iloc[0].any()  # 第一天無前值：不算方向


def test_tolerance_is_relative_to_level(base_cls):
    # 同樣 1e-8 的絕對變動：對 5 元的股票超過十億分之一（真變動），對 5000 元的股票不到（視為持平）
    ma = pd.DataFrame({'CHEAP': [5.0, 5.0 + 1e-8], 'PRICEY': [5000.0, 5000.0 + 1e-8]})
    obj = _obj(base_cls)
    assert obj._rising(ma).iloc[-1].tolist() == [True, False]


def test_scale_keeps_threshold_near_zero_crossing(base_cls):
    # DIF 穿越零：值 ±1e-13 是雜訊。只靠自身的相對門檻會判成方向；用股價當量尺才視為持平
    dif = pd.DataFrame({'X': [1e-13, -1e-13, 0.0, 0.5]})
    close = pd.DataFrame({'X': [100.0] * 4})
    obj = _obj(base_cls)
    assert obj._falling(dif).iloc[1].item() and obj._rising(dif).iloc[2].item()
    assert not obj._falling(dif, scale=close).iloc[1].item()
    assert not obj._rising(dif, scale=close).iloc[2].item()
    assert obj._rising(dif, scale=close).iloc[3].item()            # 真的上揚仍成立
    assert not obj._rising(dif, scale=100.0).iloc[2].item()        # 純量量尺（K/D 用 100）


def test_zero_tolerance_restores_strict_comparison(base_cls):
    ma = pd.DataFrame({'X': [100.0, 100.0 + 1e-11]})
    assert _obj(base_cls, tol=0.0)._rising(ma).iloc[-1].item()   # 嚴格比較：1e-13 倍也算上揚
    assert not _obj(base_cls)._rising(ma).iloc[-1].item()        # 預設容忍值：視為持平
