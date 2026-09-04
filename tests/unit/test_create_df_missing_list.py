"""`_create_df` 對「那一週根本沒有清單」的處理——實盤下單路徑，改壞會下真單。

這是實盤「缺清單就空手」的**唯一**保證。原本還有 `check_recommendation_freshness`
的 RuntimeError 擋在前面，但它一拋就是整支 job 中止、當天其他 tranche 的賣出也做不成，
所以已經改成只記警告（見 `core.trading_cycles`）。從此這裡歸零沒做到，就是拿一份
從未發布過的名單去下單，沒有第二道關卡。

情境取自實際資料：2026-01-11 沒有 weekly 清單，`resample('D').ffill()` 會把那一週
填成 01-04 那份，進場日 01-12 照跑就是買 01-04 的名單。

CI 的 requirements-dev.txt 不安裝 finlab，故以 sys.modules 注入假模組（同
tests/unit/test_finlab_auth.py 的手法）。`_create_df` 本身只用 pandas 與傳進去的
universe，不碰 finlab。
"""

import os
import sys
import types

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 清單日（週日）。**2026-01-11 刻意缺席**，前後都有。
PRESENT_SUNDAYS = ['2025-12-28', '2026-01-04', '2026-01-18', '2026-01-25']
MISSING_SUNDAY = pd.Timestamp('2026-01-11')
STOCKS = ['1101', '2330']


class _Stock:
    def __init__(self, sid, priority):
        self.id = sid
        self.priority = priority
        self.SL = 10.0
        self.TP = 90.0


class _Record:
    def __init__(self, date):
        self.date = date
        self.stocks = [_Stock(s, i) for i, s in enumerate(STOCKS)]


@pytest.fixture
def strategy(monkeypatch):
    """注入假 finlab 後載入 base，並把 RecommendationDAO 換成固定清單。"""
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
                      ('finlab.backtest', fake_backtest),
                      ('finlab.dataframe', fake_dataframe),
                      ('finlab.markets', fake_markets),
                      ('finlab.markets.tw', fake_tw)]:
        monkeypatch.setitem(sys.modules, name, mod)

    import strategy_class.golden_ai_tw_strategy_base as base

    class _DAO:
        def __init__(self, *a, **k):
            pass

        def load(self):
            return [_Record(d) for d in PRESENT_SUNDAYS]

    monkeypatch.setattr(base, 'RecommendationDAO', _DAO)
    return base.GoldenAITWStrategyBase(
        task_name='weekly_4w', config_path=os.path.join(ROOT, 'config.yaml'))


def _universe(start='2025-12-28', end='2026-02-13'):
    idx = pd.date_range(start, end, freq='D')
    return pd.DataFrame(1.0, index=idx, columns=STOCKS + ['2454'])


def _position(strategy, **kwargs):
    position, _, _ = strategy._create_df(_universe(), ranks=[1, 2], **kwargs)
    return position


class TestTheMissingWeekIsFlat:
    def test_the_whole_missing_week_holds_nothing(self, strategy):
        """週日到週六整週歸零——不只進場日。ffill 是一天一天鋪的，漏掉任何一天，
        那天就會有一份從未發布過的名單。"""
        position = _position(strategy)
        week = position.loc['2026-01-11':'2026-01-17']
        assert len(week) == 7
        assert not week.to_numpy().any()

    def test_the_entry_day_of_the_missing_week_is_flat(self, strategy):
        """買入日是週一（config buy_weekday=1）。這一格就是會不會下錯單的那一格。"""
        assert not _position(strategy).loc['2026-01-12'].any()

    def test_the_weeks_on_either_side_are_untouched(self, strategy):
        """只歸零缺席那一週。掃過頭的話，有清單的那幾輪也不進場了。"""
        position = _position(strategy)
        assert position.loc['2026-01-05', STOCKS].all()   # 前一輪的進場日
        assert position.loc['2026-01-19', STOCKS].all()   # 後一輪的進場日
        assert position.loc['2026-01-10', STOCKS].all()   # 缺席週的前一天

    def test_a_week_with_a_list_is_never_zeroed_anywhere(self, strategy):
        """有清單的那幾週，每一天都該持有——歸零條件寫反的話這裡會整片掉。"""
        position = _position(strategy)
        for sunday in PRESENT_SUNDAYS:
            week = position.loc[sunday:pd.Timestamp(sunday) + pd.Timedelta(days=6)]
            assert week[STOCKS].to_numpy().all(), f'{sunday} 那一週被誤傷'


class TestTheLiveMorningRun:
    """實盤早上跑的形狀：`end_date` 延伸到今天，超出最後一份清單。"""

    def test_days_extended_past_the_last_list_stay_with_that_list(self, strategy):
        """延伸出來的那幾天仍屬於最後一份清單的週日，不該被當成缺席。"""
        position = _position(strategy, end_date=pd.Timestamp('2026-01-29'))
        assert position.loc['2026-01-29', STOCKS].all()

    def test_extending_into_a_week_with_no_list_is_flat(self, strategy):
        """今天是 02-02（週一），02-01 那個週日還沒有清單 → 今天不進場。
        這正是清單晚發時實盤會走到的路徑。"""
        position = _position(strategy, end_date=pd.Timestamp('2026-02-02'))
        assert not position.loc['2026-02-02'].any()
        assert position.loc['2026-01-31', STOCKS].all()   # 上一輪的週六仍持有


class TestStopLevelsAreNotZeroed:
    """sl/tp 只產生出場訊號，`hold_until` 只出場真的持有中的部位，所以空手那幾天是
    no-op。反過來歸零才有害：sl 的 0 會被 replace 成 NaN、比較恆為 False，跨週持有
    到那幾天的部位會整週失去停損門檻。"""

    def test_the_missing_week_keeps_its_stop_levels(self, strategy):
        strategy.use_db_sl = True
        strategy.use_db_tp = True
        _, sl_df, tp_df = strategy._create_df(_universe(), ranks=[1, 2])
        assert (sl_df.loc['2026-01-11':'2026-01-17', STOCKS] == 10.0).to_numpy().all()
        assert (tp_df.loc['2026-01-11':'2026-01-17', STOCKS] == 90.0).to_numpy().all()


def test_the_mask_is_driven_by_owning_sunday(strategy):
    """歸零的鍵必須與 `core.trading_cycles.owning_sunday` 同一套——那支改了對齊方向
    而這裡沒跟上的話，遮罩會整體位移一天、每週的週日或週六被錯誤歸零。"""
    from core.trading_cycles import owning_sunday

    position = _position(strategy)
    present = {pd.Timestamp(s) for s in PRESENT_SUNDAYS}
    for day, row in position.iterrows():
        owner = owning_sunday([day])[0]
        assert bool(row.any()) == (owner in present), f'{day.date()} 的歸零與 owning_sunday 不一致'
    assert MISSING_SUNDAY not in present
