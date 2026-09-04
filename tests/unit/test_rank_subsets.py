"""`rank_subsets`——回測要跑哪幾組 ranks。

補跑一整段歷史時 255 組與 1 組差 255 倍的 sim 次數，所以這支決定的是「等四分鐘還是
等十五小時」。排程走的是預設（完整 powerset），補跑工具用 `--ranks` 指定單一組合。

CI 的 requirements-dev.txt 不安裝 finlab，故以 sys.modules 注入假模組
（同 tests/unit/test_create_df_missing_list.py 的手法）。
"""

import sys
import types
from itertools import combinations

import pandas as pd
import pytest


@pytest.fixture
def rank_subsets(monkeypatch):
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

    from strategy_class.golden_ai_tw_strategy_base import rank_subsets as fn
    return fn


class TestTheDefaultIsEveryCombination:
    """排程沒有傳 ranks，所以預設不能變——變了就是每晚少跑 254 組而沒人發現。"""

    def test_it_is_the_full_powerset(self, rank_subsets):
        got = rank_subsets(1, 8)
        pool = list(range(1, 9))
        expected = [list(c) for r in range(1, 9) for c in combinations(pool, r)]
        assert got == expected
        assert len(got) == 255

    def test_a_smaller_pool_shrinks_accordingly(self, rank_subsets):
        assert rank_subsets(1, 3) == [[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]


class TestOnlyOneCombination:
    def test_it_runs_just_that_one(self, rank_subsets):
        assert rank_subsets(1, 8, only=[1, 2, 3, 4, 5, 6, 7, 8]) == [[1, 2, 3, 4, 5, 6, 7, 8]]

    def test_a_subset_of_the_pool_is_fine(self, rank_subsets):
        assert rank_subsets(1, 8, only=[2, 5]) == [[2, 5]]

    def test_the_order_given_is_kept(self, rank_subsets):
        """ranks 字串是 DB 的鍵，重排會變成另一筆資料而不是同一組。"""
        assert rank_subsets(1, 8, only=[3, 1]) == [[3, 1]]


class TestBadInputIsRefused:
    """靜默接受的話，寫進 DB 的 ranks 字串 dashboard 的名次選單選不到，
    那筆資料等於存了沒人看得到。"""

    def test_a_rank_outside_the_pool_raises(self, rank_subsets):
        with pytest.raises(ValueError, match=r'\[9\]'):
            rank_subsets(1, 8, only=[1, 9])

    def test_it_names_every_offender(self, rank_subsets):
        with pytest.raises(ValueError, match=r'\[0, 12\]'):
            rank_subsets(1, 8, only=[12, 1, 0])

    def test_an_empty_selection_raises(self, rank_subsets):
        with pytest.raises(ValueError, match='空'):
            rank_subsets(1, 8, only=[])
