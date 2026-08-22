"""tranche 進場排程測試（純運算，不需要 finlab）。

這支守的是「回測與實盤用同一套進場排程」——舊的月索引（當月第 n 個週日）結構上
每年會漏掉約 4 份清單，實盤的錨點連續輪動不會。
"""

import pandas as pd
import pytest

from core.tranche_schedule import (
    NUM_TRANCHES,
    TRANCHE_ANCHOR_SUNDAYS,
    anchor_sunday,
    tranche_of,
    tranche_sundays,
)

FOUR_WEEK_STRATEGIES = ['weekly_4w', 'monthly']


def _all_sundays(start, end):
    return pd.date_range(start, end, freq='W-SUN')


class TestAnchor:
    @pytest.mark.parametrize('strategy, first_buy_day', [
        ('weekly_4w', '2025-09-29'),
        ('monthly', '2025-10-06'),
    ])
    def test_anchor_sunday_is_the_day_before_the_first_buy_day(self, strategy, first_buy_day):
        """錨點＝各策略最早一份清單對齊後的週日，隔天（buy_weekday=週一）就是首個買入日。"""
        assert anchor_sunday(strategy) + pd.Timedelta(days=1) == pd.Timestamp(first_buy_day)

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    def test_the_anchor_itself_is_tranche_one(self, strategy):
        assert tranche_of(strategy, anchor_sunday(strategy)) == 1

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    def test_anchors_are_sundays(self, strategy):
        assert TRANCHE_ANCHOR_SUNDAYS[strategy].weekday() == 6

    def test_the_two_strategies_have_their_own_phase(self):
        """兩支的清單來源不同、起跑差一週，所以同一個週日不會落在同一份 tranche。"""
        assert anchor_sunday('monthly') - anchor_sunday('weekly_4w') == pd.Timedelta(weeks=1)
        assert tranche_of('weekly_4w', '2026-08-09') != tranche_of('monthly', '2026-08-09')

    def test_a_strategy_without_a_phase_says_so(self):
        with pytest.raises(KeyError, match='weekly'):
            anchor_sunday('weekly')


class TestTrancheOf:
    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    def test_consecutive_sundays_walk_through_the_tranches_in_order(self, strategy):
        anchor = anchor_sunday(strategy)
        got = [tranche_of(strategy, anchor + pd.Timedelta(weeks=k)) for k in range(9)]
        assert got == [1, 2, 3, 4, 1, 2, 3, 4, 1]

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    def test_sundays_before_the_anchor_still_land_on_a_tranche(self, strategy):
        """錨點之前沒有清單，但相位仍要算得出來，不能因為負數取模跑到 0 或 5。"""
        anchor = anchor_sunday(strategy)
        got = [tranche_of(strategy, anchor - pd.Timedelta(weeks=k)) for k in range(1, 9)]
        assert got == [4, 3, 2, 1, 4, 3, 2, 1]

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    def test_every_sunday_of_two_years_gets_exactly_one_tranche(self, strategy):
        got = {tranche_of(strategy, d) for d in _all_sundays('2025-01-01', '2026-12-31')}
        assert got == set(range(1, NUM_TRANCHES + 1))


class TestTrancheSundays:
    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    def test_the_four_tranches_cover_every_sunday_without_overlap(self, strategy):
        """月索引一年只有 12x4=48 個位子、實際約 52.2 週，必然漏掉約 4 份清單；
        連續輪動則是不重不漏。"""
        span = _all_sundays('2025-01-01', '2026-12-31')
        picked = [set(tranche_sundays(strategy, span, n))
                  for n in range(1, NUM_TRANCHES + 1)]
        assert set().union(*picked) == set(span)
        assert sum(len(p) for p in picked) == len(span)

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    @pytest.mark.parametrize('tranche', range(1, NUM_TRANCHES + 1))
    def test_the_same_tranche_re_enters_exactly_four_weeks_later(self, strategy, tranche):
        """實盤每份 tranche 的相鄰進場間隔全部是 28 天，回測必須一致。"""
        picked = tranche_sundays(strategy, _all_sundays('2025-01-01', '2026-12-31'), tranche)
        gaps = picked.to_series().diff().dropna().unique()
        assert list(gaps) == [pd.Timedelta(weeks=NUM_TRANCHES)]

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    def test_picks_agree_with_tranche_of(self, strategy):
        span = _all_sundays('2025-09-01', '2026-08-31')
        for n in range(1, NUM_TRANCHES + 1):
            assert all(tranche_of(strategy, d) == n for d in tranche_sundays(strategy, span, n))

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    def test_a_daily_index_is_accepted_and_only_sundays_come_back(self, strategy):
        """呼叫端傳的是 position 的日索引（`_create_df` resample('D') 出來的）。"""
        daily = pd.date_range('2026-07-01', '2026-08-31', freq='D')
        picked = tranche_sundays(strategy, daily, 1)
        assert len(picked) > 0
        assert all(d.weekday() == 6 for d in picked)

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    def test_an_empty_range_picks_nothing(self, strategy):
        assert len(tranche_sundays(strategy, pd.DatetimeIndex([]), 1)) == 0


class TestFifthSundaysAreNoLongerSkipped:
    """月索引時代 `_get_nth_sundays` 只跑 n=1~4，這三份清單從來不會進場。"""

    FIFTH_SUNDAYS = ['2025-11-30', '2026-03-29', '2026-05-31']

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    @pytest.mark.parametrize('list_date', FIFTH_SUNDAYS)
    def test_they_now_belong_to_a_tranche(self, strategy, list_date):
        n = tranche_of(strategy, list_date)
        assert 1 <= n <= NUM_TRANCHES
        assert pd.Timestamp(list_date) in tranche_sundays(
            strategy, _all_sundays('2025-01-01', '2026-12-31'), n)
