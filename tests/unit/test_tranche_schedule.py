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
    def test_the_anchor_is_the_day_before_the_first_weekly_buy_day(self):
        """相位原點＝weekly 最早一份清單對齊後的週日，隔天（buy_weekday=週一）就是首個買入日。"""
        assert anchor_sunday('weekly_4w') + pd.Timedelta(days=1) == pd.Timestamp('2025-09-29')

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    def test_the_anchor_itself_is_tranche_one(self, strategy):
        assert tranche_of(strategy, anchor_sunday(strategy)) == 1

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    def test_anchors_are_sundays(self, strategy):
        assert TRANCHE_ANCHOR_SUNDAYS[strategy].weekday() == 6

    def test_both_strategies_share_one_phase(self):
        """同名必須同義：tranche_2 在兩支策略、在實盤，指的都是同一組週一。
        各自從自己的第一份清單推的話 monthly 會位移一格，名字就開始說謊。"""
        assert anchor_sunday('monthly') == anchor_sunday('weekly_4w')
        for d in ('2026-08-09', '2026-01-11', '2025-10-05'):
            assert tranche_of('weekly_4w', d) == tranche_of('monthly', d)

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

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    def test_a_range_starting_mid_week_still_covers_its_first_days(self, strategy):
        """range 從週一開始時，涵蓋那幾天的清單日是**上一個**週日、在 range 之前。
        用 idx.min() 當起點的話它會整個掉出去，開頭那幾天於是沒有任何 tranche 進場。"""
        rng = pd.date_range('2026-01-05', '2026-02-28', freq='D')   # 週一開始
        owner = pd.Timestamp('2026-01-04')                          # 擁有 01-05 的週日
        picked = set().union(*(set(tranche_sundays(strategy, rng, t))
                               for t in range(1, NUM_TRANCHES + 1)))
        assert owner in picked

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    def test_a_range_starting_on_a_sunday_is_unchanged(self, strategy):
        """週日開頭時起點就是它自己——退起點這件事不能動到既有結果。"""
        rng = pd.date_range('2026-01-04', '2026-02-28', freq='D')
        picked = set().union(*(set(tranche_sundays(strategy, rng, t))
                               for t in range(1, NUM_TRANCHES + 1)))
        assert min(picked) == pd.Timestamp('2026-01-04')


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


class TestPhaseMatchesLiveTrading:
    """回測的 tranche 編號必須就是實盤的 tranche_1~4，否則 dashboard 上的
    「tranche2」跟實倉裡的「tranche_2」是兩回事，而兩邊都不會有人發現。

    實盤在 `jobs.order_executor.load_strategies` 用 config 的 `cycle_start_date`
    當第一個買入日，`build_tranche_specs` 再往後鋪 7k 天開出 tranche_1~4。
    回測的錨點是從最早的推薦清單推出來的，跟 config 沒有任何程式上的連結——
    目前對得上是因為兩者剛好差 28 天的整數倍。所以要在這裡釘住。
    """

    @staticmethod
    def _cycle_start_date():
        import os
        import yaml

        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(root, 'config.yaml'), encoding='utf-8') as f:
            config = yaml.safe_load(f)
        constant = config['users']['kiri']['shioaji']['constant']
        return pd.Timestamp(str(constant['cycle_start_date'])), constant

    def test_live_tranche_k_is_backtest_tranche_k(self):
        cycle_start, constant = self._cycle_start_date()
        strategy = constant['golden_ai_frequency']
        live_tranches = int(constant['hold_weeks'])
        if strategy == 'weekly' and live_tranches > 1:
            strategy = 'weekly_4w'   # 實盤吃 weekly 清單、持有多週＝回測的 weekly_4w

        # 回測要跑幾份 tranche 是回測自己的事，**不要求**與實盤相同。但份數相同時
        # 編號就會被讀成同一件事（dashboard 的 tranche2 vs 實倉 state 的 tranche_2），
        # 那個對應必須成立。份數不同時這個對應本來就不存在，沒什麼好守的。
        if live_tranches != NUM_TRANCHES:
            pytest.skip(f'回測跑 {NUM_TRANCHES} 份、實盤 {live_tranches} 份，編號無對應關係')

        assert strategy in FOUR_WEEK_STRATEGIES

        # 兩支 4 週策略共用同一個相位原點，所以這個對應對兩支都要成立——只驗實盤
        # 在跑的那一支的話，另一支哪天位移了也沒人會發現。
        for s in FOUR_WEEK_STRATEGIES:
            for k in range(NUM_TRANCHES):
                entry = cycle_start + pd.Timedelta(days=7 * k)      # build_tranche_specs
                list_sunday = entry - pd.Timedelta(days=1)          # 進場日前一天的清單
                assert tranche_of(s, list_sunday) == k + 1, (
                    f"實盤 tranche_{k + 1}（買入日 {entry:%Y-%m-%d}）對到 {s} 的 "
                    f"tranche{tranche_of(s, list_sunday)}。改過 config 的 "
                    f"cycle_start_date 嗎？它必須與 TRANCHE_ANCHOR_SUNDAYS 差 "
                    f"{NUM_TRANCHES} 週的整數倍。"
                )

    def test_the_live_anchor_lands_on_a_buy_weekday_after_a_list_sunday(self):
        cycle_start, _ = self._cycle_start_date()
        assert (cycle_start - pd.Timedelta(days=1)).weekday() == 6


class TestRotationIsSeamless:
    """`NUM_TRANCHES` 同時是「開幾份」與「每份持有幾週」。這兩件事分開寫死過一次
    （`_run_core` 的 `days=22` 與 `HOLD_WEEKS` 的 4），只有恰好都是 4 才對得上。

    現在兩者都從 `NUM_TRANCHES` 導出，這裡守住導出關係沒被誰改回字面值：
    同一份 tranche 的下一次進場，必須恰好接在這次出場之後。
    """

    BUY, SELL = 0, 4          # config.yaml 的 buy_weekday 1 / sell_weekday 5 減 1

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    def test_hold_weeks_follows_the_tranche_count(self, strategy):
        from core.node_backtest import HOLD_WEEKS

        assert HOLD_WEEKS[strategy] == NUM_TRANCHES

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    @pytest.mark.parametrize('tranche', range(1, NUM_TRANCHES + 1))
    def test_a_tranche_never_re_enters_before_it_has_exited(self, strategy, tranche):
        from core.node_backtest import HOLD_WEEKS, node_dates

        sundays = tranche_sundays(strategy, _all_sundays('2025-10-01', '2026-12-31'), tranche)
        for this_week, next_week in zip(sundays, sundays[1:]):
            _, exit_date = node_dates(this_week, self.BUY, self.SELL, HOLD_WEEKS[strategy])
            next_entry, _ = node_dates(next_week, self.BUY, self.SELL, HOLD_WEEKS[strategy])
            assert exit_date < next_entry, (
                f'{strategy} tranche{tranche}: {next_week.date()} 那期在上一期 '
                f'{exit_date.date()} 出場之前就進場了——同一份 tranche 的部位會自己疊到'
                f'自己，出場訊號是整列廣播的，新買的會被上一輪的賣單掃掉。'
            )

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    def test_the_gap_between_selling_and_re_entering_is_always_the_weekend(self, strategy):
        """出場（週五）到下次進場（下週一）恆為 3 天，與 NUM_TRANCHES 無關——
        進場節奏 7N 減掉持有期 (N-1)*7 + (sell - buy)，N 會消掉。"""
        from core.node_backtest import HOLD_WEEKS, node_dates

        sundays = tranche_sundays(strategy, _all_sundays('2025-10-01', '2026-12-31'), 1)
        gaps = set()
        for this_week, next_week in zip(sundays, sundays[1:]):
            _, exit_date = node_dates(this_week, self.BUY, self.SELL, HOLD_WEEKS[strategy])
            next_entry, _ = node_dates(next_week, self.BUY, self.SELL, HOLD_WEEKS[strategy])
            gaps.add((next_entry - exit_date).days)
        assert gaps == {7 - (self.SELL - self.BUY)}


class TestBadTrancheNumbers:
    """超出範圍要噴錯，不能靜默回空——空的回測看起來像資料不足而不像參數打錯。"""

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    @pytest.mark.parametrize('bad', [0, -1, NUM_TRANCHES + 1, 99])
    def test_out_of_range_raises(self, strategy, bad):
        with pytest.raises(ValueError, match='tranche'):
            tranche_sundays(strategy, _all_sundays('2026-01-01', '2026-03-01'), bad)

    @pytest.mark.parametrize('strategy', FOUR_WEEK_STRATEGIES)
    def test_it_raises_even_when_the_range_is_empty(self, strategy):
        """先驗參數再看資料，否則空區間會把參數錯誤蓋掉。"""
        with pytest.raises(ValueError, match='tranche'):
            tranche_sundays(strategy, pd.DatetimeIndex([]), 0)
