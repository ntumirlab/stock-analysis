"""GoldenAI 下單週期日期運算的測試——實盤買賣日排程，改壞會直接影響真實下單。

案例來源：2026-07-02 錨點制修復時的重放驗證（含當時在容器中實測過的日期）。
weekday 慣例：pandas dayofweek，週一=0。kiri 實際設定 = 週一買(0)、週五賣(4)、hold 4 週。
"""

import logging

import pandas as pd
import pytest

from core.trading_cycles import (
    align_to_sunday,
    build_tranche_specs,
    owning_sunday,
    check_recommendation_freshness,
    compute_cycles,
    compute_historical_cycles,
    find_current_cycle,
    missing_list_sunday,
)


def ts(s):
    return pd.Timestamp(s)


KIRI = dict(anchor="2026-07-06", buy_weekday=0, sell_weekday=4, hold_weeks=4)


def kiri_cycles(until="2026-10-01"):
    return compute_cycles(KIRI["anchor"], KIRI["buy_weekday"], KIRI["sell_weekday"],
                          KIRI["hold_weeks"], ts(until))


class TestComputeCycles:
    def test_kiri_schedule_first_cycles(self):
        cycles = kiri_cycles()
        assert cycles[0] == (ts("2026-07-06"), ts("2026-07-31"))
        assert cycles[1] == (ts("2026-08-03"), ts("2026-08-28"))
        assert cycles[2] == (ts("2026-08-31"), ts("2026-09-25"))
        assert cycles[3][0] == ts("2026-09-28")

    def test_anchor_not_on_buy_weekday_rolls_forward(self):
        # 錨點設在週四 7/2 → 第一個買入日滾到下週一 7/6
        cycles = compute_cycles("2026-07-02", 0, 4, 4, ts("2026-08-01"))
        assert cycles[0][0] == ts("2026-07-06")

    def test_hold_one_week(self):
        cycles = compute_cycles("2026-07-06", 0, 4, 1, ts("2026-07-20"))
        assert cycles[0] == (ts("2026-07-06"), ts("2026-07-10"))
        assert cycles[1] == (ts("2026-07-13"), ts("2026-07-17"))

    def test_same_buy_sell_weekday(self):
        # 買賣同 weekday、hold 1 週：賣出日 = 買入日當天（>= 語意），下一買入日 +7
        cycles = compute_cycles("2026-07-06", 0, 0, 1, ts("2026-07-20"))
        assert cycles[0] == (ts("2026-07-06"), ts("2026-07-06"))
        assert cycles[1][0] == ts("2026-07-13")

    def test_no_gap_between_cycles(self):
        # 賣出週五 → 下週一再買，中間只隔週末
        cycles = kiri_cycles()
        for (_, exit_d), (next_entry, _) in zip(cycles, cycles[1:]):
            assert (next_entry - exit_d).days == 3

    def test_anchor_beyond_until_returns_empty(self):
        assert compute_cycles("2026-07-06", 0, 4, 4, ts("2026-07-01")) == []


class TestComputeHistoricalCycles:
    def _index(self, start, end):
        return pd.date_range(start, end, freq="D")

    def test_complete_cycles_before_boundary(self):
        # 4 月起的日曆日，boundary 在錨點 7/6：所有回傳週期必須整個在 7/6 前結束
        index = self._index("2026-04-01", "2026-07-06")
        cycles = compute_historical_cycles(index, 0, 4, 4, before=ts("2026-07-06"))
        assert len(cycles) > 0
        for entry, exit_d in cycles:
            assert exit_d < ts("2026-07-06")
            assert entry.dayofweek == 0
            assert exit_d.dayofweek == 4
            assert (exit_d - entry).days == 25  # 週一到第 4 個週五

    def test_cycle_containing_today_is_excluded(self):
        # 2026-07-02 踩過的情境：歷史週期 6/8~7/3 包含「今天 7/2」，
        # boundary=min(錨點, 今天)=7/2 時必須被排除，否則今天會誤判為持倉中
        index = self._index("2026-06-01", "2026-07-02")
        cycles = compute_historical_cycles(index, 0, 4, 4, before=ts("2026-07-02"))
        for entry, exit_d in cycles:
            assert not (entry <= ts("2026-07-02") <= exit_d)

    def test_incomplete_trailing_cycle_dropped(self):
        # 資料不夠湊滿 hold_weeks 個賣出日的尾端週期要被丟棄
        index = self._index("2026-06-01", "2026-06-20")
        cycles = compute_historical_cycles(index, 0, 4, 4, before=ts("2026-12-31"))
        assert cycles == []

    def test_empty_index(self):
        index = pd.DatetimeIndex([])
        assert compute_historical_cycles(index, 0, 4, 4, before=ts("2026-07-06")) == []


class TestBuildTrancheSpecs:
    """2026-07-03 定案的 4-tranche 滾動制：同帳戶分 4 份錯開一週、各持有 4 週。"""

    def test_kiri_four_tranches(self):
        specs = build_tranche_specs("2026-07-06", 4, invest_ratio=0.7)
        assert [name for name, _, _ in specs] == [
            "tranche_1", "tranche_2", "tranche_3", "tranche_4"]
        assert [anchor for _, anchor, _ in specs] == [
            ts("2026-07-06"), ts("2026-07-13"), ts("2026-07-20"), ts("2026-07-27")]
        for _, _, weight in specs:
            assert weight == pytest.approx(0.175)

    def test_weights_sum_to_invest_ratio(self):
        specs = build_tranche_specs("2026-07-06", 4, invest_ratio=0.7)
        assert sum(w for _, _, w in specs) == pytest.approx(0.7)

    def test_single_tranche_defaults_to_full_weight(self):
        # hold_weeks=1 的舊單 cycle 設定退化為單一 tranche、權重 = invest_ratio 預設 1.0
        assert build_tranche_specs("2026-07-06", 1) == [
            ("tranche_1", ts("2026-07-06"), 1.0)]

    def test_tranche_cycles_cover_every_monday_seamlessly(self):
        # 4 tranches × 28 天週期 → 聯集恰好每週一各有一個 tranche 進場、不重不漏
        specs = build_tranche_specs("2026-07-06", 4)
        entries = sorted(
            entry
            for _, anchor, _ in specs
            for entry, _ in compute_cycles(anchor, 0, 4, 4, ts("2026-12-31"))
            if entry <= ts("2026-11-30")
        )
        expected = list(pd.date_range("2026-07-06", "2026-11-30", freq="W-MON"))
        assert entries == expected


class TestFindCurrentCycle:
    def test_replay_verified_dates(self):
        """對應 2026-07-02 容器實測的四個日期。"""
        cycles = kiri_cycles()
        assert find_current_cycle(cycles, ts("2026-07-02")) is None          # 錨點前
        assert find_current_cycle(cycles, ts("2026-07-13"))[0] == ts("2026-07-06")  # 第 2 週
        assert find_current_cycle(cycles, ts("2026-07-31"))[1] == ts("2026-07-31")  # 賣出日當天（含）
        assert find_current_cycle(cycles, ts("2026-08-01")) is None          # 週期交界週末

    def test_entry_day_inclusive(self):
        cycles = kiri_cycles()
        assert find_current_cycle(cycles, ts("2026-07-06"))[0] == ts("2026-07-06")


class TestAlignToSunday:
    """清單日對齊規則的唯一實作——`_create_df` 的 weekly_batches 鍵、`owning_sunday`
    的反向、節點制的「哪些週日真的有清單」都是它，算錯三處一起錯。"""

    def test_sunday_stays(self):
        assert align_to_sunday(ts("2026-07-05")) == ts("2026-07-05")

    def test_weekdays_align_to_next_sunday(self):
        for d in ("2026-06-29", "2026-07-01", "2026-07-04"):  # 一、三、六
            assert align_to_sunday(ts(d)) == ts("2026-07-05")

    @pytest.mark.parametrize('raw, aligned', [
        ('2026-08-09', '2026-08-09'),   # 週日當天產出 → 留在當天
        ('2026-08-10', '2026-08-16'),   # 週一 → 下一個週日
        ('2026-08-14', '2026-08-16'),   # 週五 → 下一個週日
        ('2026-08-15', '2026-08-16'),   # 週六 → 隔天
    ])
    def test_it_takes_the_date_string_straight_from_the_db(self, raw, aligned):
        """呼叫端多半直接餵 `record.date`（str），不該逼每一處自己先轉。"""
        assert align_to_sunday(raw) == pd.Timestamp(aligned)

    def test_a_timestamp_works_too(self):
        assert align_to_sunday(pd.Timestamp('2026-08-10')) == pd.Timestamp('2026-08-16')


class TestCheckRecommendationFreshness:
    def test_outside_any_cycle_skips_check(self):
        # 錨點前就算 DB 全空也不該擋（本來就不會下單）
        check_recommendation_freshness(kiri_cycles(), ts("2026-07-02"), None)

    def test_fresh_list_passes(self):
        # 7/6 進場，7/5（上週日）的清單 → 通過
        check_recommendation_freshness(kiri_cycles(), ts("2026-07-06"), "2026-07-05")

    def test_newer_midcycle_list_passes(self):
        # 週期中 DB 出現更新的清單（7/12）不該誤判為過期
        check_recommendation_freshness(kiri_cycles(), ts("2026-07-13"), "2026-07-12")

    def test_a_stale_list_only_warns(self, caplog):
        """過期清單不再擋下單：`_create_df` 已經把缺清單那週歸零，這一份 tranche
        自己空手就好。拋出去會連帶擋掉當天其他 tranche 的賣出。"""
        # 2026-07-02 容器實測過的情境：7/13 重放時 DB 最新只有 6/28
        with caplog.at_level(logging.WARNING, logger="core.trading_cycles"):
            check_recommendation_freshness(kiri_cycles(), ts("2026-07-13"), "2026-06-28")
        assert "推薦清單過期" in caplog.text

    def test_an_entry_day_with_the_previous_weeks_list_only_warns(self, caplog):
        """進場日當天清單還是上上週的——最該空手的一天，也不該拋。"""
        with caplog.at_level(logging.WARNING, logger="core.trading_cycles"):
            check_recommendation_freshness(kiri_cycles(), ts("2026-07-06"), "2026-06-28")
        assert "推薦清單過期" in caplog.text

    def test_an_empty_db_still_raises(self):
        """一份清單都沒有時 `_create_df` 連 position 都建不起來，讓它講清楚再死。
        這種情況不可能有持股，擋下來沒有賣不掉的代價。"""
        with pytest.raises(RuntimeError, match="DB 無推薦清單"):
            check_recommendation_freshness(kiri_cycles(), ts("2026-07-06"), None)


class TestOwningSunday:
    """每一天用的是哪一份清單。`_create_df` 靠它找出「那週根本沒清單」的日子。"""

    def test_a_sunday_owns_itself(self):
        assert list(owning_sunday(['2026-01-11'])) == [pd.Timestamp('2026-01-11')]

    def test_the_whole_week_after_a_sunday_belongs_to_it(self):
        week = pd.date_range('2026-01-11', '2026-01-17', freq='D')   # 日 ~ 六
        assert set(owning_sunday(week)) == {pd.Timestamp('2026-01-11')}

    def test_the_next_sunday_starts_a_new_owner(self):
        assert owning_sunday(['2026-01-18'])[0] == pd.Timestamp('2026-01-18')

    def test_it_is_the_inverse_of_align_to_sunday(self):
        """align_to_sunday 把清單日推到它生效的那個週日；這支把任一天推回它的清單日。"""
        for d in pd.date_range('2026-01-04', '2026-02-15', freq='D'):
            sunday = owning_sunday([d])[0]
            assert sunday.weekday() == 6
            assert sunday <= d < sunday + pd.Timedelta(days=7)

    def test_an_empty_input_gives_an_empty_index(self):
        assert len(owning_sunday(pd.DatetimeIndex([]))) == 0


class TestMissingListSunday:
    """當期該用的週日沒有清單——freshness 檢查看不到的那種缺席。

    kiri 的排程：週一買、hold 4 週，第一個週期 2026-07-06 ~ 07-31，
    該用的是 07-05（週日）的清單。
    """

    def test_a_present_list_is_not_reported(self):
        assert missing_list_sunday(kiri_cycles(), ts("2026-07-06"),
                                   [ts("2026-07-05")]) is None

    def test_a_missing_list_is_reported_on_the_entry_day(self):
        assert missing_list_sunday(kiri_cycles(), ts("2026-07-06"),
                                   [ts("2026-06-28")]) == ts("2026-07-05")

    def test_it_keeps_reporting_for_the_rest_of_that_week(self):
        """那一週清單還補得進來，補了就會進場，所以整週都該喊。"""
        for d in ("2026-07-07", "2026-07-09", "2026-07-11"):   # 二、四、六
            assert missing_list_sunday(kiri_cycles(), ts(d), [ts("2026-06-28")]) == ts("2026-07-05")

    def test_it_goes_quiet_once_that_week_is_over(self):
        """過了那週該輪確定空手，每天再喊也改變不了。"""
        assert missing_list_sunday(kiri_cycles(), ts("2026-07-13"), [ts("2026-06-28")]) is None

    def test_outside_any_cycle_is_not_reported(self):
        assert missing_list_sunday(kiri_cycles(), ts("2026-07-02"), []) is None

    def test_it_catches_exactly_what_freshness_misses(self):
        """實盤四次缺清單的共同形狀：清單在進場日當天（週一）才發。

        對齊規則把它推到**下一個**週日，於是 freshness 通過（07-12 >= 07-05），
        但當期該用的 07-05 仍然不存在——那一輪空手。
        """
        monday_list = "2026-07-06"
        check_recommendation_freshness(kiri_cycles(), ts("2026-07-06"), monday_list)   # 不拋
        assert missing_list_sunday(kiri_cycles(), ts("2026-07-06"),
                                   [align_to_sunday(ts(monday_list))]) == ts("2026-07-05")

    def test_an_empty_db_is_reported_too(self):
        assert missing_list_sunday(kiri_cycles(), ts("2026-07-06"), []) == ts("2026-07-05")

    def test_it_accepts_plain_strings_as_list_sundays(self):
        assert missing_list_sunday(kiri_cycles(), ts("2026-07-06"), ["2026-07-05"]) is None
