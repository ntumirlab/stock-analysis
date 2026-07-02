"""GoldenAI 下單週期日期運算的測試——實盤買賣日排程，改壞會直接影響真實下單。

案例來源：2026-07-02 錨點制修復時的重放驗證（含當時在容器中實測過的日期）。
weekday 慣例：pandas dayofweek，週一=0。kiri 實際設定 = 週一買(0)、週五賣(4)、hold 4 週。
"""

import pandas as pd
import pytest

from core.trading_cycles import (
    align_to_sunday,
    check_recommendation_freshness,
    compute_cycles,
    compute_historical_cycles,
    find_current_cycle,
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
    def test_sunday_stays(self):
        assert align_to_sunday(ts("2026-07-05")) == ts("2026-07-05")

    def test_weekdays_align_to_next_sunday(self):
        for d in ("2026-06-29", "2026-07-01", "2026-07-04"):  # 一、三、六
            assert align_to_sunday(ts(d)) == ts("2026-07-05")


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

    def test_stale_list_raises(self):
        # 2026-07-02 容器實測過的情境：7/13 重放時 DB 最新只有 6/28 → 擋下
        with pytest.raises(RuntimeError, match="推薦清單過期"):
            check_recommendation_freshness(kiri_cycles(), ts("2026-07-13"), "2026-06-28")

    def test_missing_list_raises(self):
        with pytest.raises(RuntimeError, match="DB 無推薦清單"):
            check_recommendation_freshness(kiri_cycles(), ts("2026-07-06"), None)

    def test_entry_day_with_previous_week_list_raises(self):
        # 進場日當天清單還是上上週的 → 不下單
        with pytest.raises(RuntimeError, match="推薦清單過期"):
            check_recommendation_freshness(kiri_cycles(), ts("2026-07-06"), "2026-06-28")
