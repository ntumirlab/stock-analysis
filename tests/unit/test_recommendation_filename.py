"""檔名解析測試——上游（金策智能）命名格式已變更過多次，是已知回歸熱點。"""

from core.recommendation_parsing import extract_recommendation_date


class TestNewFormat:
    """新格式（2026-06-28 起）：{YYYYMMDD}_{HHMMSS}_tw_{N}d_recommendation.md"""

    def test_weekly(self):
        assert extract_recommendation_date(
            "20260628_061019_tw_5d_recommendation.md", "weekly") == "2026-06-28"

    def test_monthly(self):
        assert extract_recommendation_date(
            "20260601_120000_tw_30d_recommendation.md", "monthly") == "2026-06-01"

    def test_days_mismatch_returns_none(self):
        # weekly task 拿到 30d 檔案要跳過
        assert extract_recommendation_date(
            "20260628_061019_tw_30d_recommendation.md", "weekly") is None
        assert extract_recommendation_date(
            "20260628_061019_tw_5d_recommendation.md", "monthly") is None


class TestOldFormat:
    """舊格式：{YYYYMMDD}_{HHMM}_推薦股票_台股{N}日_金策智能.md"""

    def test_weekly(self):
        assert extract_recommendation_date(
            "20260628_1430_推薦股票_台股5日_金策智能.md", "weekly") == "2026-06-28"

    def test_monthly(self):
        assert extract_recommendation_date(
            "20260601_0900_推薦股票_台股30日_金策智能.md", "monthly") == "2026-06-01"

    def test_days_mismatch_returns_none(self):
        assert extract_recommendation_date(
            "20260628_1430_推薦股票_台股30日_金策智能.md", "weekly") is None


class TestRejection:
    def test_unrelated_filename(self):
        assert extract_recommendation_date("readme.md", "weekly") is None

    def test_not_md_extension(self):
        assert extract_recommendation_date(
            "20260628_061019_tw_5d_recommendation.txt", "weekly") is None

    def test_invalid_date(self):
        # 2026-13-99 不是合法日期
        assert extract_recommendation_date(
            "20261399_061019_tw_5d_recommendation.md", "weekly") is None

    def test_old_format_wrong_time_digits(self):
        # 舊格式時間是 4 碼，給 6 碼不該匹配
        assert extract_recommendation_date(
            "20260628_143000_推薦股票_台股5日_金策智能.md", "weekly") is None


class TestUnknownTask:
    def test_unknown_task_skips_days_check(self):
        # 目前行為：task 不在 TASK_EXPECTED_DAYS 裡就不過濾天數（文件化既有行為）
        assert extract_recommendation_date(
            "20260628_061019_tw_99d_recommendation.md", "sometask") == "2026-06-28"
