"""RecommendationDAO 測試（真實 SQLite on tmp file，不用 mock）。"""

import pytest

from dao.recommendation_dao import RecommendationDAO, RecommendationRecord, Stock


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


def _record(date, ids):
    return RecommendationRecord(
        date=date,
        stocks=[Stock(id=i, sentiment="BUY") for i in ids],
    )


def test_add_and_get_latest_roundtrip(db_path):
    dao = RecommendationDAO(db_path=db_path, frequency="weekly")
    dao.add_record(_record("2026-06-28", ["2337", "2382", "2330"]))

    latest = dao.get_latest()
    assert latest.date == "2026-06-28"
    # priority 順序必須保留（rank 依賴這個）
    assert [s.id for s in latest.stocks] == ["2337", "2382", "2330"]


def test_get_latest_picks_newest_date(db_path):
    dao = RecommendationDAO(db_path=db_path, frequency="weekly")
    dao.add_record(_record("2026-06-21", ["1101"]))
    dao.add_record(_record("2026-06-28", ["2330"]))
    assert dao.get_latest().date == "2026-06-28"


def test_add_record_same_date_replaces(db_path):
    dao = RecommendationDAO(db_path=db_path, frequency="weekly")
    dao.add_record(_record("2026-06-28", ["1101", "1102"]))
    dao.add_record(_record("2026-06-28", ["2330"]))

    record = dao.get_by_date("2026-06-28")
    assert [s.id for s in record.stocks] == ["2330"]


def test_frequency_isolation(db_path):
    weekly = RecommendationDAO(db_path=db_path, frequency="weekly")
    monthly = RecommendationDAO(db_path=db_path, frequency="monthly")
    weekly.add_record(_record("2026-06-28", ["2330"]))
    monthly.add_record(_record("2026-06-01", ["2317"]))

    assert weekly.get_latest().date == "2026-06-28"
    assert monthly.get_latest().date == "2026-06-01"
    assert [r.date for r in weekly.load()] == ["2026-06-28"]


def test_load_sorted_by_date(db_path):
    dao = RecommendationDAO(db_path=db_path, frequency="weekly")
    for d in ("2026-06-28", "2026-06-14", "2026-06-21"):
        dao.add_record(_record(d, ["2330"]))
    assert [r.date for r in dao.load()] == ["2026-06-14", "2026-06-21", "2026-06-28"]


def test_empty_db_returns_none_and_empty(db_path):
    dao = RecommendationDAO(db_path=db_path, frequency="weekly")
    assert dao.get_latest() is None
    assert dao.get_by_date("2026-06-28") is None
    assert dao.load() == []
    assert dao.get_stock_ids("2026-06-28") == []


def test_add_record_requires_frequency(db_path):
    dao = RecommendationDAO(db_path=db_path, frequency=None)
    with pytest.raises(ValueError):
        dao.add_record(_record("2026-06-28", ["2330"]))


def test_sl_tp_persisted(db_path):
    dao = RecommendationDAO(db_path=db_path, frequency="weekly")
    dao.add_record(RecommendationRecord(
        date="2026-06-28",
        stocks=[Stock(id="2330", sentiment="STRONG_BUY", TP=1200.0, SL=950.0, name="台積電")],
    ))
    stock = dao.get_latest().stocks[0]
    assert stock.TP == 1200.0
    assert stock.SL == 950.0
    assert stock.name == "台積電"
