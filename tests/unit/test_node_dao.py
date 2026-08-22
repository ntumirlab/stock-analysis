"""GoldenAIBacktestNodesDAO 測試（真實 SQLite on tmp file）。"""

import pytest

from dao.golden_ai_backtest_nodes_dao import GoldenAIBacktestNodesDAO, extract_metrics


@pytest.fixture
def dao(tmp_path):
    return GoldenAIBacktestNodesDAO(db_path=str(tmp_path / "test.db"))


NODE = {
    "strategy": "weekly",
    "list_date": "2026-07-05",
    "ranks": "1,2,3,4,5,6,7,8",
    "entry_date": "2026-07-06",
    "exit_date": "2026-07-13",
    "n_stocks": 7,
    "node_return": -0.059837,
}


class FakeReport:
    def __init__(self, metrics=None, raises=False):
        self._metrics = metrics
        self._raises = raises

    def get_metrics(self):
        if self._raises:
            raise RuntimeError("finlab blew up")
        return self._metrics


FULL_METRICS = {
    "profitability": {"annualReturn": -0.9563},
    "ratio": {"sharpeRatio": -2.1, "sortinoRatio": -3.2},
    "risk": {"maxDrawdown": -0.0764},
    "winrate": {"winRate": 0.2857},
}


def test_save_and_load_roundtrip(dao):
    assert dao.save(**NODE, report=FakeReport(FULL_METRICS)) is True

    df = dao.load(strategy="weekly")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["list_date"] == "2026-07-05"
    assert row["entry_date"] == "2026-07-06"
    assert row["exit_date"] == "2026-07-13"
    assert row["n_stocks"] == 7
    assert row["node_return"] == pytest.approx(-0.059837)
    assert row["annual_return"] == pytest.approx(-0.9563)
    assert row["win_ratio"] == pytest.approx(0.2857)
    assert row["created_at"]


def test_same_node_is_not_written_twice(dao):
    assert dao.save(**NODE) is True
    assert dao.save(**NODE) is False

    assert len(dao.load()) == 1


def test_other_ranks_on_the_same_list_coexist(dao):
    dao.save(**NODE)
    dao.save(**{**NODE, "ranks": "1,2,3", "n_stocks": 3, "node_return": -0.0518})

    df = dao.load(strategy="weekly")
    assert len(df) == 2
    assert set(df["ranks"]) == {"1,2,3,4,5,6,7,8", "1,2,3"}


def test_load_filters_and_orders_by_exit_date(dao):
    dao.save(**{**NODE, "list_date": "2026-07-12", "entry_date": "2026-07-13",
                "exit_date": "2026-07-17"})
    dao.save(**NODE)
    dao.save(**{**NODE, "strategy": "monthly", "week_of_month": 1})

    df = dao.load(strategy="weekly", ranks="1,2,3,4,5,6,7,8")
    assert list(df["exit_date"]) == ["2026-07-13", "2026-07-17"]

    monthly = dao.load(strategy="monthly")
    assert len(monthly) == 1
    assert monthly.iloc[0]["week_of_month"] == 1


def test_exists_checks_the_node_identity(dao):
    dao.save(**NODE)

    assert dao.exists("weekly", "2026-07-05", "1,2,3,4,5,6,7,8") is True
    assert dao.exists("weekly", "2026-07-05", "1,2,3") is False
    assert dao.exists("monthly", "2026-07-05", "1,2,3,4,5,6,7,8") is False


def test_load_on_empty_table_returns_empty_frame(dao):
    df = dao.load(strategy="weekly")
    assert df.empty


def test_metrics_are_null_without_a_report(dao):
    dao.save(**NODE)

    row = dao.load().iloc[0]
    assert row["annual_return"] is None
    assert row["sharpe"] is None
    # 節點自己的數字仍然要在
    assert row["node_return"] == pytest.approx(-0.059837)


def test_extract_metrics_survives_a_broken_report():
    assert extract_metrics(FakeReport(raises=True)) == {
        "annual_return": None, "sharpe": None, "sortino": None,
        "max_drawdown": None, "win_ratio": None,
    }


def test_extract_metrics_tolerates_missing_sections():
    assert extract_metrics(FakeReport({"profitability": {"annualReturn": 0.5}})) == {
        "annual_return": 0.5, "sharpe": None, "sortino": None,
        "max_drawdown": None, "win_ratio": None,
    }
