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
    dao.save(**{**NODE, "strategy": "monthly", "tranche": 1})

    df = dao.load(strategy="weekly", ranks="1,2,3,4,5,6,7,8")
    assert list(df["exit_date"]) == ["2026-07-13", "2026-07-17"]

    monthly = dao.load(strategy="monthly")
    assert len(monthly) == 1
    assert monthly.iloc[0]["tranche"] == 1


def test_stored_list_dates_is_scoped_to_strategy_and_ranks(dao):
    dao.save(**NODE)
    dao.save(**{**NODE, "list_date": "2026-07-12", "exit_date": "2026-07-17"})
    dao.save(**{**NODE, "ranks": "1,2,3"})
    dao.save(**{**NODE, "strategy": "monthly"})

    assert dao.stored_list_dates("weekly", "1,2,3,4,5,6,7,8") == {
        "2026-07-05", "2026-07-12"}
    assert dao.stored_list_dates("weekly", "1,2,3") == {"2026-07-05"}
    assert dao.stored_list_dates("monthly", "1,2,3,4,5,6,7,8") == {"2026-07-05"}


def test_stored_list_dates_is_empty_for_an_unknown_combination(dao):
    dao.save(**NODE)

    assert dao.stored_list_dates("weekly", "4,5,6") == set()
    assert dao.stored_list_dates("weekly_4w", "1,2,3,4,5,6,7,8") == set()


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


# ── week_of_month -> tranche migration ──

OLD_NODES_SCHEMA = """
    CREATE TABLE golden_ai_backtest_nodes (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy      TEXT NOT NULL,
        list_date     TEXT NOT NULL,
        ranks         TEXT NOT NULL,
        entry_date    TEXT NOT NULL,
        exit_date     TEXT NOT NULL,
        week_of_month INTEGER,
        n_stocks      INTEGER NOT NULL,
        node_return   REAL NOT NULL,
        annual_return REAL,
        sharpe        REAL,
        sortino       REAL,
        max_drawdown  REAL,
        win_ratio     REAL,
        created_at    TEXT NOT NULL
    );
"""

# (strategy, list_date, 舊的 week_of_month, 新制正確的 tranche)
# 舊值＝當月第幾個週日，新值＝距相位原點幾週取模。三個都不是隨手編的，是照定義算出來的。
# 兩支 4 週策略共用同一個原點（見 `core.tranche_schedule`），所以同一個 list_date
# 的新標籤必須一致——最後兩列刻意用同一天，位移一格就會被抓到。
LEGACY_NODES = [
    ("weekly_4w", "2026-07-19", 3, 3),   # 這期舊新湊巧相同，也必須維持正確
    ("weekly_4w", "2026-05-03", 1, 4),
    ("monthly",   "2026-05-03", 1, 4),
]


@pytest.fixture
def legacy_nodes_db(tmp_path):
    """舊 schema、且已經有回填資料的節點表（正式機 2026-08 的狀態）。"""
    import sqlite3

    path = str(tmp_path / "legacy_nodes.db")
    conn = sqlite3.connect(path)
    conn.executescript(OLD_NODES_SCHEMA)
    for strategy, list_date, week_of_month, _ in LEGACY_NODES:
        for ranks in ("1,2", "1,2,3"):
            conn.execute(
                "INSERT INTO golden_ai_backtest_nodes (strategy, list_date, ranks, "
                "entry_date, exit_date, week_of_month, n_stocks, node_return, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (strategy, list_date, ranks, "2026-07-20", "2026-08-14",
                 week_of_month, 2, 0.01, "2026-08-23 23:20:00"),
            )
    # weekly 沒有相位，這一列必須全程不被碰到
    conn.execute(
        "INSERT INTO golden_ai_backtest_nodes (strategy, list_date, ranks, "
        "entry_date, exit_date, week_of_month, n_stocks, node_return, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("weekly", "2026-07-19", "1,2", "2026-07-20", "2026-07-24",
         None, 2, 0.02, "2026-08-23 23:10:00"),
    )
    conn.commit()
    conn.close()
    return path


def _nodes_columns(path):
    import sqlite3

    conn = sqlite3.connect(path)
    try:
        return {r[1] for r in conn.execute("PRAGMA table_info(golden_ai_backtest_nodes)")}
    finally:
        conn.close()


def _tranche(path, strategy, list_date):
    import sqlite3

    conn = sqlite3.connect(path)
    try:
        return {
            r[0] for r in conn.execute(
                "SELECT tranche FROM golden_ai_backtest_nodes "
                "WHERE strategy = ? AND list_date = ?", (strategy, list_date))
        }
    finally:
        conn.close()


def test_week_of_month_is_renamed_to_tranche(legacy_nodes_db):
    assert "week_of_month" in _nodes_columns(legacy_nodes_db)
    GoldenAIBacktestNodesDAO(db_path=legacy_nodes_db)
    cols = _nodes_columns(legacy_nodes_db)
    assert "tranche" in cols and "week_of_month" not in cols


@pytest.mark.parametrize("strategy, list_date, old, new", LEGACY_NODES)
def test_legacy_labels_are_recomputed_not_just_renamed(
        legacy_nodes_db, strategy, list_date, old, new):
    """節點的數字與排程無關，`tranche` 是唯一失真的欄位——就地換算，不必重跑回測。"""
    GoldenAIBacktestNodesDAO(db_path=legacy_nodes_db)
    assert _tranche(legacy_nodes_db, strategy, list_date) == {new}


def test_weekly_rows_keep_a_null_phase(legacy_nodes_db):
    """weekly 一週一輪、沒有相位，重算不能把 NULL 填成數字。"""
    GoldenAIBacktestNodesDAO(db_path=legacy_nodes_db)
    assert _tranche(legacy_nodes_db, "weekly", "2026-07-19") == {None}


def test_node_numbers_are_untouched_by_the_migration(legacy_nodes_db):
    import sqlite3

    GoldenAIBacktestNodesDAO(db_path=legacy_nodes_db)
    conn = sqlite3.connect(legacy_nodes_db)
    try:
        rows = conn.execute(
            "SELECT COUNT(*), SUM(node_return), COUNT(DISTINCT exit_date) "
            "FROM golden_ai_backtest_nodes").fetchone()
    finally:
        conn.close()
    assert rows[0] == 7
    assert rows[1] == pytest.approx(0.01 * 6 + 0.02)


def test_migration_is_idempotent(legacy_nodes_db):
    GoldenAIBacktestNodesDAO(db_path=legacy_nodes_db)
    first = {(s, d): _tranche(legacy_nodes_db, s, d) for s, d, _, _ in LEGACY_NODES}
    GoldenAIBacktestNodesDAO(db_path=legacy_nodes_db)
    assert {(s, d): _tranche(legacy_nodes_db, s, d) for s, d, _, _ in LEGACY_NODES} == first


def test_a_failed_relabel_takes_the_rename_down_with_it(legacy_nodes_db, monkeypatch):
    """改標失敗時改名必須跟著退回，否則閘門關上、錯的標籤再也補不回來。

    sqlite3 的 DDL 是 autocommit 跑的，所以這個保證只在呼叫端自己開 transaction 時
    才成立——這支測試守的就是那句 `BEGIN IMMEDIATE`。
    """
    import sqlite3

    def boom(cursor):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(GoldenAIBacktestNodesDAO, "_relabel_tranches", staticmethod(boom))
    with pytest.raises(sqlite3.OperationalError):
        GoldenAIBacktestNodesDAO(db_path=legacy_nodes_db)
    assert "week_of_month" in _nodes_columns(legacy_nodes_db)
    assert "tranche" not in _nodes_columns(legacy_nodes_db)

    # 改名沒發生，閘門就還開著：下次啟動照樣會改名並補上標籤
    monkeypatch.undo()
    GoldenAIBacktestNodesDAO(db_path=legacy_nodes_db)
    assert "tranche" in _nodes_columns(legacy_nodes_db)
    assert _tranche(legacy_nodes_db, "monthly", "2026-05-03") == {4}


def test_an_already_migrated_db_is_opened_without_taking_the_write_lock(legacy_nodes_db):
    """migration 跑完之後再開這個 DAO，不該再去搶寫鎖。

    `golden_ai_backtest_dashboard` 在 module import 就建這個 DAO；節點回填正在寫的
    時候若白搶一次鎖，會被擋滿 30 秒 busy timeout 然後整個 import 炸掉。
    """
    import sqlite3

    GoldenAIBacktestNodesDAO(db_path=legacy_nodes_db)   # 先把 migration 跑完

    writer = sqlite3.connect(legacy_nodes_db, timeout=1)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("UPDATE golden_ai_backtest_nodes SET n_stocks = 3")
    try:
        GoldenAIBacktestNodesDAO(db_path=legacy_nodes_db)   # 卡住就是回歸
    finally:
        writer.rollback()
        writer.close()


def test_a_fresh_db_is_created_with_tranche(tmp_path):
    path = str(tmp_path / "fresh_nodes.db")
    GoldenAIBacktestNodesDAO(db_path=path)
    cols = _nodes_columns(path)
    assert "tranche" in cols and "week_of_month" not in cols


def test_a_strategy_without_an_anchor_is_left_unlabelled_instead_of_blowing_up(tmp_path):
    """沒有錨點的策略若帶著相位值出現，不該讓整個 DAO 建構失敗——但也不能留著舊標籤。

    改名是一次性閘門，這一輪之後不會再有人來補。留著月份索引就是讓所有讀的人
    把它當錨點相位讀；清成 NULL 至少是誠實的「沒有標籤」。
    """
    import sqlite3

    path = str(tmp_path / "odd.db")
    conn = sqlite3.connect(path)
    conn.executescript(OLD_NODES_SCHEMA)
    conn.execute(
        "INSERT INTO golden_ai_backtest_nodes (strategy, list_date, ranks, "
        "entry_date, exit_date, week_of_month, n_stocks, node_return, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("some_future_strategy", "2026-07-19", "1,2", "2026-07-20", "2026-08-14",
         2, 2, 0.01, "2026-08-23 23:20:00"),
    )
    conn.commit()
    conn.close()

    GoldenAIBacktestNodesDAO(db_path=path)
    assert _tranche(path, "some_future_strategy", "2026-07-19") == {None}


def test_a_list_date_that_is_not_a_sunday_is_left_unlabelled(tmp_path):
    """相位是按「距離錨點幾週」算的，非週日會靜默算到隔壁的槽位。

    改名是一次性、不可逆的閘門——標錯之後 `week_of_month` 已經不在了，沒有人能發現，
    也沒有人能補。寧可標成不知道。
    """
    import sqlite3

    path = str(tmp_path / "unaligned.db")
    conn = sqlite3.connect(path)
    conn.executescript(OLD_NODES_SCHEMA)
    for list_date in ("2026-07-19", "2026-07-22"):        # 週日、週三
        conn.execute(
            "INSERT INTO golden_ai_backtest_nodes (strategy, list_date, ranks, "
            "entry_date, exit_date, week_of_month, n_stocks, node_return, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("weekly_4w", list_date, "1,2", "2026-07-20", "2026-08-14",
             3, 2, 0.01, "2026-08-23 23:20:00"),
        )
    conn.commit()
    conn.close()

    GoldenAIBacktestNodesDAO(db_path=path)
    assert _tranche(path, "weekly_4w", "2026-07-22") == {None}     # 週三 → 不知道
    assert _tranche(path, "weekly_4w", "2026-07-19") == {3}        # 同批的週日照算


def test_unlabelling_an_unknown_strategy_does_not_touch_the_known_ones(legacy_nodes_db):
    """混在同一批裡時，沒錨點的被清成 NULL，有錨點的照樣算出新標籤。"""
    import sqlite3

    conn = sqlite3.connect(legacy_nodes_db)
    conn.execute(
        "INSERT INTO golden_ai_backtest_nodes (strategy, list_date, ranks, "
        "entry_date, exit_date, week_of_month, n_stocks, node_return, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("some_future_strategy", "2026-07-19", "1,2", "2026-07-20", "2026-08-14",
         5, 2, 0.01, "2026-08-23 23:20:00"),
    )
    conn.commit()
    conn.close()

    GoldenAIBacktestNodesDAO(db_path=legacy_nodes_db)
    assert _tranche(legacy_nodes_db, "some_future_strategy", "2026-07-19") == {None}
    assert _tranche(legacy_nodes_db, "monthly", "2026-05-03") == {4}
    assert _tranche(legacy_nodes_db, "weekly", "2026-07-19") == {None}
