"""GoldenAI 回測 metrics/reports DAO 的欄位改名 migration。

`week`（Week1~4，當月第幾個週日）在相位改由錨點連續輪動定義之後沒有日曆語意了，
改名為 `tranche`。既有的正式機 DB 有幾萬列，migration 必須是純 metadata 操作、
不動資料，且重複執行不出事。
"""

import sqlite3

import pytest

from dao.golden_ai_backtest_metrics_dao import GoldenAIBacktestMetricsDAO

TABLES = ('golden_ai_backtest_metrics', 'golden_ai_backtest_reports')

OLD_SCHEMA = """
    CREATE TABLE golden_ai_backtest_metrics (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp     TEXT NOT NULL,
        strategy      TEXT NOT NULL,
        week          TEXT,
        ranks         TEXT NOT NULL DEFAULT '',
        annual_return REAL,
        sharpe        REAL,
        sortino       REAL,
        max_drawdown  REAL,
        win_ratio     REAL
    );
    CREATE TABLE golden_ai_backtest_reports (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp     TEXT NOT NULL,
        strategy      TEXT NOT NULL,
        week          TEXT,
        ranks         TEXT NOT NULL DEFAULT '',
        report_json   TEXT NOT NULL,
        position_json TEXT NOT NULL
    );
"""


class FakeReport:
    """finlab `Report.get_metrics()` 的巢狀結構，只保留 DAO 會讀的五個值。"""

    def get_metrics(self):
        return {
            'profitability': {'annualReturn': 0.12},
            'ratio': {'sharpeRatio': 1.1, 'sortinoRatio': 1.4},
            'risk': {'maxDrawdown': -0.08},
            'winrate': {'winRate': 0.6},
        }


@pytest.fixture
def legacy_db(tmp_path):
    """一份舊 schema、且已經有資料的 DB。"""
    path = str(tmp_path / 'legacy.db')
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    conn.execute("INSERT INTO golden_ai_backtest_metrics "
                 "(timestamp, strategy, week, ranks, annual_return) VALUES (?, ?, ?, ?, ?)",
                 ('2026-06-17 22:45:00', 'weekly_4w', 'Week2', '1,2,3', 0.21))
    conn.execute("INSERT INTO golden_ai_backtest_metrics "
                 "(timestamp, strategy, week, ranks, annual_return) VALUES (?, ?, ?, ?, ?)",
                 ('2026-06-17 22:35:00', 'weekly', None, '1,2,3', 0.09))
    conn.execute("INSERT INTO golden_ai_backtest_reports "
                 "(timestamp, strategy, week, ranks, report_json, position_json) "
                 "VALUES (?, ?, ?, ?, ?, ?)",
                 ('2026-06-17 22:45:00', 'weekly_4w', 'Week2', '1,2,3', '{}', '{}'))
    conn.commit()
    conn.close()
    return path


def _columns(path, table):
    conn = sqlite3.connect(path)
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


@pytest.mark.parametrize('table', TABLES)
def test_week_is_renamed_to_tranche(legacy_db, table):
    assert 'week' in _columns(legacy_db, table)
    GoldenAIBacktestMetricsDAO(db_path=legacy_db)
    cols = _columns(legacy_db, table)
    assert 'tranche' in cols and 'week' not in cols


def test_existing_rows_survive_untouched(legacy_db):
    GoldenAIBacktestMetricsDAO(db_path=legacy_db)
    conn = sqlite3.connect(legacy_db)
    rows = conn.execute("SELECT strategy, tranche, annual_return FROM golden_ai_backtest_metrics "
                        "ORDER BY strategy").fetchall()
    conn.close()
    # 改名不動值：舊的 Week2 仍是 Week2（要換成 tranche2 得靠重跑），weekly 仍是 NULL
    assert rows == [('weekly', None, 0.09), ('weekly_4w', 'Week2', 0.21)]


def test_migration_is_idempotent(legacy_db):
    GoldenAIBacktestMetricsDAO(db_path=legacy_db)
    before = {t: _columns(legacy_db, t) for t in TABLES}
    GoldenAIBacktestMetricsDAO(db_path=legacy_db)
    assert {t: _columns(legacy_db, t) for t in TABLES} == before


def test_an_ancient_db_with_top_n_migrates_in_the_right_order(tmp_path):
    """`top_n` 那支 migration 是整表重建、搬運時把欄位名寫死。

    它的 INSERT/SELECT 已經改用 `tranche`，所以來源表必須先被改名——
    也就是 week->tranche 那支得跑在它前面，否則會 `no such column: tranche`。
    """
    path = str(tmp_path / 'ancient.db')
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE golden_ai_backtest_metrics (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            strategy      TEXT NOT NULL,
            week          TEXT,
            top_n         INTEGER,
            ranks         TEXT NOT NULL DEFAULT '',
            annual_return REAL,
            sharpe        REAL,
            sortino       REAL,
            max_drawdown  REAL,
            win_ratio     REAL
        )
    """)
    conn.execute("INSERT INTO golden_ai_backtest_metrics "
                 "(timestamp, strategy, week, top_n, annual_return) VALUES (?, ?, ?, ?, ?)",
                 ('2026-01-05 22:45:00', 'monthly', 'Week3', 5, 0.33))
    conn.commit()
    conn.close()

    GoldenAIBacktestMetricsDAO(db_path=path)

    cols = _columns(path, 'golden_ai_backtest_metrics')
    assert 'tranche' in cols
    assert 'week' not in cols and 'top_n' not in cols

    conn = sqlite3.connect(path)
    try:
        # top_n 被折進 ranks（原本的行為），tranche 的值原樣搬過來
        assert conn.execute("SELECT strategy, tranche, ranks, annual_return "
                            "FROM golden_ai_backtest_metrics").fetchall() == [
            ('monthly', 'Week3', '5', 0.33)]
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE '%_old'").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_fresh_db_is_created_with_tranche(tmp_path):
    path = str(tmp_path / 'fresh.db')
    GoldenAIBacktestMetricsDAO(db_path=path)
    for table in TABLES:
        cols = _columns(path, table)
        assert 'tranche' in cols and 'week' not in cols


def test_save_and_load_round_trip_on_the_renamed_column(tmp_path):
    path = str(tmp_path / 'fresh.db')
    dao = GoldenAIBacktestMetricsDAO(db_path=path)
    dao.save(timestamp='2026-08-22 22:45:00', strategy='weekly_4w',
             tranche='tranche3', ranks='1,2,3', report=FakeReport())
    dao.save(timestamp='2026-08-22 22:35:00', strategy='weekly',
             tranche=None, ranks='1,2,3', report=FakeReport())

    df = dao.load(strategy='weekly_4w', tranche='tranche3')
    assert len(df) == 1
    assert df.iloc[0]['tranche'] == 'tranche3'
    assert df.iloc[0]['sharpe'] == pytest.approx(1.1)
    assert dao.load(strategy='weekly_4w', tranche='tranche1').empty


def test_report_json_is_addressed_by_tranche(tmp_path):
    path = str(tmp_path / 'fresh.db')
    dao = GoldenAIBacktestMetricsDAO(db_path=path)
    dao.save_report(timestamp='2026-08-22 22:45:00', strategy='monthly',
                    tranche='tranche4', ranks='1,2', report_json='{"a":1}',
                    position_json='{"b":2}')

    assert dao.get_report('2026-08-22 22:45:00', 'monthly',
                          tranche='tranche4', ranks='1,2') == ('{"a":1}', '{"b":2}')
    assert dao.get_report('2026-08-22 22:45:00', 'monthly',
                          tranche='tranche1', ranks='1,2') is None
    assert list(dao.list_reports('monthly')['tranche']) == ['tranche4']
