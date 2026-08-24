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
    # 改名不動值：舊的 Week2 仍是 Week2（要換成 tranche_2 得靠重跑），weekly 仍是 NULL
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
             tranche='tranche_3', ranks='1,2,3', report=FakeReport())
    dao.save(timestamp='2026-08-22 22:35:00', strategy='weekly',
             tranche=None, ranks='1,2,3', report=FakeReport())

    df = dao.load(strategy='weekly_4w', tranche='tranche_3')
    assert len(df) == 1
    assert df.iloc[0]['tranche'] == 'tranche_3'
    assert df.iloc[0]['sharpe'] == pytest.approx(1.1)
    assert dao.load(strategy='weekly_4w', tranche='tranche_1').empty


def test_report_json_is_addressed_by_tranche(tmp_path):
    path = str(tmp_path / 'fresh.db')
    dao = GoldenAIBacktestMetricsDAO(db_path=path)
    dao.save_report(timestamp='2026-08-22 22:45:00', strategy='monthly',
                    tranche='tranche_4', ranks='1,2', report_json='{"a":1}',
                    position_json='{"b":2}')

    assert dao.get_report('2026-08-22 22:45:00', 'monthly',
                          tranche='tranche_4', ranks='1,2') == ('{"a":1}', '{"b":2}')
    assert dao.get_report('2026-08-22 22:45:00', 'monthly',
                          tranche='tranche_1', ranks='1,2') is None
    assert list(dao.list_reports('monthly')['tranche']) == ['tranche_4']


# ── 並行改名（部署時多個容器同時啟動）──

class _RacingCursor:
    """在 PRAGMA 與 ALTER 之間插隊改名的假 cursor——模擬另一個容器搶先做完。

    真正的競態視窗就在這兩句之間：`_rename_column_if_needed` 讀完欄位快照才發 ALTER，
    中間沒有鎖。用假 cursor 把插隊點釘死，比起真的開兩條 thread 賽跑穩定得多。
    """

    def __init__(self, cursor, db_path, sql):
        self._cursor = cursor
        self._db_path = db_path
        self._sql = sql
        self.raced = False

    def execute(self, sql, *args):
        if sql == self._sql and not self.raced:
            self.raced = True
            other = sqlite3.connect(self._db_path)
            other.execute(sql)
            other.commit()
            other.close()
        return self._cursor.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._cursor, name)


def test_losing_a_rename_race_is_not_an_error(legacy_db):
    """後手的 ALTER 會噴 no such column——先手已經做完了，確認結果對就放行。"""
    from dao.golden_ai_backtest_metrics_dao import _rename_column_if_needed

    conn = sqlite3.connect(legacy_db)
    try:
        cursor = _RacingCursor(
            conn.cursor(), legacy_db,
            "ALTER TABLE golden_ai_backtest_metrics RENAME COLUMN week TO tranche")
        renamed = _rename_column_if_needed(
            cursor, 'golden_ai_backtest_metrics', 'week', 'tranche')
        conn.commit()
    finally:
        conn.close()

    assert cursor.raced, '假 cursor 沒有真的插隊，這個測試就沒測到東西'
    # 沒改到名 -> 回 False -> 呼叫端不會重複做一次性的資料修補
    assert renamed is False
    assert 'tranche' in _columns(legacy_db, 'golden_ai_backtest_metrics')


def test_a_rename_that_fails_for_any_other_reason_still_raises():
    """只有「已經被改好了」才吞。欄位仍然不在就是真的壞了，要往外丟。"""
    from dao.golden_ai_backtest_metrics_dao import _rename_column_if_needed

    class _BrokenCursor:
        def execute(self, sql, *args):
            if sql.startswith('ALTER'):
                raise sqlite3.OperationalError('database is locked')
            self._rows = [(0, 'week', 'TEXT', 0, None, 0)]
            return self

        def fetchall(self):
            return self._rows

    with pytest.raises(sqlite3.OperationalError, match='locked'):
        _rename_column_if_needed(_BrokenCursor(), 'whatever', 'week', 'tranche')


# ── 寫到一半的殘缺組 ──

def _rows(path, table='golden_ai_backtest_metrics'):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


@pytest.fixture
def partial_db(tmp_path):
    """4 週策略那天只寫進 2 份 tranche 就掛了（例如 DB 鎖住）。"""
    path = str(tmp_path / 'partial.db')
    dao = GoldenAIBacktestMetricsDAO(db_path=path)
    for tranche in ('tranche_1', 'tranche_2'):
        dao.save(timestamp='2026-08-22 22:45:00', strategy='weekly_4w',
                 tranche=tranche, ranks='1,2,3', report=FakeReport())
        dao.save_report(timestamp='2026-08-22 22:45:00', strategy='weekly_4w',
                        tranche=tranche, ranks='1,2,3',
                        report_json='{}', position_json='{}')
    return path, dao


def test_a_partial_set_does_not_count_as_done(partial_db):
    """只問「有沒有」的話，缺的那兩份會被永久跳過而且沒人發現。"""
    path, dao = partial_db
    assert dao.exists_for_date('2026-08-22', 'weekly_4w', '1,2,3') is True     # 舊語意
    assert dao.exists_for_date('2026-08-22', 'weekly_4w', '1,2,3', expected=4) is False


def test_a_complete_set_counts_as_done(tmp_path):
    path = str(tmp_path / 'full.db')
    dao = GoldenAIBacktestMetricsDAO(db_path=path)
    for n in range(1, 5):
        dao.save(timestamp='2026-08-22 22:45:00', strategy='weekly_4w',
                 tranche=f'tranche_{n}', ranks='1,2,3', report=FakeReport())
    assert dao.exists_for_date('2026-08-22', 'weekly_4w', '1,2,3', expected=4) is True


def test_the_default_expectation_is_one_row(tmp_path):
    """weekly 一天一列，行為必須與改動前逐字相同。"""
    path = str(tmp_path / 'w.db')
    dao = GoldenAIBacktestMetricsDAO(db_path=path)
    assert dao.exists_for_date('2026-08-22', 'weekly', '1,2,3') is False
    dao.save(timestamp='2026-08-22 22:35:00', strategy='weekly',
             tranche=None, ranks='1,2,3', report=FakeReport())
    assert dao.exists_for_date('2026-08-22', 'weekly', '1,2,3') is True


def test_delete_for_date_clears_both_tables(partial_db):
    path, dao = partial_db
    assert dao.delete_for_date('2026-08-22', 'weekly_4w', '1,2,3') == 4   # 2 metrics + 2 reports
    assert _rows(path) == 0
    assert _rows(path, 'golden_ai_backtest_reports') == 0


def test_delete_for_date_is_scoped(partial_db):
    """別的日期／策略／ranks 不能被掃到。"""
    path, dao = partial_db
    dao.save(timestamp='2026-08-21 22:45:00', strategy='weekly_4w',
             tranche='tranche_1', ranks='1,2,3', report=FakeReport())
    dao.save(timestamp='2026-08-22 22:45:00', strategy='monthly',
             tranche='tranche_1', ranks='1,2,3', report=FakeReport())
    dao.save(timestamp='2026-08-22 22:45:00', strategy='weekly_4w',
             tranche='tranche_1', ranks='1,2', report=FakeReport())

    assert dao.delete_for_date('2026-08-22', 'weekly_4w', '1,2,3') == 4
    assert _rows(path) == 3


def test_delete_for_date_on_nothing_is_a_no_op(tmp_path):
    dao = GoldenAIBacktestMetricsDAO(db_path=str(tmp_path / 'empty.db'))
    assert dao.delete_for_date('2026-08-22', 'weekly_4w', '1,2,3') == 0
