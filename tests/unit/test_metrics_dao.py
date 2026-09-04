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

    真正的競態視窗就在這兩句之間：`rename_column_if_needed` 讀完欄位快照才發 ALTER，
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
    from dao.golden_ai_backtest_metrics_dao import rename_column_if_needed

    conn = sqlite3.connect(legacy_db)
    try:
        cursor = _RacingCursor(
            conn.cursor(), legacy_db,
            "ALTER TABLE golden_ai_backtest_metrics RENAME COLUMN week TO tranche")
        renamed = rename_column_if_needed(
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
    from dao.golden_ai_backtest_metrics_dao import rename_column_if_needed

    class _BrokenCursor:
        def execute(self, sql, *args):
            if sql.startswith('ALTER'):
                raise sqlite3.OperationalError('database is locked')
            self._rows = [(0, 'week', 'TEXT', 0, None, 0)]
            return self

        def fetchall(self):
            return self._rows

    with pytest.raises(sqlite3.OperationalError, match='locked'):
        rename_column_if_needed(_BrokenCursor(), 'whatever', 'week', 'tranche')


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
    """weekly 一天一列（tranche 為 NULL），行為必須與改動前逐字相同。"""
    path = str(tmp_path / 'w.db')
    dao = GoldenAIBacktestMetricsDAO(db_path=path)
    assert dao.exists_for_date('2026-08-22', 'weekly', '1,2,3') is False
    dao.save(timestamp='2026-08-22 22:35:00', strategy='weekly',
             tranche=None, ranks='1,2,3', report=FakeReport())
    assert dao.exists_for_date('2026-08-22', 'weekly', '1,2,3') is True


def test_two_overlapping_partial_sets_do_not_add_up_to_done(tmp_path):
    """兩個殘缺組並存時不能用列數湊滿。

    正式機累積的舊列是「先刪、再分四次 INSERT」那版寫的（現在走 `save_group`）；
    兩支行程重疊時「B 刪完 → A 寫入 → B 寫入」就會讓兩組並存。數列數的話這裡是 4、
    判定完成、從此永久跳過，而實際上 tranche_4 一次都沒算過。
    """
    path = str(tmp_path / 'overlap.db')
    dao = GoldenAIBacktestMetricsDAO(db_path=path)
    for ts, tranches in [('2026-08-22 09:00:00', ('tranche_1', 'tranche_2')),
                         ('2026-08-22 22:45:00', ('tranche_1', 'tranche_3'))]:
        for tranche in tranches:
            dao.save(timestamp=ts, strategy='weekly_4w', tranche=tranche,
                     ranks='1,2,3', report=FakeReport())

    assert _rows(path) == 4                                                   # 列數湊滿了
    assert dao.exists_for_date('2026-08-22', 'weekly_4w', '1,2,3', expected=4) is False


def test_a_duplicated_complete_set_still_counts_as_done(tmp_path):
    """同樣的交錯下若兩組都寫完，四份都在，就不該再重算一次。"""
    path = str(tmp_path / 'dup.db')
    dao = GoldenAIBacktestMetricsDAO(db_path=path)
    for ts in ('2026-08-22 09:00:00', '2026-08-22 22:45:00'):
        for n in range(1, 5):
            dao.save(timestamp=ts, strategy='weekly_4w', tranche=f'tranche_{n}',
                     ranks='1,2,3', report=FakeReport())
    assert dao.exists_for_date('2026-08-22', 'weekly_4w', '1,2,3', expected=4) is True


def _tranches(path):
    conn = sqlite3.connect(path)
    try:
        return sorted(r[0] for r in conn.execute(
            "SELECT tranche FROM golden_ai_backtest_metrics"))
    finally:
        conn.close()


FOUR = {f'tranche_{n}': FakeReport() for n in range(1, 5)}


def test_save_group_replaces_the_partial_set(partial_db):
    """殘缺的兩份被換成完整的四份，不是疊上去——`save` 沒有唯一鍵，疊上去的話
    `_normalized()` 的平均會被重複計入的那幾份拉偏。"""
    path, dao = partial_db
    dao.save_group(timestamp='2026-08-22 23:10:00', strategy='weekly_4w',
                   ranks='1,2,3', reports=FOUR)
    assert _rows(path) == 4
    assert _tranches(path) == ['tranche_1', 'tranche_2', 'tranche_3', 'tranche_4']


def test_save_group_leaves_the_reports_alone(partial_db):
    """報告不能跟著整組刪：它們的重寫要先跑 display() 產 HTML，中途掛掉就補不回來了。
    那半由 `replace_report` 逐份就地換。"""
    path, dao = partial_db
    dao.save_group(timestamp='2026-08-22 23:10:00', strategy='weekly_4w',
                   ranks='1,2,3', reports=FOUR)
    assert _rows(path, 'golden_ai_backtest_reports') == 2


def test_save_group_is_scoped(partial_db):
    """別的日期／策略／ranks 不能被掃到。"""
    path, dao = partial_db
    for ts, strategy, ranks in [('2026-08-21 22:45:00', 'weekly_4w', '1,2,3'),
                                ('2026-08-22 22:45:00', 'monthly', '1,2,3'),
                                ('2026-08-22 22:45:00', 'weekly_4w', '1,2')]:
        dao.save(timestamp=ts, strategy=strategy, tranche='tranche_1',
                 ranks=ranks, report=FakeReport())

    dao.save_group(timestamp='2026-08-22 23:10:00', strategy='weekly_4w',
                   ranks='1,2,3', reports=FOUR)
    assert _rows(path) == 4 + 3        # 換掉的那 2 列不見了，另外三列原封不動


def test_save_group_is_atomic(partial_db):
    """清舊與寫新同一筆 transaction。分開做的話，掛在中間就是把稍早算好的那幾列
    刪光而且補不回來——`exists_for_date` 永遠只問今天，沒有人會回頭看舊日期。

    製造中途失敗的方式：讓其中一份的指標是 sqlite 綁不了的型別，INSERT 會在
    DELETE 已經執行之後才炸。"""
    path, dao = partial_db
    before = _tranches(path)

    class Unbindable:
        def get_metrics(self):
            return {'profitability': {'annualReturn': object()}}

    with pytest.raises(sqlite3.InterfaceError):
        dao.save_group(timestamp='2026-08-22 23:10:00', strategy='weekly_4w',
                       ranks='1,2,3', reports=dict(FOUR, tranche_3=Unbindable()))

    assert _tranches(path) == before    # 舊列還在，不是被刪光之後留下空白


def test_save_group_on_an_empty_day_just_writes(tmp_path):
    dao = GoldenAIBacktestMetricsDAO(db_path=str(tmp_path / 'empty.db'))
    dao.save_group(timestamp='2026-08-22 22:45:00', strategy='weekly_4w',
                   ranks='1,2,3', reports=FOUR)
    assert _tranches(str(tmp_path / 'empty.db')) == [
        'tranche_1', 'tranche_2', 'tranche_3', 'tranche_4']


def test_save_group_keeps_saving_when_one_report_has_no_metrics(partial_db):
    """抽不到指標寫 NULL 但不中斷，與 `save` 一致——而且是在開 transaction 之前抽的。"""
    path, dao = partial_db

    class Broken:
        def get_metrics(self):
            raise ValueError('no metrics')

    reports = dict(FOUR, tranche_3=Broken())
    dao.save_group(timestamp='2026-08-22 23:10:00', strategy='weekly_4w',
                   ranks='1,2,3', reports=reports)
    conn = sqlite3.connect(path)
    try:
        got = dict(conn.execute(
            "SELECT tranche, annual_return FROM golden_ai_backtest_metrics"))
    finally:
        conn.close()
    assert got['tranche_3'] is None
    assert got['tranche_1'] is not None
    assert len(got) == 4


# ── replace_report：逐份就地換掉當天的報告 ──

def _report_rows(path, tranche):
    """某一份 tranche 的報告列。`tranche=None` 是 weekly 那條路徑的真實值，不是「全部」。"""
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            "SELECT timestamp, report_json FROM golden_ai_backtest_reports "
            "WHERE tranche IS ?", (tranche,)).fetchall()
    finally:
        conn.close()


def test_replace_report_swaps_that_tranche_only(partial_db):
    """換掉 tranche_1，tranche_2 上一次的報告原封不動——這正是掛在第 n 份時，
    還沒輪到的那幾份留著舊報告而不是空白的原因。"""
    path, dao = partial_db
    dao.replace_report(timestamp='2026-08-22 23:10:00', strategy='weekly_4w',
                       tranche='tranche_1', ranks='1,2,3',
                       report_json='{"new": 1}', position_json='{}')
    assert _report_rows(path, 'tranche_1') == [('2026-08-22 23:10:00', '{"new": 1}')]
    assert _report_rows(path, 'tranche_2') == [('2026-08-22 22:45:00', '{}')]


def test_replace_report_does_not_accumulate_across_runs(partial_db):
    """同一天重跑幾次都只留最後一份。`save_report` 是純 INSERT，沒有這個保證，
    而 get_report 是 LIMIT 1 without ORDER BY——堆起來就會隨機拿到舊的那份。"""
    path, dao = partial_db
    for ts in ('2026-08-22 23:10:00', '2026-08-22 23:40:00'):
        dao.replace_report(timestamp=ts, strategy='weekly_4w', tranche='tranche_1',
                           ranks='1,2,3', report_json=f'{{"ts": "{ts}"}}', position_json='{}')
    assert _report_rows(path, 'tranche_1') == [
        ('2026-08-22 23:40:00', '{"ts": "2026-08-22 23:40:00"}')]


def test_replace_report_is_scoped_to_the_same_day(partial_db):
    """昨天的報告不該被今天的重跑掃掉——那是別一天的資料點。"""
    path, dao = partial_db
    dao.save_report(timestamp='2026-08-21 22:45:00', strategy='weekly_4w',
                    tranche='tranche_1', ranks='1,2,3',
                    report_json='{"yesterday": 1}', position_json='{}')
    dao.replace_report(timestamp='2026-08-22 23:10:00', strategy='weekly_4w',
                       tranche='tranche_1', ranks='1,2,3',
                       report_json='{"today": 1}', position_json='{}')
    assert sorted(_report_rows(path, 'tranche_1')) == [
        ('2026-08-21 22:45:00', '{"yesterday": 1}'),
        ('2026-08-22 23:10:00', '{"today": 1}'),
    ]


def test_replace_report_matches_a_null_tranche(tmp_path):
    """weekly 的 tranche 是 NULL，`= NULL` 永遠不成立——用 `IS ?` 才換得掉。"""
    path = str(tmp_path / 'weekly.db')
    dao = GoldenAIBacktestMetricsDAO(db_path=path)
    dao.save_report(timestamp='2026-08-22 22:35:00', strategy='weekly', tranche=None,
                    ranks='1,2,3', report_json='{"old": 1}', position_json='{}')
    dao.replace_report(timestamp='2026-08-22 23:00:00', strategy='weekly', tranche=None,
                       ranks='1,2,3', report_json='{"new": 1}', position_json='{}')
    assert _report_rows(path, None) == [('2026-08-22 23:00:00', '{"new": 1}')]
