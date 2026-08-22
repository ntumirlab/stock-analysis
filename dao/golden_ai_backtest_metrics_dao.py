import sqlite3
import logging
import pandas as pd
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _rename_column_if_needed(cursor, table: str, old: str, new: str) -> None:
    """欄位改名，已經改過就跳過。DAO 建構時呼叫，所以每個 process 都會跑到。

    讀 PRAGMA 與 ALTER 之間沒有鎖，部署時多個容器同時啟動的話，後手拿到的欄位快照
    會是舊的、ALTER 會噴 `no such column`。那不是錯誤——先手已經把事情做完了，
    確認結果對就好，不對才往外丟。
    """
    cursor.execute(f"PRAGMA table_info({table})")
    cols = {row[1] for row in cursor.fetchall()}
    if old not in cols or new in cols:
        return
    try:
        cursor.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")
    except sqlite3.OperationalError:
        cursor.execute(f"PRAGMA table_info({table})")
        if new not in {row[1] for row in cursor.fetchall()}:
            raise
        logger.info(f"{table}.{old} 已由其他 process 改名為 {new}")


class GoldenAIBacktestMetricsDAO:
    def __init__(self, db_path="data_prod.db"):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS golden_ai_backtest_metrics (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT NOT NULL,
                    strategy      TEXT NOT NULL,
                    tranche       TEXT,
                    ranks         TEXT NOT NULL DEFAULT '',
                    annual_return REAL,
                    sharpe        REAL,
                    sortino       REAL,
                    max_drawdown  REAL,
                    win_ratio     REAL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS golden_ai_backtest_reports (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT NOT NULL,
                    strategy      TEXT NOT NULL,
                    tranche       TEXT,
                    ranks         TEXT NOT NULL DEFAULT '',
                    report_json   TEXT NOT NULL,
                    position_json TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_reports_lookup
                ON golden_ai_backtest_reports(strategy, timestamp, ranks)
            """)

            # Migration: 4 週策略的相位改由錨點連續輪動定義（見 core/tranche_schedule），
            # 值從 Week1~4 變成 tranche1~4，欄位名跟著改。純 metadata 操作，與表裡有幾列無關。
            # 放在 top_n migration 之前，那支重建表時才會讀到已經改好名的來源欄位。
            for table in ('golden_ai_backtest_metrics', 'golden_ai_backtest_reports'):
                _rename_column_if_needed(cursor, table, 'week', 'tranche')

            # Migration: if top_n column exists, recreate table without it
            cursor.execute("PRAGMA table_info(golden_ai_backtest_metrics)")
            columns = {row[1] for row in cursor.fetchall()}
            if 'top_n' in columns:
                cursor.execute("ALTER TABLE golden_ai_backtest_metrics RENAME TO golden_ai_backtest_metrics_old")
                cursor.execute("""
                    CREATE TABLE golden_ai_backtest_metrics (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp     TEXT NOT NULL,
                        strategy      TEXT NOT NULL,
                        tranche       TEXT,
                        ranks         TEXT NOT NULL DEFAULT '',
                        annual_return REAL,
                        sharpe        REAL,
                        sortino       REAL,
                        max_drawdown  REAL,
                        win_ratio     REAL
                    )
                """)
                cursor.execute("""
                    INSERT INTO golden_ai_backtest_metrics
                        (timestamp, strategy, tranche, ranks, annual_return, sharpe, sortino, max_drawdown, win_ratio)
                    SELECT timestamp, strategy, tranche,
                        COALESCE(NULLIF(ranks, ''), CAST(top_n AS TEXT), ''),
                        annual_return, sharpe, sortino, max_drawdown, win_ratio
                    FROM golden_ai_backtest_metrics_old
                """)
                cursor.execute("DROP TABLE golden_ai_backtest_metrics_old")
                conn.commit()
            elif 'ranks' not in columns:
                cursor.execute("ALTER TABLE golden_ai_backtest_metrics ADD COLUMN ranks TEXT NOT NULL DEFAULT ''")
                conn.commit()

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_golden_ai_metrics_strategy_ranks
                ON golden_ai_backtest_metrics(strategy, ranks)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_golden_ai_metrics_timestamp
                ON golden_ai_backtest_metrics(timestamp)
            """)

            conn.commit()
        finally:
            conn.close()

    def save(self, timestamp: str, strategy: str, tranche: Optional[str], ranks: str, report) -> None:
        try:
            metrics = report.get_metrics()
            annual_return = metrics.get('profitability', {}).get('annualReturn')
            sharpe        = metrics.get('ratio', {}).get('sharpeRatio')
            sortino       = metrics.get('ratio', {}).get('sortinoRatio')
            max_drawdown  = metrics.get('risk', {}).get('maxDrawdown')
            win_ratio     = metrics.get('winrate', {}).get('winRate')
        except Exception as e:
            logger.warning(f"get_metrics() failed for {strategy} {tranche} Ranks[{ranks}]: {e}. Saving NULLs.")
            annual_return = sharpe = sortino = max_drawdown = win_ratio = None

        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.execute("""
                INSERT INTO golden_ai_backtest_metrics
                    (timestamp, strategy, tranche, ranks, annual_return, sharpe, sortino, max_drawdown, win_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, strategy, tranche, ranks, annual_return, sharpe, sortino, max_drawdown, win_ratio))
            conn.commit()
        finally:
            conn.close()

        logger.info(f"Saved metrics: {strategy} {tranche} Ranks[{ranks}] @ {timestamp}")

    def exists_for_date(self, date_str: str, strategy: str, ranks: str) -> bool:
        """檢查指定日期、策略、ranks 是否已有紀錄"""
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM golden_ai_backtest_metrics WHERE strategy = ? AND timestamp LIKE ? AND ranks = ? LIMIT 1",
                (strategy, f"{date_str}%", ranks)
            )
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def load(self, strategy: Optional[str] = None, tranche: Optional[str] = None,
             ranks: Optional[str] = None) -> pd.DataFrame:
        conditions = []
        params = []

        if strategy is not None:
            conditions.append("strategy = ?")
            params.append(strategy)
        if tranche is not None:
            conditions.append("tranche = ?")
            params.append(tranche)
        if ranks is not None:
            conditions.append("ranks = ?")
            params.append(ranks)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM golden_ai_backtest_metrics {where} ORDER BY timestamp ASC",
                conn,
                params=params
            )
        finally:
            conn.close()
        return df

    # ── Report JSON persistence ──

    def save_report(self, timestamp: str, strategy: str, tranche: Optional[str],
                    ranks: str, report_json: str, position_json: str) -> None:
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.execute("""
                INSERT INTO golden_ai_backtest_reports
                    (timestamp, strategy, tranche, ranks, report_json, position_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (timestamp, strategy, tranche, ranks, report_json, position_json))
            conn.commit()
        finally:
            conn.close()
        logger.info(f"Saved report JSON: {strategy} {tranche} Ranks[{ranks}] @ {timestamp}")

    def get_report(self, timestamp: str, strategy: str,
                   tranche: Optional[str] = None,
                   ranks: Optional[str] = None) -> Optional[Tuple[str, str]]:
        conditions = ["strategy = ?", "timestamp = ?"]
        params: list = [strategy, timestamp]
        if tranche is not None:
            conditions.append("tranche = ?")
            params.append(tranche)
        if ranks is not None:
            conditions.append("ranks = ?")
            params.append(ranks)

        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT report_json, position_json FROM golden_ai_backtest_reports "
                f"WHERE {' AND '.join(conditions)} LIMIT 1",
                params
            )
            row = cursor.fetchone()
        finally:
            conn.close()
        return (row[0], row[1]) if row else None

    def list_reports(self, strategy: str,
                     date_from: Optional[str] = None,
                     date_to: Optional[str] = None) -> pd.DataFrame:
        conditions = ["strategy = ?"]
        params: list = [strategy]
        if date_from:
            conditions.append("timestamp >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("timestamp <= ?")
            params.append(date_to + " 23:59:59")

        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            df = pd.read_sql_query(
                f"SELECT timestamp, strategy, tranche, ranks "
                f"FROM golden_ai_backtest_reports "
                f"WHERE {' AND '.join(conditions)} "
                f"ORDER BY timestamp DESC",
                conn, params=params
            )
        finally:
            conn.close()
        return df
