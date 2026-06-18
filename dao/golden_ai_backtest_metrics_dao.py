import sqlite3
import logging
import pandas as pd
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


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
                    week          TEXT,
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
                    week          TEXT,
                    ranks         TEXT NOT NULL DEFAULT '',
                    report_json   TEXT NOT NULL,
                    position_json TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_reports_lookup
                ON golden_ai_backtest_reports(strategy, timestamp, ranks)
            """)

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
                        week          TEXT,
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
                        (timestamp, strategy, week, ranks, annual_return, sharpe, sortino, max_drawdown, win_ratio)
                    SELECT timestamp, strategy, week,
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

    def save(self, timestamp: str, strategy: str, week: Optional[str], ranks: str, report) -> None:
        try:
            metrics = report.get_metrics()
            annual_return = metrics.get('profitability', {}).get('annualReturn')
            sharpe        = metrics.get('ratio', {}).get('sharpeRatio')
            sortino       = metrics.get('ratio', {}).get('sortinoRatio')
            max_drawdown  = metrics.get('risk', {}).get('maxDrawdown')
            win_ratio     = metrics.get('winrate', {}).get('winRate')
        except Exception as e:
            logger.warning(f"get_metrics() failed for {strategy} {week} Ranks[{ranks}]: {e}. Saving NULLs.")
            annual_return = sharpe = sortino = max_drawdown = win_ratio = None

        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.execute("""
                INSERT INTO golden_ai_backtest_metrics
                    (timestamp, strategy, week, ranks, annual_return, sharpe, sortino, max_drawdown, win_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, strategy, week, ranks, annual_return, sharpe, sortino, max_drawdown, win_ratio))
            conn.commit()
        finally:
            conn.close()

        logger.info(f"Saved metrics: {strategy} {week} Ranks[{ranks}] @ {timestamp}")

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

    def load(self, strategy: Optional[str] = None, week: Optional[str] = None,
             ranks: Optional[str] = None) -> pd.DataFrame:
        conditions = []
        params = []

        if strategy is not None:
            conditions.append("strategy = ?")
            params.append(strategy)
        if week is not None:
            conditions.append("week = ?")
            params.append(week)
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

    def save_report(self, timestamp: str, strategy: str, week: Optional[str],
                    ranks: str, report_json: str, position_json: str) -> None:
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.execute("""
                INSERT INTO golden_ai_backtest_reports
                    (timestamp, strategy, week, ranks, report_json, position_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (timestamp, strategy, week, ranks, report_json, position_json))
            conn.commit()
        finally:
            conn.close()
        logger.info(f"Saved report JSON: {strategy} {week} Ranks[{ranks}] @ {timestamp}")

    def get_report(self, timestamp: str, strategy: str,
                   week: Optional[str] = None,
                   ranks: Optional[str] = None) -> Optional[Tuple[str, str]]:
        conditions = ["strategy = ?", "timestamp = ?"]
        params: list = [strategy, timestamp]
        if week is not None:
            conditions.append("week = ?")
            params.append(week)
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
                f"SELECT timestamp, strategy, week, ranks "
                f"FROM golden_ai_backtest_reports "
                f"WHERE {' AND '.join(conditions)} "
                f"ORDER BY timestamp DESC",
                conn, params=params
            )
        finally:
            conn.close()
        return df
