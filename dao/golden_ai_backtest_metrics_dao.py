import sqlite3
import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


class GoldenAIBacktestMetricsDAO:
    def __init__(self, db_path="data_prod.db"):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        conn = sqlite3.connect(self.db_path)
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

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO golden_ai_backtest_metrics
                (timestamp, strategy, week, ranks, annual_return, sharpe, sortino, max_drawdown, win_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, strategy, week, ranks, annual_return, sharpe, sortino, max_drawdown, win_ratio))
        conn.commit()
        conn.close()

        logger.info(f"Saved metrics: {strategy} {week} Ranks[{ranks}] @ {timestamp}")

    def exists_for_date(self, date_str: str, strategy: str, ranks: str) -> bool:
        """檢查指定日期、策略、ranks 是否已有紀錄"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM golden_ai_backtest_metrics WHERE strategy = ? AND timestamp LIKE ? AND ranks = ? LIMIT 1",
            (strategy, f"{date_str}%", ranks)
        )
        found = cursor.fetchone() is not None
        conn.close()
        return found

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

        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            f"SELECT * FROM golden_ai_backtest_metrics {where} ORDER BY timestamp ASC",
            conn,
            params=params
        )
        conn.close()
        return df
