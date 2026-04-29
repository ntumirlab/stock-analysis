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
                top_n         INTEGER NOT NULL,
                annual_return REAL,
                sharpe        REAL,
                sortino       REAL,
                max_drawdown  REAL,
                win_ratio     REAL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_golden_ai_metrics_strategy_week_topn
            ON golden_ai_backtest_metrics(strategy, week, top_n)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_golden_ai_metrics_timestamp
            ON golden_ai_backtest_metrics(timestamp)
        """)

        conn.commit()
        conn.close()

    def save(self, timestamp: str, strategy: str, week: Optional[str], top_n: int, report) -> None:
        try:
            metrics = report.get_metrics()
            annual_return = metrics.get('profitability', {}).get('annualReturn')
            sharpe        = metrics.get('ratio', {}).get('sharpeRatio')
            sortino       = metrics.get('ratio', {}).get('sortinoRatio')
            max_drawdown  = metrics.get('risk', {}).get('maxDrawdown')
            win_ratio     = metrics.get('winrate', {}).get('winRate')
        except Exception as e:
            logger.warning(f"get_metrics() failed for {strategy} {week} Top{top_n}: {e}. Saving NULLs.")
            annual_return = sharpe = sortino = max_drawdown = win_ratio = None

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO golden_ai_backtest_metrics
                (timestamp, strategy, week, top_n, annual_return, sharpe, sortino, max_drawdown, win_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, strategy, week, top_n, annual_return, sharpe, sortino, max_drawdown, win_ratio))
        conn.commit()
        conn.close()

        logger.info(f"Saved metrics: {strategy} {week} Top{top_n} @ {timestamp}")

    def exists_for_date(self, date_str: str, strategy: str) -> bool:
        """檢查指定日期（YYYY-MM-DD）與策略是否已有紀錄"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM golden_ai_backtest_metrics WHERE strategy = ? AND timestamp LIKE ? LIMIT 1",
            (strategy, f"{date_str}%")
        )
        found = cursor.fetchone() is not None
        conn.close()
        return found

    def load(self, strategy: Optional[str] = None, week: Optional[str] = None,
             top_n: Optional[int] = None) -> pd.DataFrame:
        conditions = []
        params = []

        if strategy is not None:
            conditions.append("strategy = ?")
            params.append(strategy)
        if week is not None:
            conditions.append("week = ?")
            params.append(week)
        if top_n is not None:
            conditions.append("top_n = ?")
            params.append(top_n)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            f"SELECT * FROM golden_ai_backtest_metrics {where} ORDER BY timestamp ASC",
            conn,
            params=params
        )
        conn.close()
        return df
