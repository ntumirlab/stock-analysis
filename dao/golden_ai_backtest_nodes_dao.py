"""
Data Access Object for GoldenAI node backtests.

一個節點 = 一份推薦清單 × 一組 ranks 的一次獨立回測（進場一次、出場一次）。
與 golden_ai_backtest_metrics 的差別在身分：滾動視窗的那張表按「哪天跑的」
記錄，同一份清單每天都會被重算；節點結算後就固定不變，所以身分是
(strategy, list_date, ranks)，跟計算時間無關。

只有已結算的節點才會寫進來，因此不需要 settled 欄位——表裡有的就是能畫的。
"""

import logging
import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def extract_metrics(report) -> dict:
    """從 finlab report 取出五個純量。取不到就全給 None，不讓單一節點擋下整批回填。

    鍵路徑與 GoldenAIBacktestMetricsDAO.save 相同——兩處要一起改。
    """
    try:
        metrics = report.get_metrics()
        return {
            'annual_return': metrics.get('profitability', {}).get('annualReturn'),
            'sharpe':        metrics.get('ratio', {}).get('sharpeRatio'),
            'sortino':       metrics.get('ratio', {}).get('sortinoRatio'),
            'max_drawdown':  metrics.get('risk', {}).get('maxDrawdown'),
            'win_ratio':     metrics.get('winrate', {}).get('winRate'),
        }
    except Exception as e:
        logger.warning(f"get_metrics() failed: {e}. Saving NULLs.")
        return {k: None for k in
                ('annual_return', 'sharpe', 'sortino', 'max_drawdown', 'win_ratio')}


class GoldenAIBacktestNodesDAO:
    def __init__(self, db_path="data_prod.db"):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS golden_ai_backtest_nodes (
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
                )
            """)

            # 節點的身分。回填因此是 idempotent 的：重跑同一份清單會被擋下而不是寫成第二列
            cursor.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_key
                ON golden_ai_backtest_nodes(strategy, list_date, ranks)
            """)
            # dashboard 取一條線的查詢路徑：某策略某組 ranks，按結算日排序
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_nodes_series
                ON golden_ai_backtest_nodes(strategy, ranks, exit_date)
            """)
            conn.commit()
        finally:
            conn.close()

    def exists(self, strategy: str, list_date: str, ranks: str) -> bool:
        """回填用的快速跳過檢查——省下建構策略與跑 sim 的成本。"""
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM golden_ai_backtest_nodes "
                "WHERE strategy = ? AND list_date = ? AND ranks = ? LIMIT 1",
                (strategy, list_date, ranks),
            )
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def save(self, strategy: str, list_date: str, ranks: str, entry_date: str,
             exit_date: str, n_stocks: int, node_return: float,
             week_of_month: Optional[int] = None, report=None,
             created_at: Optional[str] = None) -> bool:
        """寫入一個已結算的節點。已存在則不動，回傳 False。

        `report` 給的話會一併存下 finlab 的五個指標；node_return 一律由呼叫端算好
        （各股報酬的單純平均），不從 report 推導。
        """
        metrics = extract_metrics(report) if report is not None else {
            k: None for k in ('annual_return', 'sharpe', 'sortino', 'max_drawdown', 'win_ratio')
        }
        created_at = created_at or datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            cursor = conn.execute("""
                INSERT OR IGNORE INTO golden_ai_backtest_nodes
                    (strategy, list_date, ranks, entry_date, exit_date, week_of_month,
                     n_stocks, node_return, annual_return, sharpe, sortino,
                     max_drawdown, win_ratio, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy, list_date, ranks, entry_date, exit_date, week_of_month,
                n_stocks, node_return, metrics['annual_return'], metrics['sharpe'],
                metrics['sortino'], metrics['max_drawdown'], metrics['win_ratio'],
                created_at,
            ))
            conn.commit()
            inserted = cursor.rowcount > 0
        finally:
            conn.close()

        if inserted:
            logger.info(f"Saved node: {strategy} {list_date} Ranks[{ranks}] "
                        f"{entry_date}~{exit_date} {node_return:+.4%}")
        else:
            logger.debug(f"Node already exists, skipped: {strategy} {list_date} Ranks[{ranks}]")
        return inserted

    def load(self, strategy: Optional[str] = None,
             ranks: Optional[str] = None) -> pd.DataFrame:
        """按結算日排序，因為節點畫在 exit_date 上。"""
        conditions = []
        params = []

        if strategy is not None:
            conditions.append("strategy = ?")
            params.append(strategy)
        if ranks is not None:
            conditions.append("ranks = ?")
            params.append(ranks)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM golden_ai_backtest_nodes {where} ORDER BY exit_date ASC",
                conn,
                params=params,
            )
        finally:
            conn.close()
        return df
