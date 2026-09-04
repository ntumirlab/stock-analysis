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

from core.tranche_schedule import TRANCHE_ANCHOR_SUNDAYS, tranche_of
from dao.golden_ai_backtest_metrics_dao import _rename_column_if_needed

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
                    tranche       INTEGER,
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

            # Migration: 相位改由錨點連續輪動定義後，`week_of_month`（當月第幾個週日）
            # 沒有日曆語意了，改名為 `tranche`，既有的值也要跟著換算。
            #
            # 節點可以就地重算、metrics/reports 不行，差別在身分：節點的進出場日與
            # 報酬只由 (list_date, ranks) 決定（見 `jobs.golden_ai_node_executor.run_node`），
            # 換排程不影響任何數字，`tranche` 是唯一失真的欄位；metrics 那邊的 `Week1~4`
            # 記的是舊排程當時真的買了哪幾週，改標才是說謊。
            #
            # `BEGIN IMMEDIATE` 是這裡的關鍵：sqlite3 只在 DML 之前隱式開 transaction，
            # DDL 走 autocommit——改名一下去就落地了。少了這句，改標中途失敗會留下
            # 「名改了、值沒補」，而且下次啟動已經沒得改名、閘門不會再開，錯的標籤就
            # 永遠留著。順帶先把寫鎖拿到手：多個容器同時啟動時，後手會等先手 commit
            # 完再讀欄位快照，而不是撞在改名與改標之間。
            #
            # 但只在真的還沒改名時才開。搶寫鎖是有代價的：dashboard 在 module import
            # 就建這個 DAO，回填正在寫的時候，白搶一次鎖會被擋滿 busy timeout、然後
            # 炸在 import。交易裡的 `_rename_column_if_needed` 會自己重讀欄位快照
            # （拿到寫鎖之後才讀），所以這個先探不會把競態放回來。
            cursor.execute("PRAGMA table_info(golden_ai_backtest_nodes)")
            if any(row[1] == 'week_of_month' for row in cursor.fetchall()):
                conn.execute("BEGIN IMMEDIATE")
                if _rename_column_if_needed(cursor, 'golden_ai_backtest_nodes',
                                            'week_of_month', 'tranche'):
                    self._relabel_tranches(cursor)

            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _relabel_tranches(cursor) -> None:
        """把改名前寫進來的相位值換算成新定義。與改名一起成敗——同進同退靠的是呼叫端
        開的 `BEGIN IMMEDIATE`，不是 driver 自己會包。

        `tranche IS NULL` 的列（weekly，一週一輪、沒有相位）原樣不動。相位是
        (strategy, list_date) 的純函數，所以同一份清單的所有 ranks 一次更新。

        算不出新標籤的（策略沒有錨點）一律清成 NULL：改名是一次性閘門，這一輪
        commit 完欄位就叫 `tranche`、之後不會再有人來補，留著舊的月份索引等於讓
        每個讀的人把它當錨點相位讀（而且月份索引可以是 5，根本不在 1~4 裡）。
        沒有標籤是誠實的，錯的標籤不是。
        """
        rows = cursor.execute(
            "SELECT DISTINCT strategy, list_date FROM golden_ai_backtest_nodes "
            "WHERE tranche IS NOT NULL"
        ).fetchall()

        updates, unknown = [], set()
        for strategy, list_date in rows:
            if strategy not in TRANCHE_ANCHOR_SUNDAYS:
                unknown.add(strategy)
                updates.append((None, strategy, list_date))
                continue
            updates.append((tranche_of(strategy, list_date), strategy, list_date))

        if unknown:
            logger.warning(f"有相位值卻沒有 tranche 錨點，標籤清成 NULL: {sorted(unknown)}")
        if not updates:
            return

        # 這個數字是操作人員確認改標真的動到東西的唯一依據，所以不靠 `executemany`
        # 之後的 `cursor.rowcount`（driver 的累加行為），改用連線層實際數過的差值。
        changed_before = cursor.connection.total_changes
        cursor.executemany(
            "UPDATE golden_ai_backtest_nodes SET tranche = ? "
            "WHERE strategy = ? AND list_date = ?",
            updates,
        )
        changed = cursor.connection.total_changes - changed_before
        logger.info(f"tranche 標籤已重算: {len(updates)} 份清單、{changed} 列")

    def stored_list_dates(self, strategy: str, ranks: str) -> set:
        """某策略某組 ranks 已經存過的清單日。

        回填時一組 ranks 查一次，之後在記憶體裡比對。逐節點各查一次的話，全量
        回填（255 組 × 48 份清單）會開關一萬兩千次連線，其中絕大多數只是為了確認
        「這個已經有了」。
        """
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            cursor = conn.execute(
                "SELECT list_date FROM golden_ai_backtest_nodes "
                "WHERE strategy = ? AND ranks = ?",
                (strategy, ranks),
            )
            return {row[0] for row in cursor}
        finally:
            conn.close()

    def save(self, strategy: str, list_date: str, ranks: str, entry_date: str,
             exit_date: str, n_stocks: int, node_return: float,
             tranche: Optional[int] = None, report=None,
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
                    (strategy, list_date, ranks, entry_date, exit_date, tranche,
                     n_stocks, node_return, annual_return, sharpe, sortino,
                     max_drawdown, win_ratio, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy, list_date, ranks, entry_date, exit_date, tranche,
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
