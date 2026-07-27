import sqlite3
import logging
import json

logger = logging.getLogger(__name__)


class ProfitLossDAO:
    def __init__(self, db_path="data_prod.db"):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        """建立 profit_loss_history 表，記錄券商回報的已實現損益（平倉損益）。

        與 inventory_history（未實現）互補：未實現隨市價每日變動、只留當日快照，
        已實現則是成交後不再變動的事實，需要長期累積。

        唯一索引存在的理由：fetcher 每天重抓一段區間（補某天 job 掛掉的漏），
        同一筆平倉會被抓到多次，靠唯一鍵讓重複資料被忽略而非累積成假損益。
        """
        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS profit_loss_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER,
                trade_date TEXT,
                stock_id TEXT,
                stock_name TEXT,
                quantity REAL,
                price REAL,
                pnl REAL,
                pr_ratio REAL,
                cond TEXT,
                dseq TEXT,
                seqno TEXT,
                raw_data TEXT,
                fetch_timestamp TEXT,
                create_timestamp TEXT DEFAULT (datetime('now','localtime')),
                FOREIGN KEY (account_id) REFERENCES account(account_id)
            );
        """)
        # 券商的 ProfitLoss.id 疑似為單次查詢內的序號（inventory 的 raw_data 就出現過
        # id=0），不可當穩定主鍵，故以業務欄位組合去重
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_profit_loss_unique
            ON profit_loss_history (account_id, trade_date, stock_id, dseq, seqno);
        """)
        conn.commit()
        conn.close()

    def insert_profit_loss(self, account_id, profit_loss_data, fetch_timestamp=None):
        """寫入已實現損益資料，重複的筆數會被唯一索引擋掉。

        Args:
            account_id (int): 對應 account 表的 account_id
            profit_loss_data (list[dict]): 每筆需含 'trade_date', 'stock_id', 'stock_name',
                'quantity', 'price', 'pnl', 'pr_ratio', 'cond', 'dseq', 'seqno', 'raw_data'
            fetch_timestamp (datetime.datetime): 抓取時間

        Returns:
            int: 實際新增的筆數（已存在的不計）
        """
        if fetch_timestamp is None:
            raise ValueError("fetch_timestamp cannot be None")

        batch_ts_str = fetch_timestamp.strftime("%Y-%m-%d %H:%M:%S")

        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.cursor()

        inserted = 0
        for item in profit_loss_data:
            raw_data_json = json.dumps(item.get('raw_data', {}), default=str)
            cursor.execute("""
                INSERT OR IGNORE INTO profit_loss_history (
                    account_id,
                    trade_date,
                    stock_id,
                    stock_name,
                    quantity,
                    price,
                    pnl,
                    pr_ratio,
                    cond,
                    dseq,
                    seqno,
                    raw_data,
                    fetch_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                account_id,
                item.get('trade_date'),
                item.get('stock_id'),
                item.get('stock_name'),
                item.get('quantity'),
                item.get('price'),
                item.get('pnl'),
                item.get('pr_ratio'),
                item.get('cond'),
                item.get('dseq'),
                item.get('seqno'),
                raw_data_json,
                batch_ts_str
            ))
            inserted += cursor.rowcount

        conn.commit()
        conn.close()

        logger.info(
            f"Profit/loss records for account_id {account_id}: "
            f"{len(profit_loss_data)} fetched, {inserted} newly inserted"
        )
        return inserted

    def get_profit_loss(self, account_id, start_date, end_date):
        """取得指定期間內的已實現損益明細（依成交日新到舊）。

        Args:
            account_id (int): 帳戶ID
            start_date (datetime.date): 起始日（含）
            end_date (datetime.date): 結束日（含）

        Returns:
            list[dict]: 已實現損益記錄
        """
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT trade_date, stock_id, stock_name, quantity, price,
                   pnl, pr_ratio, cond
            FROM profit_loss_history
            WHERE account_id = ? AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date DESC, stock_id ASC
        """, (
            account_id,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        ))

        records = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return records

    def get_latest_trade_date(self, account_id):
        """取得該帳戶已入庫的最新成交日，供 fetcher 決定回補起點。

        Returns:
            str | None: "YYYY-MM-DD"，無資料時回 None
        """
        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MAX(trade_date) FROM profit_loss_history WHERE account_id = ?
        """, (account_id,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row and row[0] else None
