import sqlite3
import logging

logger = logging.getLogger(__name__)


class OrderDAO:
    def __init__(self, db_path="data_prod.db"):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        """建立 order_history 資料表，記錄下單紀錄並以 account_id 作為外鍵"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS order_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER,
                order_timestamp TEXT,
                create_timestamp TEXT DEFAULT (datetime('now','localtime')),
                action TEXT,
                stock_id TEXT,
                stock_name TEXT,
                quantity REAL,
                limit_price REAL,
                extra_bid_pct REAL,
                order_condition TEXT,
                view_only INTEGER DEFAULT 0,
                FOREIGN KEY (account_id) REFERENCES account(account_id)
            );
        """)
        conn.commit()
        conn.close()

    def insert_order_logs(self, order_logs, account_id, order_timestamp, view_only=False):
        """
        將下單記錄寫入 order_history 表。

        Args:
            order_logs (list[dict]): 每筆訂單資料，包含 'action', 'stock_id', 'stock_name',
                'quantity', 'limit_price', 'extra_bid_pct', 'order_condition'
            account_id (int): 對應 Account 資料表的 account_id。
            order_timestamp (datetime.datetime): 批次時間，用於標記這次下單作業。
            view_only (bool): 是否為模擬下單 (True) 或實際下單 (False)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if order_timestamp is None:
            raise ValueError("order_timestamp cannot be None")
        # 將批次時間轉為字串格式儲存 (ISO 格式)
        batch_ts_str = order_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        view_only_int = 1 if view_only else 0  # 轉換布爾值為整數
        for order in order_logs:
            cursor.execute(
                """
                INSERT INTO order_history (
                    account_id,
                    order_timestamp,
                    action,
                    stock_id,
                    stock_name,
                    quantity,
                    limit_price,
                    extra_bid_pct,
                    order_condition,
                    view_only
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    account_id,
                    batch_ts_str,
                    order.get("action"),
                    order.get("stock_id"),
                    order.get("stock_name"),
                    order.get("quantity"),
                    order.get("limit_price"),
                    order.get("extra_bid_pct"),
                    order.get("order_condition"),
                    view_only_int,
                ),
            )
        conn.commit()
        conn.close()
        logger.info(f"Inserted {len(order_logs)} order logs into order_history")

    def get_orders_by_account_and_date(self, account_id, query_date):
        """
        根據 account_id 與 query_date 撈取當天的訂單資料。
        query_date 為 datetime.date 物件
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 使結果以字典形式返回
        cursor = conn.cursor()

        # 將日期轉換為開始和結束時間範圍
        date_str = query_date.strftime("%Y-%m-%d")
        start_date = f"{date_str} 00:00:00"
        end_date = f"{date_str} 23:59:59"

        cursor.execute(
            """
            SELECT account_id, order_timestamp, create_timestamp, action, stock_id, stock_name, 
                quantity, limit_price, extra_bid_pct, order_condition, view_only
            FROM order_history
            WHERE account_id = ? AND order_timestamp BETWEEN ? AND ?
        """,
            (account_id, start_date, end_date),
        )

        results = cursor.fetchall()

        # 將結果轉換為列表
        orders = [dict(row) for row in results]

        conn.close()
        return orders

    def get_available_years(self, account_id):
        """取得指定帳戶所有可用的年份"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT strftime('%Y', order_timestamp) as year
            FROM order_history
            WHERE account_id = ?
            ORDER BY year DESC
        """,
            (account_id,),
        )
        years = [row[0] for row in cursor.fetchall()]
        conn.close()
        return years

    def get_available_months(self, account_id, year):
        """取得指定帳戶和年份所有可用的月份"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT strftime('%m', order_timestamp) as month
            FROM order_history
            WHERE account_id = ? AND strftime('%Y', order_timestamp) = ?
            ORDER BY month DESC
        """,
            (account_id, year),
        )
        months = [row[0] for row in cursor.fetchall()]
        conn.close()
        return months

    def get_available_days(self, account_id, year, month):
        """取得指定帳戶、年份和月份所有可用的日期"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT strftime('%d', order_timestamp) as day
            FROM order_history
            WHERE account_id = ? 
            AND strftime('%Y', order_timestamp) = ? 
            AND strftime('%m', order_timestamp) = ?
            ORDER BY day DESC
        """,
            (account_id, year, month),
        )
        days = [row[0] for row in cursor.fetchall()]
        conn.close()
        return days
