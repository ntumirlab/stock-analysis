import sqlite3
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class InventoryDAO:
    def __init__(self, db_path="data_prod.db"):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        """建立 inventory_history 資料表，記錄庫存資料並以 account_id 作為外鍵"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER,
                fetch_timestamp TEXT,
                create_timestamp TEXT DEFAULT (datetime('now','localtime')),
                stock_id TEXT,
                stock_name TEXT,
                quantity REAL,
                last_price REAL,
                pnl REAL,
                raw_data TEXT,
                FOREIGN KEY (account_id) REFERENCES account(account_id)
            );
        """)
        conn.commit()
        conn.close()

    def insert_inventory_data(self, account_id, inventory_data, fetch_timestamp=None):
        """
        將庫存資料寫入 inventory_history 表。

        Args:
            account_id (int): 對應 Account 資料表的 account_id。
            inventory_data (list[dict]): 庫存資料，每筆需包含 'stock_id', 'stock_name',
                'quantity', 'last_price', 'pnl', 以及原始資料('raw_data')
            fetch_timestamp (str, optional): 資料擷取時間戳記，若未提供則使用當前時間
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 使用提供的擷取時間戳記或當前時間
        if fetch_timestamp is None:
            raise ValueError("fetch_timestamp cannot be None")
        batch_ts_str = fetch_timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # 寫入新資料 (不刪除舊資料，以保留歷史記錄)
        for item in inventory_data:
            # 將原始資料轉為 JSON 字串儲存
            raw_data_json = json.dumps(item.get('raw_data', {}))

            cursor.execute(
                """
                INSERT INTO inventory_history (
                    account_id,
                    fetch_timestamp,
                    stock_id,
                    stock_name,
                    quantity,
                    last_price,
                    pnl,
                    raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    account_id,
                    batch_ts_str,
                    item.get('stock_id'),
                    item.get('stock_name'),
                    item.get('quantity'),
                    item.get('last_price'),
                    item.get('pnl'),
                    raw_data_json,
                ),
            )

        conn.commit()
        conn.close()
        logger.info(f"Inserted {len(inventory_data)} inventory_history records for account_id {account_id}")

    def get_inventories_by_account_and_date(self, account_id, query_date):
        """
        根據帳戶ID和日期獲取該帳戶當天的庫存記錄

        Args:
            account_id (int): 帳戶ID
            query_date (datetime.date): 查詢日期

        Returns:
            list: 包含庫存記錄的列表
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
            SELECT stock_id, stock_name, quantity, last_price, pnl, raw_data
            FROM inventory_history
            WHERE account_id = ? AND fetch_timestamp BETWEEN ? AND ?
            ORDER BY fetch_timestamp DESC
        """,
            (account_id, start_date, end_date),
        )

        results = cursor.fetchall()

        # 將結果轉換為列表
        inventories = []
        for row in results:
            inventory = dict(row)

            # 將 raw_data 從 JSON 字串轉回字典
            try:
                inventory['raw_data'] = json.loads(inventory['raw_data'])
            except (json.JSONDecodeError, TypeError):
                inventory['raw_data'] = {}

            inventories.append(inventory)

        conn.close()
        return inventories
