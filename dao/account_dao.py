import sqlite3
import logging

logger = logging.getLogger(__name__)


class AccountDAO:
    def __init__(self, db_path="data_prod.db"):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        """建立 account 資料表，包含 account_id (PK)user_name 與建立時間"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS account (
                account_id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_name TEXT UNIQUE,
                broker_name TEXT,
                user_name TEXT,
                created_timestamp TEXT DEFAULT (datetime('now','localtime'))
            );
        """)
        conn.commit()
        conn.close()

    def get_account_id(self, account_name, broker_name, user_name):
        """
        根據 account_name 取得 account_id，若不存在則新增一筆帳戶資料並回傳 account_id。

        Args:
            account_name (str): 帳戶名稱，例如 "junting_fugle"。
            broker_name (str): 券商名稱，例如 "fugle"。
            user_name (str): 使用者名稱，例如 "junting"。

        Returns:
            int: 資料表 account 的主鍵 account_id。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT account_id FROM account WHERE account_name = ?", (account_name,))
        row = cursor.fetchone()
        if row:
            account_id = row[0]
            logger.info(f"Found existing account_id: {account_id}")
        else:
            cursor.execute(
                """
                INSERT INTO account (account_name, broker_name, user_name)
                VALUES (?, ?, ?)
            """,
                (account_name, broker_name, user_name),
            )
            conn.commit()
            account_id = cursor.lastrowid
            logger.info(f"Created new account_id: {account_id}")
        conn.close()
        return account_id

    def get_all_accounts(self):
        """
        取得所有帳戶資料，回傳一個列表，每個項目為 dict，包含 account_id, account_name, broker_name, user_name, created_timestamp。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT account_id, account_name, broker_name, user_name, created_timestamp FROM account")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        accounts = [dict(zip(columns, row)) for row in rows]
        return accounts
