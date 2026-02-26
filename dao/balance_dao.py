import sqlite3
import logging
import datetime
from dao.account_dao import AccountDAO

logger = logging.getLogger(__name__)


class BalanceDAO:
    def __init__(self, db_path="data_prod.db"):
        self.db_path = db_path
        self.account_dao = AccountDAO(db_path)
        self._create_table()

    def _create_table(self):
        """建立 balance_history 資料表，記錄帳戶餘額資訊"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS balance_history (
                balance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER,
                bank_balance REAL,
                settlements REAL,
                adjusted_bank_balance REAL,
                market_value REAL,
                total_assets REAL,
                fetch_timestamp TEXT,
                created_timestamp TEXT DEFAULT (datetime('now','localtime')),
                FOREIGN KEY (account_id) REFERENCES account (account_id)
            );
        """)
        conn.commit()
        conn.close()

    def get_account_id(self, account_name, broker_name, user_name):
        """
        依照帳戶名稱取得帳戶ID，若不存在則建立新帳戶

        Args:
            account_name (str): 帳戶名稱
            broker_name (str): 券商名稱
            user_name (str): 使用者名稱

        Returns:
            int: 帳戶ID
        """
        return self.account_dao.get_account_id(account_name, broker_name, user_name)

    def insert_balance(self, account_id, balance_data, fetch_timestamp=None):
        """
        插入餘額資料至資料庫

        Args:
            account_id (int): 帳戶ID
            balance_data (dict): 餘額資料，包含 bank_balance, settlements, adjusted_bank_balance, market_value, total_assets
            fetch_timestamp (datetime, optional): 紀錄時間戳記

        Returns:
            int: 新增資料的 balance_id
        """
        if fetch_timestamp is None:
            raise ValueError("timestamp cannot be None")

        batch_ts_str = fetch_timestamp.strftime("%Y-%m-%d %H:%M:%S")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO balance_history (
                account_id, bank_balance, settlements, adjusted_bank_balance, 
                market_value, total_assets, fetch_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                account_id,
                balance_data['bank_balance'],
                balance_data['settlements'],
                balance_data['adjusted_bank_balance'],
                balance_data['market_value'],
                balance_data['total_assets'],
                batch_ts_str,
            ),
        )

        conn.commit()
        balance_id = cursor.lastrowid
        conn.close()

        logger.info(f"Inserted balance record with ID {balance_id} for account {account_id}")
        return balance_id

    def get_balance_history(self, account_id, start_date, end_date):
        """
        獲取指定日期範圍內的帳戶餘額歷史記錄

        Args:
            account_id (int): 帳戶ID
            start_date (datetime.date): 開始日期
            end_date (datetime.date): 結束日期

        Returns:
            list: 餘額歷史記錄列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 使結果以字典形式返回
        cursor = conn.cursor()

        # 轉換日期格式
        start_date_str = start_date.strftime("%Y-%m-%d") + " 00:00:00"
        end_date_str = end_date.strftime("%Y-%m-%d") + " 23:59:59"

        cursor.execute(
            """
            SELECT 
                balance_id, account_id, bank_balance, settlements, 
                adjusted_bank_balance, market_value, total_assets, 
                fetch_timestamp, created_timestamp
            FROM 
                balance_history
            WHERE 
                account_id = ? AND
                fetch_timestamp BETWEEN ? AND ?
            ORDER BY 
                fetch_timestamp ASC
        """,
            (account_id, start_date_str, end_date_str),
        )

        results = cursor.fetchall()
        balance_history = [dict(row) for row in results]

        conn.close()
        return balance_history

    def get_latest_balance(self, account_id):
        """
        獲取帳戶最新的餘額記錄

        Args:
            account_id (int): 帳戶ID

        Returns:
            dict: 最新餘額記錄，如果沒有則返回None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT 
                balance_id, account_id, bank_balance, settlements, 
                adjusted_bank_balance, market_value, total_assets, 
                fetch_timestamp, created_timestamp
            FROM 
                balance_history
            WHERE 
                account_id = ?
            ORDER BY 
                fetch_timestamp DESC
            LIMIT 1
        """,
            (account_id,),
        )

        result = cursor.fetchone()
        latest_balance = dict(result) if result else None

        conn.close()
        return latest_balance

    def get_monthly_balance_data(self, account_id, start_year, end_year):
        """
        獲取每月首日和末日的餘額資料，用於計算月度報酬率

        Args:
            account_id (int): 帳戶ID
            start_year (int): 開始年份
            end_year (int): 結束年份

        Returns:
            dict: 包含每月首日和末日餘額的字典，格式為 {年份: {月份: {"start": 值, "end": 值}}}
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 獲取每個月的第一天和最後一天的資料
        query = """
        WITH monthly_data AS (
            -- 為每條記錄添加月份信息
            SELECT 
                strftime('%Y', fetch_timestamp) as year,
                strftime('%m', fetch_timestamp) as month,
                total_assets,
                fetch_timestamp,
                -- 在每個月中的排序 (1表示最早，-1表示最晚)
                ROW_NUMBER() OVER (
                    PARTITION BY strftime('%Y-%m', fetch_timestamp) 
                    ORDER BY fetch_timestamp ASC
                ) as month_start_rank,
                ROW_NUMBER() OVER (
                    PARTITION BY strftime('%Y-%m', fetch_timestamp) 
                    ORDER BY fetch_timestamp DESC
                ) as month_end_rank
            FROM 
                balance_history
            WHERE 
                account_id = ? AND
                strftime('%Y', fetch_timestamp) BETWEEN ? AND ?
        )
        -- 選取每個月的第一條和最後一條記錄
        SELECT 
            year, month, total_assets, fetch_timestamp,
            CASE 
                WHEN month_start_rank = 1 THEN 'start'
                WHEN month_end_rank = 1 THEN 'end'
            END as position
        FROM 
            monthly_data
        WHERE 
            month_start_rank = 1 OR month_end_rank = 1
        ORDER BY 
            year, month, position
        """

        cursor.execute(query, (account_id, str(start_year), str(end_year)))
        results = cursor.fetchall()

        # 整理數據為所需格式
        monthly_data = {}
        for row in results:
            year = row['year']
            month = row['month']
            position = row['position']
            total_assets = row['total_assets']

            if year not in monthly_data:
                monthly_data[year] = {}

            if month not in monthly_data[year]:
                monthly_data[year][month] = {}

            monthly_data[year][month][position] = total_assets

        conn.close()
        return monthly_data
