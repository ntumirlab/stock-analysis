"""
Data Access Object (DAO) for stock recommendations.
Handles storage and retrieval of stock recommendation data using SQLite database.
"""

import sqlite3
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class Stock:
    """Represents a single stock recommendation"""

    def __init__(
        self,
        id: str,
        sentiment: str,
        TP: Optional[float] = None,
        SL: Optional[float] = None,
        name: Optional[str] = None,
    ):
        self.id = str(id)
        self.sentiment = sentiment  # STRONG_BUY, BUY, NEUTRAL, SELL
        self.TP = TP  # Target Price
        self.SL = SL  # Stop Loss
        self.name = name

    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values"""
        result = {"id": self.id, "sentiment": self.sentiment}
        if self.TP is not None:
            result["TP"] = self.TP
        if self.SL is not None:
            result["SL"] = self.SL
        if self.name is not None:
            result["name"] = self.name
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'Stock':
        """Create Stock instance from dictionary"""
        return cls(
            id=str(data.get('id')),
            sentiment=data.get('sentiment', 'NEUTRAL'),
            TP=data.get('TP'),
            SL=data.get('SL'),
            name=data.get('name'),
        )


class RecommendationRecord:
    """Represents a recommendation record for a specific date"""

    def __init__(self, date: str, stocks: List[Stock] = None):
        self.date = date  # YYYY-MM-DD format
        self.stocks = stocks or []

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {"date": self.date, "stocks": [stock.to_dict() for stock in self.stocks]}

    @classmethod
    def from_dict(cls, data: Dict) -> 'RecommendationRecord':
        """Create RecommendationRecord instance from dictionary"""
        stocks = [Stock.from_dict(stock_data) for stock_data in data.get('stocks', [])]
        return cls(date=data.get('date'), stocks=stocks)


class RecommendationDAO:
    """Data Access Object for managing stock recommendations using SQLite"""

    def __init__(self, db_path="data_prod.db", frequency: Optional[str] = None):
        """
        Initialize DAO with database path and frequency filter

        Args:
            db_path: Path to the SQLite database file
            frequency: Filter by frequency ('weekly' or 'monthly'). If None, returns all.
        """
        self.db_path = db_path
        self.frequency = frequency
        self._create_table()

    def _create_table(self):
        """建立 recommendation_stocks 資料表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Single table design: recommendation_stocks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_stocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                frequency TEXT NOT NULL,
                priority INTEGER NOT NULL,
                stock_id TEXT NOT NULL,
                stock_name TEXT,
                sentiment TEXT NOT NULL,
                target_price REAL,
                stop_loss REAL,
                created_timestamp TEXT DEFAULT (datetime('now','localtime')),
                updated_timestamp TEXT DEFAULT (datetime('now','localtime'))
            );
        """)

        # Create indexes for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_recommendation_stocks_date 
            ON recommendation_stocks(date DESC);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_recommendation_stocks_stock_id 
            ON recommendation_stocks(stock_id);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_recommendation_stocks_date_stock 
            ON recommendation_stocks(date, stock_id);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_recommendation_stocks_frequency 
            ON recommendation_stocks(frequency);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_recommendation_stocks_freq_date 
            ON recommendation_stocks(frequency, date DESC);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_recommendation_stocks_date_priority 
            ON recommendation_stocks(date, priority);
        """)

        conn.commit()
        conn.close()

    def load(self) -> List[RecommendationRecord]:
        """
        Load all recommendation records from database

        Returns:
            List of RecommendationRecord objects sorted by date
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all unique dates (filtered by frequency if set)
        if self.frequency:
            cursor.execute(
                """
                SELECT DISTINCT date FROM recommendation_stocks 
                WHERE frequency = ?
                ORDER BY date ASC
            """,
                (self.frequency,),
            )
        else:
            cursor.execute("""
                SELECT DISTINCT date FROM recommendation_stocks 
                ORDER BY date ASC
            """)

        records = []
        for row in cursor.fetchall():
            date = row['date']

            # Get stocks for this date (filtered by frequency if set)
            if self.frequency:
                cursor.execute(
                    """
                    SELECT stock_id, stock_name, sentiment, target_price, stop_loss
                    FROM recommendation_stocks
                    WHERE date = ? AND frequency = ?
                    ORDER BY priority ASC
                """,
                    (date, self.frequency),
                )
            else:
                cursor.execute(
                    """
                    SELECT stock_id, stock_name, sentiment, target_price, stop_loss
                    FROM recommendation_stocks
                    WHERE date = ?
                    ORDER BY priority ASC
                """,
                    (date,),
                )

            stocks = [
                Stock(
                    id=stock_row['stock_id'],
                    name=stock_row['stock_name'],
                    sentiment=stock_row['sentiment'],
                    TP=stock_row['target_price'],
                    SL=stock_row['stop_loss'],
                )
                for stock_row in cursor.fetchall()
            ]

            records.append(RecommendationRecord(date=date, stocks=stocks))

        conn.close()
        return records

    def save(self, records: List[RecommendationRecord]) -> None:
        """
        Save recommendation records to database (full replacement for this frequency)

        Args:
            records: List of RecommendationRecord objects to save
        """
        if not self.frequency:
            raise ValueError("frequency must be set when calling save()")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Clear existing data for this frequency only
        cursor.execute("DELETE FROM recommendation_stocks WHERE frequency = ?", (self.frequency,))

        # Insert new records
        sorted_records = sorted(records, key=lambda r: r.date)
        for record in sorted_records:
            for idx, stock in enumerate(record.stocks):
                cursor.execute(
                    """
                    INSERT INTO recommendation_stocks 
                    (date, frequency, priority, stock_id, stock_name, sentiment, target_price, stop_loss)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (record.date, self.frequency, idx, stock.id, stock.name, stock.sentiment, stock.TP, stock.SL),
                )

        conn.commit()
        conn.close()
        logger.info(f"Saved {len(records)} recommendation records to database")

    def add_record(self, record: RecommendationRecord) -> None:
        """
        Add or update a single recommendation record

        Args:
            record: RecommendationRecord to add
        """
        if not self.frequency:
            raise ValueError("frequency must be set when calling add_record()")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete existing stocks for this date and frequency
        cursor.execute(
            "DELETE FROM recommendation_stocks WHERE date = ? AND frequency = ?", (record.date, self.frequency)
        )

        # Insert stocks
        for idx, stock in enumerate(record.stocks):
            cursor.execute(
                """
                INSERT INTO recommendation_stocks 
                (date, frequency, priority, stock_id, stock_name, sentiment, target_price, stop_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (record.date, self.frequency, idx, stock.id, stock.name, stock.sentiment, stock.TP, stock.SL),
            )

        conn.commit()
        conn.close()
        logger.info(f"Added/updated {self.frequency} recommendation for {record.date} with {len(record.stocks)} stocks")

    def get_by_date(self, date: str) -> Optional[RecommendationRecord]:
        """
        Get recommendation record by date

        Args:
            date: Date string in YYYY-MM-DD format

        Returns:
            RecommendationRecord or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if self.frequency:
            cursor.execute(
                """
                SELECT stock_id, stock_name, sentiment, target_price, stop_loss
                FROM recommendation_stocks
                WHERE date = ? AND frequency = ?
                ORDER BY priority ASC
            """,
                (date, self.frequency),
            )
        else:
            cursor.execute(
                """
                SELECT stock_id, stock_name, sentiment, target_price, stop_loss
                FROM recommendation_stocks
                WHERE date = ?
                ORDER BY priority ASC
            """,
                (date,),
            )

        rows = cursor.fetchall()

        if not rows:
            conn.close()
            return None

        stocks = [
            Stock(
                id=row['stock_id'],
                name=row['stock_name'],
                sentiment=row['sentiment'],
                TP=row['target_price'],
                SL=row['stop_loss'],
            )
            for row in rows
        ]

        conn.close()
        return RecommendationRecord(date=date, stocks=stocks)

    def get_latest(self) -> Optional[RecommendationRecord]:
        """
        Get the latest recommendation record

        Returns:
            Latest RecommendationRecord or None if empty
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get latest date (filtered by frequency if set)
        if self.frequency:
            cursor.execute(
                """
                SELECT DISTINCT date FROM recommendation_stocks 
                WHERE frequency = ?
                ORDER BY date DESC LIMIT 1
            """,
                (self.frequency,),
            )
        else:
            cursor.execute("""
                SELECT DISTINCT date FROM recommendation_stocks 
                ORDER BY date DESC LIMIT 1
            """)

        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        date = row['date']

        if self.frequency:
            cursor.execute(
                """
                SELECT stock_id, stock_name, sentiment, target_price, stop_loss
                FROM recommendation_stocks
                WHERE date = ? AND frequency = ?
                ORDER BY priority ASC
            """,
                (date, self.frequency),
            )
        else:
            cursor.execute(
                """
                SELECT stock_id, stock_name, sentiment, target_price, stop_loss
                FROM recommendation_stocks
                WHERE date = ?
                ORDER BY priority ASC
            """,
                (date,),
            )

        stocks = [
            Stock(
                id=stock_row['stock_id'],
                name=stock_row['stock_name'],
                sentiment=stock_row['sentiment'],
                TP=stock_row['target_price'],
                SL=stock_row['stop_loss'],
            )
            for stock_row in cursor.fetchall()
        ]

        conn.close()
        return RecommendationRecord(date=date, stocks=stocks)

    def get_stock_ids(self, date: str) -> List[str]:
        """
        Get list of stock IDs for a specific date

        Args:
            date: Date string in YYYY-MM-DD format

        Returns:
            List of stock ID strings
        """
        record = self.get_by_date(date)
        return [stock.id for stock in record.stocks] if record else []

    def delete_by_date(self, date: str) -> None:
        """
        Delete recommendation record by date

        Args:
            date: Date string in YYYY-MM-DD format
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if self.frequency:
            cursor.execute("DELETE FROM recommendation_stocks WHERE date = ? AND frequency = ?", (date, self.frequency))
            logger.info(f"Deleted {self.frequency} recommendation for {date}")
        else:
            cursor.execute("DELETE FROM recommendation_stocks WHERE date = ?", (date,))
            logger.info(f"Deleted recommendation for {date}")

        conn.commit()
        conn.close()
