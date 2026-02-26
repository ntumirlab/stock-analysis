import unittest
import sys
import os
import sqlite3
from service.order_service import OrderService

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestOrderService(unittest.TestCase):
    def setUp(self):
        self.db_path = ":memory:"
        self.connection = sqlite3.connect(self.db_path)
        self.create_tables()
        self.insert_test_data()
        self.order_service = OrderService(db_path=self.db_path)

    def tearDown(self):
        self.connection.close()

    def create_tables(self):
        cursor = self.connection.cursor()
        cursor.execute('''
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                account_id INTEGER,
                date TEXT,
                amount INTEGER
            )
        ''')
        self.connection.commit()

    def insert_test_data(self):
        cursor = self.connection.cursor()
        cursor.execute('''
            INSERT INTO orders (account_id, date, amount) VALUES
            (1, '2023-10-01', 100),
            (1, '2023-10-01', 200)
        ''')
        self.connection.commit()


if __name__ == "__main__":
    unittest.main()
