import datetime
from dao.order_dao import OrderDAO


class OrderService:
    def __init__(self, db_path="data_prod.db"):
        self.order_dao = OrderDAO(db_path=db_path)

    def get_orders_by_account_and_date(self, account_id, query_date):
        raw_orders = self.order_dao.get_orders_by_account_and_date(account_id, query_date)
        filtered_orders = []
        for order in raw_orders:
            filtered_orders.append(
                {
                    "order_timestamp": order.get("order_timestamp"),
                    "action": order.get("action"),
                    "stock_id": order.get("stock_id"),
                    "stock_name": order.get("stock_name"),
                    "quantity": order.get("quantity"),
                    "limit_price": order.get("limit_price"),
                    "extra_bid_pct": order.get("extra_bid_pct"),
                    "order_condition": order.get("order_condition"),
                    "view_only": order.get("view_only"),
                }
            )
        return filtered_orders

    def get_available_years(self, account_id):
        """取得可用年份並轉成下拉選單格式"""
        years = self.order_dao.get_available_years(account_id)
        return [{'label': year, 'value': year} for year in years]

    def get_available_months(self, account_id, year):
        """取得可用月份並轉成下拉選單格式"""
        months = self.order_dao.get_available_months(account_id, year)
        return [{'label': month, 'value': month} for month in months]

    def get_available_days(self, account_id, year, month):
        """取得可用日期並轉成下拉選單格式"""
        days = self.order_dao.get_available_days(account_id, year, month)
        return [{'label': day, 'value': day} for day in days]


if __name__ == "__main__":
    order_service = OrderService()
    date_str = f"2025-03-05"
    query_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    orders = order_service.get_orders_by_account_and_date(account_id=1, query_date=query_date)
    print(orders)
