from dao.inventory_dao import InventoryDAO


class InventoryService:
    def __init__(self):
        self.inventory_dao = InventoryDAO()

    def get_inventories_by_account_and_date(self, account_id, query_date):
        raw_inventories = self.inventory_dao.get_inventories_by_account_and_date(account_id, query_date)
        filtered_inventories = []
        for inventory in raw_inventories:
            filtered_inventories.append(
                {
                    "stock_id": inventory.get("stock_id"),
                    "stock_name": inventory.get("stock_name"),
                    "quantity": inventory.get("quantity"),
                    "last_price": inventory.get("last_price"),
                    "pnl": inventory.get("pnl"),
                }
            )
        return filtered_inventories
