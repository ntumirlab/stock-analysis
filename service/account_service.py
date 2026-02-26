from dao.account_dao import AccountDAO


class AccountService:
    def __init__(self, db_path="data_prod.db"):
        self.account_dao = AccountDAO(db_path=db_path)

    def get_all_accounts(self):
        accounts = self.account_dao.get_all_accounts()
        return accounts
