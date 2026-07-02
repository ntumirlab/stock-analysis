"""AccountDAO 測試（真實 SQLite on tmp file）。"""

import pytest

from dao.account_dao import AccountDAO


@pytest.fixture
def dao(tmp_path):
    return AccountDAO(db_path=str(tmp_path / "test.db"))


def test_get_account_id_creates_then_reuses(dao):
    first = dao.get_account_id("kiri_shioaji", broker_name="shioaji", user_name="kiri")
    second = dao.get_account_id("kiri_shioaji", broker_name="shioaji", user_name="kiri")
    assert first == second


def test_different_accounts_get_different_ids(dao):
    a = dao.get_account_id("junting_shioaji", broker_name="shioaji", user_name="junting")
    b = dao.get_account_id("kiri_shioaji", broker_name="shioaji", user_name="kiri")
    assert a != b


def test_get_all_accounts_newest_first(dao):
    # dashboard 預設選第一筆，因此必須是最新建立的帳戶（account_id DESC）
    dao.get_account_id("junting_shioaji", broker_name="shioaji", user_name="junting")
    dao.get_account_id("kiri_shioaji", broker_name="shioaji", user_name="kiri")

    accounts = dao.get_all_accounts()
    assert accounts[0]["account_name"] == "kiri_shioaji"
    assert accounts[1]["account_name"] == "junting_shioaji"


def test_get_all_accounts_fields(dao):
    dao.get_account_id("kiri_shioaji", broker_name="shioaji", user_name="kiri")
    acc = dao.get_all_accounts()[0]
    assert acc["broker_name"] == "shioaji"
    assert acc["user_name"] == "kiri"
    assert acc["created_timestamp"]
