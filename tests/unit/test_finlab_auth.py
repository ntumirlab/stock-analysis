"""utils.finlab_auth 的登入路徑選擇測試。

正式排程（jobs.scheduler、jobs.order_executor、jobs.profit_loss_fetcher）啟動後
第一件事就是這支登入，選錯路徑或誤拋例外的後果是整批 job 失敗，故每個分支都釘住。

兩個不可退讓的行為：
1. session 憑證可用時**不呼叫** `finlab.login()`——該函式在憑證失效時會退回瀏覽器
   流程，容器內會卡到逾時。
2. session 憑證失效時**自動退回** `FINLAB_API_TOKEN`——client 端（lite）無人可即時
   處理認證問題，這條 fallback 是服務不中斷的保證。

CI 的 requirements-dev.txt 不安裝 finlab（見該檔註解），因此以 sys.modules 注入假
模組；本檔只驗本專案的分支邏輯，不驗 finlab 套件本身。
"""

import sys
import types

import pytest

SESSION_VARS = ("FINLAB_REFRESH_TOKEN", "FINLAB_SESSION_ID", "FINLAB_API_KEY")
ALL_VARS = SESSION_VARS + ("FINLAB_API_TOKEN",)


@pytest.fixture
def finlab_auth(monkeypatch):
    """以假 finlab 模組載入 utils.finlab_auth，回傳 (module, fake_finlab)。

    fake_finlab.login_calls 記錄 legacy 路徑實際收到的 token；
    fake_finlab.auth.get_id_token 可由各測試改寫以模擬憑證狀態。
    """
    fake_finlab = types.ModuleType("finlab")
    fake_finlab.login_calls = []
    fake_finlab.login = lambda token: fake_finlab.login_calls.append(token)

    fake_auth = types.ModuleType("finlab.auth")
    fake_auth.get_id_token = lambda: "fake-id-token"
    fake_finlab.auth = fake_auth

    monkeypatch.setitem(sys.modules, "finlab", fake_finlab)
    monkeypatch.setitem(sys.modules, "finlab.auth", fake_auth)

    for var in ALL_VARS:
        monkeypatch.delenv(var, raising=False)

    sys.modules.pop("utils.finlab_auth", None)
    import utils.finlab_auth as module

    yield module, fake_finlab

    sys.modules.pop("utils.finlab_auth", None)


def _set_session_vars(monkeypatch):
    for var in SESSION_VARS:
        monkeypatch.setenv(var, f"value-of-{var}")


def test_session_credentials_skip_legacy_login(finlab_auth, monkeypatch):
    # 走到 finlab.login() 就等於容器有機會卡在瀏覽器流程，這是本模組存在的理由
    module, fake = finlab_auth
    _set_session_vars(monkeypatch)
    monkeypatch.setenv("FINLAB_API_TOKEN", "legacy-token")

    module.login_finlab()

    assert fake.login_calls == []


def test_missing_one_session_var_falls_back_to_api_token(finlab_auth, monkeypatch):
    # 三個變數是一組，缺一即視為未設定（get_session 的判定也是三個都要）
    module, fake = finlab_auth
    _set_session_vars(monkeypatch)
    monkeypatch.delenv("FINLAB_SESSION_ID")
    monkeypatch.setenv("FINLAB_API_TOKEN", "legacy-token")

    module.login_finlab()

    assert fake.login_calls == ["legacy-token"]


def test_empty_session_var_falls_back_to_api_token(finlab_auth, monkeypatch):
    # .env 範本的欄位是空字串而非缺鍵，未填寫時必須與「沒設定」等價
    module, fake = finlab_auth
    _set_session_vars(monkeypatch)
    monkeypatch.setenv("FINLAB_REFRESH_TOKEN", "")
    monkeypatch.setenv("FINLAB_API_TOKEN", "legacy-token")

    module.login_finlab()

    assert fake.login_calls == ["legacy-token"]


def test_id_token_none_falls_back_to_api_token(finlab_auth, monkeypatch):
    # refresh token 被撤銷時 get_id_token() 回 None（不丟例外），服務須續行
    module, fake = finlab_auth
    _set_session_vars(monkeypatch)
    monkeypatch.setenv("FINLAB_API_TOKEN", "legacy-token")
    monkeypatch.setattr(fake.auth, "get_id_token", lambda: None)

    module.login_finlab()

    assert fake.login_calls == ["legacy-token"]


def test_id_token_exception_falls_back_to_api_token(finlab_auth, monkeypatch):
    # 換發過程的網路例外不可往上炸掉整支 job
    module, fake = finlab_auth
    _set_session_vars(monkeypatch)
    monkeypatch.setenv("FINLAB_API_TOKEN", "legacy-token")

    def _boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(fake.auth, "get_id_token", _boom)

    module.login_finlab()

    assert fake.login_calls == ["legacy-token"]


def test_no_credentials_raises_with_variable_names(finlab_auth):
    # 訊息要能讓值班的人直接知道該補哪些變數
    module, _ = finlab_auth

    with pytest.raises(EnvironmentError) as exc_info:
        module.login_finlab()

    message = str(exc_info.value)
    for var in SESSION_VARS:
        assert var in message


def test_empty_api_token_argument_raises(finlab_auth, monkeypatch):
    # Authenticator 傳的是 get_env_var() 的結果：.env 留空時為 ""，不可拿空字串去登入
    module, fake = finlab_auth

    with pytest.raises(EnvironmentError):
        module.login_finlab("")

    assert fake.login_calls == []


def test_explicit_api_token_wins_over_environment(finlab_auth, monkeypatch):
    # Authenticator 走 ConfigLoader 取值，明確傳入時以參數為準
    module, fake = finlab_auth
    monkeypatch.setenv("FINLAB_API_TOKEN", "from-environment")

    module.login_finlab("from-argument")

    assert fake.login_calls == ["from-argument"]


def test_login_rejecting_api_token_raises_environment_error(finlab_auth, monkeypatch):
    # finlab 真的移除 api_token 後，要給可行動的訊息而非 finlab 內部的 TypeError
    module, fake = finlab_auth
    monkeypatch.setenv("FINLAB_API_TOKEN", "legacy-token")

    def _rejects(token):
        raise TypeError("login() takes 0 positional arguments")

    monkeypatch.setattr(fake, "login", _rejects)

    with pytest.raises(EnvironmentError) as exc_info:
        module.login_finlab()

    for var in SESSION_VARS:
        assert var in str(exc_info.value)
