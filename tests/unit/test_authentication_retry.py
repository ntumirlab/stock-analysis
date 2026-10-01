"""utils.authentication 的 shioaji 登入重試測試。

08:10 下單一天只跑一次，登入一次逾時就整天沒下單（2026-10-01）。這裡釘住：
暫時失敗會重試、失敗的 session 會登出、重試成功發 TG 警告、用盡才拋錯。

utils.authentication 在 module 層 import finlab / keyring，CI 不安裝這些，
因此以 sys.modules 注入假模組（手法同 test_finlab_auth.py）。
"""

import sys
import types

import pytest


class _FakeApi:
    def __init__(self):
        self.logged_out = False

    def logout(self):
        self.logged_out = True
        return True


class _FakeSinopacAccount:
    """依 outcomes 腳本決定每次 __init__ 成功或丟例外；instances 記錄所有建立過的物件。"""

    outcomes = []
    instances = []

    def __init__(self):
        type(self).instances.append(self)
        # 比照真實 SinopacAccount：先建 api 再 login，login 失敗時 api 已存在
        self.api = _FakeApi()
        outcome = type(self).outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome


class _FakeNotifier:
    def __init__(self):
        self.warnings = []

    def send_warning(self, **kwargs):
        self.warnings.append(kwargs)
        return True


class _FakeConfigLoader:
    config = {"notification": {}}

    def get_env_var(self, key):
        import os
        return os.environ.get(key)


@pytest.fixture
def auth_module(monkeypatch, tmp_path):
    fake_finlab = types.ModuleType("finlab")
    fake_online = types.ModuleType("finlab.online")
    fake_sinopac = types.ModuleType("finlab.online.sinopac_account")
    fake_sinopac.SinopacAccount = _FakeSinopacAccount
    fake_finlab_auth = types.ModuleType("utils.finlab_auth")
    fake_finlab_auth.login_finlab = lambda token=None: None

    monkeypatch.setitem(sys.modules, "finlab", fake_finlab)
    monkeypatch.setitem(sys.modules, "finlab.online", fake_online)
    monkeypatch.setitem(sys.modules, "finlab.online.sinopac_account", fake_sinopac)
    monkeypatch.setitem(sys.modules, "keyring", types.ModuleType("keyring"))
    monkeypatch.setitem(sys.modules, "utils.finlab_auth", fake_finlab_auth)

    cert = tmp_path / "cert.pfx"
    cert.write_bytes(b"")
    for var in ("SHIOAJI_API_KEY", "SHIOAJI_SECRET_KEY", "SHIOAJI_CERT_PERSON_ID", "SHIOAJI_CERT_PASSWORD"):
        monkeypatch.setenv(var, "x")
    monkeypatch.setenv("SHIOAJI_CERT_PATH", str(cert))

    _FakeSinopacAccount.outcomes = []
    _FakeSinopacAccount.instances = []

    sys.modules.pop("utils.authentication", None)
    import utils.authentication as module

    # 測試不真的等待
    monkeypatch.setattr(module, "SHIOAJI_LOGIN_RETRY_DELAYS", (0, 0, 0))
    notifier = _FakeNotifier()
    monkeypatch.setattr(module, "create_notification_manager", lambda config, logger=None: notifier)

    yield module, notifier

    sys.modules.pop("utils.authentication", None)


def _login(module):
    return module.Authenticator(_FakeConfigLoader()).login_broker("shioaji")


def test_first_try_success_sends_no_warning(auth_module):
    module, notifier = auth_module
    _FakeSinopacAccount.outcomes = [None]

    account = _login(module)

    assert account is _FakeSinopacAccount.instances[0]
    assert notifier.warnings == []


def test_retries_after_timeouts_and_logs_out_failed_sessions(auth_module):
    module, notifier = auth_module
    _FakeSinopacAccount.outcomes = [TimeoutError("error_tracking"), TimeoutError("again"), None]

    account = _login(module)

    failed, succeeded = _FakeSinopacAccount.instances[:2], _FakeSinopacAccount.instances[2]
    assert account is succeeded
    # 失敗那幾次已建立的 session 要登出；成功的那個不能被登出
    assert all(a.api.logged_out for a in failed)
    assert not succeeded.api.logged_out


def test_retry_success_sends_one_warning(auth_module):
    module, notifier = auth_module
    _FakeSinopacAccount.outcomes = [TimeoutError("error_tracking"), None]

    _login(module)

    assert len(notifier.warnings) == 1
    warning = notifier.warnings[0]
    assert warning["broker_name"] == "shioaji"
    assert "失敗 1 次後重試成功" in warning["body"]
    assert "TimeoutError: error_tracking" in warning["body"]


def test_exhausted_retries_raise_last_error_without_warning(auth_module):
    module, notifier = auth_module
    last = TimeoutError("t4")
    _FakeSinopacAccount.outcomes = [TimeoutError("t1"), TimeoutError("t2"), TimeoutError("t3"), last]

    with pytest.raises(TimeoutError) as excinfo:
        _login(module)

    assert excinfo.value is last
    assert len(_FakeSinopacAccount.instances) == 4
    # 失敗由 job 的 send_error 回報，這裡不另發警告
    assert notifier.warnings == []


def test_logout_failure_does_not_stop_retry(auth_module, monkeypatch):
    module, _ = auth_module
    _FakeSinopacAccount.outcomes = [TimeoutError("t1"), None]

    def broken_logout(self):
        raise ConnectionError("solace down")

    monkeypatch.setattr(_FakeApi, "logout", broken_logout)

    assert _login(module) is _FakeSinopacAccount.instances[1]


def test_warning_failure_does_not_break_login(auth_module, monkeypatch):
    module, _ = auth_module
    _FakeSinopacAccount.outcomes = [TimeoutError("t1"), None]

    def broken_factory(config, logger=None):
        raise RuntimeError("telegram config broken")

    monkeypatch.setattr(module, "create_notification_manager", broken_factory)

    assert _login(module) is _FakeSinopacAccount.instances[1]


def test_missing_env_var_fails_fast_without_retry(auth_module, monkeypatch):
    # 我們自己的設定檢查在重試迴圈之外：缺設定重試也沒用
    module, _ = auth_module
    monkeypatch.delenv("SHIOAJI_API_KEY")

    with pytest.raises(EnvironmentError):
        _login(module)

    assert _FakeSinopacAccount.instances == []
