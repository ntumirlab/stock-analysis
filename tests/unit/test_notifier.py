"""utils/notifier 的單元測試：標題前綴（Lite 與正式機訊息的區分）、敏感資訊遮罩。"""

import sys
import types

import pytest

from utils.notifier import NotificationManager, TelegramNotifier, redact


class _StubTelegram:
    def __init__(self):
        self.messages = []

    def send_message(self, message, parse_mode='Markdown'):
        self.messages.append(message)
        return True


def _manager(config):
    manager = NotificationManager(config)
    manager.enabled = True
    manager.telegram = _StubTelegram()
    return manager


def _send_all_levels(manager):
    manager.send_success(task_name="t", body="b")
    manager.send_warning(task_name="t", body="b")
    manager.send_error(task_name="t", error_message="boom")
    return [m.splitlines()[0] for m in manager.telegram.messages]


def test_title_prefix_applied_to_all_levels():
    titles = _send_all_levels(_manager({'title_prefix': '[Lite] '}))
    assert titles == [
        "✅ *[Lite] 股票系統通知*",
        "⚠️ *[Lite] 股票系統警告*",
        "🚨 *[Lite] 股票系統錯誤通知*",
    ]


def test_titles_unchanged_without_prefix():
    # 正式機 config 沒有 title_prefix，訊息必須與加前綴功能之前完全相同
    titles = _send_all_levels(_manager({}))
    assert titles == [
        "✅ *股票系統通知*",
        "⚠️ *股票系統警告*",
        "🚨 *股票系統錯誤通知*",
    ]


def test_null_prefix_treated_as_no_prefix():
    # Lite 範本自帶 title_prefix key；client 把值清空（YAML null）時
    # 必須回到無前綴，不得把 None 渲染進標題
    titles = _send_all_levels(_manager({'title_prefix': None}))
    assert titles == [
        "✅ *股票系統通知*",
        "⚠️ *股票系統警告*",
        "🚨 *股票系統錯誤通知*",
    ]


# ---- 敏感資訊遮罩 ----
# 範例值皆為虛構；格式比照 2026-10-01 shioaji 登入逾時的錯誤訊息

FAKE_ID = "A123456789"
FAKE_JWT = "eyJ0eXAiOiJKV1QifQ.eyJwZXJzb25faWQiOiJBMTIzNDU2Nzg5In0.c2lnbmF0dXJlLXBhcnQ"


def test_redact_shioaji_timeout_message():
    text = (
        f"TimeoutError: Topic: api/v1/auth/error_tracking, Corr: c2, "
        f"Client: PYAPI/{FAKE_ID}/1001/001003/377474/140.112.174.201, "
        f"payload: {{'token': '{FAKE_JWT}', 'person_id': '{FAKE_ID}'}}"
    )
    out = redact(text)
    assert FAKE_ID not in out
    assert FAKE_JWT not in out
    assert "PYAPI/A#######89/1001" in out
    assert "'token': '<token>'" in out
    # 除錯需要的部分保留
    assert "api/v1/auth/error_tracking" in out


@pytest.mark.parametrize("pid", ["A123456789", "F287654321", "A812345678", "AC12345678"])
def test_redact_person_id_formats(pid):
    # 身分證（第二碼 1/2）、新式居留證（8/9）、舊式居留證（A-D）
    assert pid not in redact(f"id={pid}")


@pytest.mark.parametrize("text", [
    "order 2330 qty 1000",
    "A12345678",        # 只有 8 位數字
    "A1234567890",      # 11 位數字（較長數字的一部分不算）
    "XA123456789",      # 前面接英數字（較長識別碼的一部分）
    "A323456789",       # 第二碼不合法
])
def test_redact_leaves_non_ids_alone(text):
    assert redact(text) == text


def test_redact_env_secret_by_name_part(monkeypatch):
    # 名稱有前綴（KIRI_）也要遮：比對的是名稱片段而非完整名稱
    monkeypatch.setenv("KIRI_SHIOAJI_SECRET_KEY", "s3cr3t-value-123")
    monkeypatch.setenv("FUGLE_ACCOUNT", "98765432")
    monkeypatch.setenv("SOME_SERVICE_PASSWORD", "hunter2hunter2")
    out = redact("key=s3cr3t-value-123 acct=98765432 pw=hunter2hunter2")
    assert out == "key=<redacted> acct=<redacted> pw=<redacted>"


def test_redact_skips_chat_id_paths_and_short_values(monkeypatch):
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1001234567890")
    monkeypatch.setenv("GOOGLE_TOKEN_PATH", "/app/config/credentials/token.json")
    monkeypatch.setenv("TINY_TOKEN", "true")
    text = "chat -1001234567890 path /app/config/credentials/token.json flag true"
    assert redact(text) == text


def test_redact_longer_secret_first(monkeypatch):
    # 某值是另一值的子字串時，長的要先整段遮掉，不可只遮一半
    monkeypatch.setenv("A_API_KEY", "abcdefgh")
    monkeypatch.setenv("B_API_KEY", "abcdefgh-ijklmnop")
    assert redact("v=abcdefgh-ijklmnop") == "v=<redacted>"


def test_send_message_redacts_before_posting(monkeypatch):
    posted = []

    class _Response:
        def raise_for_status(self):
            pass

    fake_requests = types.ModuleType("requests")
    fake_requests.post = lambda url, json, timeout: posted.append(json) or _Response()
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    assert TelegramNotifier("bot-token", "chat").send_message(f"boom {FAKE_ID} {FAKE_JWT}")
    assert posted[0]["text"] == "boom A#######89 <token>"
