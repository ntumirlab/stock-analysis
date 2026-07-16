"""utils/notifier 的單元測試：標題前綴（Lite 與正式機訊息的區分）。"""

from utils.notifier import NotificationManager


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
