"""
通知系統模組 - Telegram 通知

三個等級：send_success（✅ 下單摘要、清單入庫）、send_warning（⚠️ 非致命異常，
如週日沒新清單）、send_error（🚨 job 掛掉）。無事件的日子保持安靜。
採用可擴展架構，未來可輕鬆新增其他通知渠道。

Author: Stock Analysis System
Date: 2025-11-01
"""

import logging
import os
import re
from datetime import datetime
from typing import Optional, Dict, Any
from zoneinfo import ZoneInfo


# ---- 送出前遮罩敏感資訊 ----
# 錯誤訊息與 traceback 會原樣進 TG 群組（2026-10-01 的 shioaji 逾時把 JWT 與
# 身分證字號整段貼出）。兩層：格式固定的直接比對；其餘密鑰從環境變數取值比對。
# 只作用於 TG 訊息，log 檔維持完整內容以便除錯。

# 身分證／居留證：英文字母 + (1|2|8|9|A-D) + 8 位數字。保留頭尾辨識是哪個帳戶
_PERSON_ID_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z])[1289A-D]\d{6}(\d{2})(?![0-9])")
# JWT：三段 base64url，header 一律以 eyJ（'{"' 的編碼）開頭
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*")

# 變數名稱含這些字即視為敏感。比對名稱片段而非完整名稱：同一個值在不同部署
# 可能叫 SHIOAJI_API_KEY 或 KIRI_SHIOAJI_API_KEY。TELEGRAM_CHAT_ID 刻意不遮
_SECRET_NAME_PARTS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "SESSION", "PERSON_ID", "ACCOUNT")
# 路徑類變數（如 GOOGLE_TOKEN_PATH）的值不是秘密，遮了反而妨礙除錯
_NON_SECRET_NAME_SUFFIXES = ("_PATH", "_DIR")
# 太短的值會把訊息中不相干的同字串一起換掉（如值為 "1"、"true"）
_MIN_SECRET_LENGTH = 8
_REDACTED = "<redacted>"


def _secret_env_values() -> list:
    values = set()
    for name, value in os.environ.items():
        upper = name.upper()
        if upper.endswith(_NON_SECRET_NAME_SUFFIXES):
            continue
        if any(part in upper for part in _SECRET_NAME_PARTS) and len(value) >= _MIN_SECRET_LENGTH:
            values.add(value)
    # 長的先換，避免某值是另一值的子字串時只遮掉一半
    return sorted(values, key=len, reverse=True)


def redact(text: str) -> str:
    """遮罩 TG 訊息中的 token、身分證字號與環境變數中的密鑰。"""
    for value in _secret_env_values():
        text = text.replace(value, _REDACTED)
    text = _JWT_RE.sub("<token>", text)
    # 遮罩字元用 #：* 和 _ 在 legacy Markdown 會與訊息其他符號配對
    text = _PERSON_ID_RE.sub(r"\1#######\2", text)
    return text


class TelegramNotifier:
    """Telegram Bot 通知類別"""

    def __init__(self, bot_token: str, chat_id: str, logger: Optional[logging.Logger] = None):
        """
        初始化 Telegram 通知器

        Args:
            bot_token: Telegram Bot Token (從 @BotFather 取得)
            chat_id: Telegram Chat ID (發送目標)
            logger: 日誌記錄器（可選）
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.logger = logger or logging.getLogger(__name__)

    def send_message(self, message: str, parse_mode: str = 'Markdown') -> bool:
        """
        發送文字訊息

        Args:
            message: 訊息內容（支援 Markdown 格式）
            parse_mode: 解析模式 ('Markdown' 或 'HTML')

        Returns:
            bool: 發送成功返回 True
        """
        try:
            import requests

            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                # 在唯一的送出點遮罩，所有等級與所有 job 的通知都一併涵蓋
                "text": redact(message),
                "parse_mode": parse_mode
            }

            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()

            self.logger.debug("Telegram 訊息發送成功")
            return True

        except Exception as e:
            self.logger.error(f"Telegram 訊息發送失敗: {e}")
            return False


class NotificationManager:
    """
    通知管理器

    統一管理通知的發送（success / warning / error 三個等級），目前支援 Telegram。
    客戶可透過 config.yaml 關閉通知功能。
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        初始化通知管理器

        Args:
            config: config.yaml 中的 notification 設定
            logger: 日誌記錄器（可選）
        """
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = config.get('enabled', False)
        # 選配的標題前綴（如 "[Lite] "）：區分不同部署（Lite / 正式機）發出的訊息。
        # 用 or '' 而非 get 預設值：key 存在但值留空（YAML null）也要回到無前綴
        self.title_prefix = config.get('title_prefix') or ''

        # 初始化 Telegram
        self.telegram = None
        if self.enabled:
            telegram_config = config.get('telegram', {})
            bot_token = telegram_config.get('bot_token')
            chat_id = telegram_config.get('chat_id')

            if bot_token and chat_id:
                self.telegram = TelegramNotifier(bot_token, chat_id, self.logger)
                self.logger.info("Telegram 通知已啟用")
            else:
                self.logger.warning("Telegram 設定不完整 (bot_token 或 chat_id 未設定)，通知功能將被停用")
                self.enabled = False

    def _send_notice(
        self,
        emoji: str,
        title: str,
        task_name: str,
        body: str,
        user_name: Optional[str] = None,
        broker_name: Optional[str] = None
    ) -> bool:
        """組合通用版面（標題/時間/任務/用戶/券商 + 內文）並發送"""
        if not self.is_enabled():
            return False

        timestamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")

        message = f"{emoji} *{self.title_prefix}{title}*\n\n"
        message += f"📅 *時間*: `{timestamp}`\n"
        # 動態欄位以反引號包住：legacy Markdown 不支援 \ 跳脫，
        # 值含 _ / * 時會與訊息其他符號配對成斜體段（inline code 內不解析）
        message += f"📋 *任務*: `{task_name}`\n"

        if user_name:
            message += f"👤 *用戶*: `{user_name}`\n"
        if broker_name:
            message += f"📊 *券商*: `{broker_name}`\n"

        message += f"\n{body}"

        return self.telegram.send_message(message)

    def send_success(
        self,
        task_name: str,
        body: str,
        user_name: Optional[str] = None,
        broker_name: Optional[str] = None
    ) -> bool:
        """
        發送成功/資訊通知（如下單摘要、清單解析入庫）

        Args:
            task_name: 任務名稱（如 "早盤下單摘要"、"清單解析 (weekly)"）
            body: 已格式化的內文（見 core/notification_formats.py）
            user_name: 使用者名稱（可選）
            broker_name: 券商名稱（可選）
        """
        return self._send_notice("✅", "股票系統通知", task_name, body, user_name, broker_name)

    def send_warning(
        self,
        task_name: str,
        body: str,
        user_name: Optional[str] = None,
        broker_name: Optional[str] = None
    ) -> bool:
        """
        發送警告通知（非致命異常，如週日沒有新清單、清單解析失敗）

        參數同 send_success。
        """
        return self._send_notice("⚠️", "股票系統警告", task_name, body, user_name, broker_name)

    def send_error(
        self,
        task_name: str,
        error_message: str,
        user_name: Optional[str] = None,
        broker_name: Optional[str] = None,
        error_traceback: Optional[str] = None
    ) -> bool:
        """
        發送錯誤通知

        Args:
            task_name: 任務名稱（如 "早盤下單"、"回測執行"、"帳務抓取"）
            error_message: 錯誤訊息
            user_name: 使用者名稱（可選）
            broker_name: 券商名稱（可選）
            error_traceback: 完整的 traceback（可選）

        Returns:
            bool: 發送成功返回 True

        Example:
            >>> notifier.send_error(
            ...     task_name="早盤下單",
            ...     error_message="'PortfolioSyncManager' object has no attribute 'order_executor'",
            ...     user_name="junting",
            ...     broker_name="shioaji",
            ...     error_traceback=traceback.format_exc()
            ... )
        """
        if not self.is_enabled():
            return False

        # 格式化錯誤訊息
        timestamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")

        message = f"🚨 *{self.title_prefix}股票系統錯誤通知*\n\n"
        message += f"📅 *時間*: `{timestamp}`\n"
        # 動態欄位包反引號的理由同 _send_notice
        message += f"📋 *任務*: `{task_name}`\n"

        if user_name:
            message += f"👤 *用戶*: `{user_name}`\n"
        if broker_name:
            message += f"📊 *券商*: `{broker_name}`\n"

        message += f"❌ *狀態*: 失敗\n\n"
        message += f"⚠️ *錯誤訊息*:\n```\n{error_message}\n```"

        if error_traceback:
            # 只顯示最後 10 行 traceback（避免訊息太長超過 Telegram 4096 字元限制）
            tb_lines = error_traceback.strip().split('\n')
            if len(tb_lines) > 10:
                short_tb = '\n'.join(tb_lines[-10:])
                message += f"\n\n📄 *Traceback* (最後10行):\n```\n{short_tb}\n```"
            else:
                message += f"\n\n📄 *Traceback*:\n```\n{error_traceback}\n```"

        # 發送訊息
        return self.telegram.send_message(message)

    def is_enabled(self) -> bool:
        """檢查通知系統是否啟用且可用"""
        return self.enabled and self.telegram is not None


def create_notification_manager(
    config_dict: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> NotificationManager:
    """
    從 config.yaml 的 notification 區塊建立 NotificationManager

    Args:
        config_dict: config.yaml 中的 notification 設定
        logger: 日誌記錄器（可選）

    Returns:
        NotificationManager 實例

    Example:
        >>> from utils.config_loader import ConfigLoader
        >>> config = ConfigLoader("config.yaml").load_config()
        >>> notifier = create_notification_manager(
        ...     config.get('notification', {}),
        ...     logger
        ... )
    """
    return NotificationManager(config_dict, logger)
