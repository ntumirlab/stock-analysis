"""
通知系統模組 - Telegram 錯誤通知

僅在系統發生錯誤時發送 Telegram 通知，避免訊息過於頻繁。
採用可擴展架構，未來可輕鬆新增其他通知渠道。

Author: Stock Analysis System
Date: 2025-11-01
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from zoneinfo import ZoneInfo


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
            data = {"chat_id": self.chat_id, "text": message, "parse_mode": parse_mode}

            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()

            self.logger.debug("Telegram 訊息發送成功")
            return True

        except Exception as e:
            self.logger.error(f"Telegram 訊息發送失敗: {e}")
            return False


class NotificationManager:
    """
    通知管理器 - 僅支援錯誤通知

    統一管理錯誤通知的發送，目前支援 Telegram。
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

    def send_error(
        self,
        task_name: str,
        error_message: str,
        user_name: Optional[str] = None,
        broker_name: Optional[str] = None,
        error_traceback: Optional[str] = None,
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

        message = "🚨 *股票系統錯誤通知*\n\n"
        message += f"📅 *時間*: `{timestamp}`\n"
        message += f"📋 *任務*: {task_name}\n"

        if user_name:
            message += f"👤 *用戶*: {user_name}\n"
        if broker_name:
            message += f"📊 *券商*: {broker_name}\n"

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
    config_dict: Dict[str, Any], logger: Optional[logging.Logger] = None
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
