import os
import logging
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from core.notification_formats import format_fetch_failures, format_fetch_success
from core.recommendation_fetching import fetch_missing_records
from core.recommendation_publishing import is_folder_id_configured
from dao.recommendation_dao import RecommendationDAO
from utils.config_loader import ConfigLoader
from utils.logger_manager import LoggerManager
from utils.notifier import create_notification_manager

logger = logging.getLogger(__name__)

# Lite 端只讀取發布資料夾。readonly 是 full drive 的子集，token 授權
# 涵蓋 readonly 以上（如開發方 token 的 full drive）都能用。
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


class RecommendationsFetcher:
    def __init__(self, config_path="config.yaml", base_log_directory="logs"):
        self.timestamp = datetime.now(ZoneInfo("Asia/Taipei"))

        self.logger_manager = LoggerManager(
            base_log_directory=base_log_directory,
            current_datetime=self.timestamp,
        )
        self.log_file = self.logger_manager.setup_logging()

        self.config_loader = ConfigLoader(config_path)
        self.config_loader.load_global_env_vars()

        self.tasks_config = self.config_loader.config.get('recommendation_tasks', {})

        token_path = self.config_loader.get_env_var('GOOGLE_TOKEN_PATH')
        if not token_path:
            raise FileNotFoundError(
                "GOOGLE_TOKEN_PATH is not set — add it to .env and point it at the Drive OAuth token file."
            )

        creds = None
        if os.path.exists(token_path):
            try:
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)
            except Exception as e:
                logger.error(f"Error loading token from {token_path}: {e}")

        # 自動更新過期憑證
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    logger.info("Refreshing expired token...")
                    creds.refresh(Request())
                    with open(token_path, 'w') as token:
                        token.write(creds.to_json())
                    logger.info("Token refreshed and saved.")
                except Exception as e:
                    raise RuntimeError(f"Token refresh failed: {e}. Please run get_token.py locally.")
            else:
                raise FileNotFoundError(f"Valid token not found at '{token_path}'.")

        try:
            self.service = build('drive', 'v3', credentials=creds)
        except Exception as e:
            logger.error(f"Failed to initialize Drive Service: {e}")
            raise

    def _list_folder_files(self, folder_id):
        """回傳資料夾內的 {檔名: file_id}。"""
        files = {}
        page_token = None

        while True:
            query = f"'{folder_id}' in parents and trashed = false"
            response = self.service.files().list(
                q=query,
                fields="nextPageToken, files(id, name)",
                pageSize=100,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()

            for item in response.get('files', []):
                files[item['name']] = item['id']
            page_token = response.get('nextPageToken')

            if not page_token:
                break

        return files

    def _download_text(self, file_id):
        content = self.service.files().get_media(
            fileId=file_id,
            supportsAllDrives=True
        ).execute()
        return content.decode('utf-8')

    def _fetch_task(self, task_name, folder_id):
        dao = RecommendationDAO(frequency=task_name)
        remote_files = self._list_folder_files(folder_id)
        return fetch_missing_records(remote_files, self._download_text, dao, task_name)

    def run(self):
        """回傳 {task_name: (本次新入庫的 records, 驗證失敗的日期)}，供呼叫端組通知。"""
        results = {}

        if not self.tasks_config:
            logger.warning("No recommendation_tasks found in config.yaml.")
            return results

        for task_name, task_config in self.tasks_config.items():
            folder_id = task_config.get('fetch_folder_id')
            if not is_folder_id_configured(folder_id):
                logger.info(f"Task '{task_name}': fetch_folder_id not configured, skipping.")
                continue
            results[task_name] = self._fetch_task(task_name, folder_id)

        return results


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    # 初始化通知管理器
    config_loader = ConfigLoader(os.path.join(root_dir, "config.yaml"))
    notifier = create_notification_manager(config_loader.config.get('notification', {}), logger)

    try:
        fetcher = RecommendationsFetcher()
        results = fetcher.run()

        for task_name, (records, failed_dates) in results.items():
            if records:
                notifier.send_success(
                    task_name=f"清單同步 ({task_name})",
                    body=format_fetch_success(task_name, records)
                )
            if failed_dates:
                notifier.send_warning(
                    task_name=f"清單同步 ({task_name})",
                    body=format_fetch_failures(task_name, failed_dates)
                )
    except Exception as e:
        logger.exception(e)

        # 發送錯誤通知（含驗證失敗整份拒收：缺當週清單會被 freshness 檢查擋單）
        notifier.send_error(
            task_name="Stock Recommendations Fetcher",
            error_message=str(e),
            error_traceback=traceback.format_exc()
        )
