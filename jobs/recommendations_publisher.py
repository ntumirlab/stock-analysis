import os
import io
import argparse
import logging
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from core.recommendation_publishing import (
    build_publish_payload,
    dates_missing_from_drive,
    is_folder_id_configured,
    publish_filename,
)
from dao.recommendation_dao import RecommendationDAO
from utils.config_loader import ConfigLoader
from utils.logger_manager import LoggerManager
from utils.notifier import create_notification_manager

logger = logging.getLogger(__name__)

# 發布需要寫入權限。用完整 drive scope 而非 drive.file，因為 drive.file
# 看不到使用者手動建立的發布資料夾；token 需以 get_token.py 重新授權涵蓋此 scope。
SCOPES = ['https://www.googleapis.com/auth/drive']


class RecommendationsPublisher:
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

        creds = None
        token_path = self.config_loader.get_env_var('GOOGLE_TOKEN_PATH')

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

    def _list_existing_filenames(self, folder_id):
        filenames = []
        page_token = None

        while True:
            query = f"'{folder_id}' in parents and trashed = false"
            response = self.service.files().list(
                q=query,
                fields="nextPageToken, files(name)",
                pageSize=100,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()

            filenames.extend(item['name'] for item in response.get('files', []))
            page_token = response.get('nextPageToken')

            if not page_token:
                break

        return filenames

    def _upload(self, folder_id, filename, payload):
        logger.info(f"Uploading: {filename}")
        media = MediaIoBaseUpload(
            io.BytesIO(payload.encode('utf-8')),
            mimetype='application/json'
        )
        self.service.files().create(
            body={'name': filename, 'parents': [folder_id]},
            media_body=media,
            fields='id',
            supportsAllDrives=True
        ).execute()

    def _publish_task(self, task_name, folder_id):
        dao = RecommendationDAO(frequency=task_name)
        records = dao.load()

        if not records:
            logger.info(f"{task_name}: no recommendations in database, nothing to publish.")
            return

        by_date = {record.date: record for record in records}
        existing = self._list_existing_filenames(folder_id)
        missing = dates_missing_from_drive(by_date.keys(), existing, task_name)

        if not missing:
            logger.info(f"{task_name}: all {len(by_date)} dates already published.")
            return

        logger.info(f"{task_name}: publishing {len(missing)} new dates: {missing}")
        published_at = self.timestamp.isoformat()

        for date in missing:
            payload = build_publish_payload(by_date[date], task_name, published_at)
            self._upload(folder_id, publish_filename(task_name, date), payload)

        logger.info(f"{task_name}: published {len(missing)} files.")

    def run(self):
        if not self.tasks_config:
            logger.warning("No recommendation_tasks found in config.yaml.")
            return

        for task_name, task_config in self.tasks_config.items():
            folder_id = task_config.get('publish_folder_id')
            if not is_folder_id_configured(folder_id):
                logger.info(f"Task '{task_name}': publish_folder_id not configured, skipping.")
                continue
            self._publish_task(task_name, folder_id)


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    parser = argparse.ArgumentParser(description="Publish parsed recommendations as JSON to Google Drive")
    args = parser.parse_args()

    # 初始化通知管理器
    config_loader = ConfigLoader(os.path.join(root_dir, "config.yaml"))
    notifier = create_notification_manager(config_loader.config.get('notification', {}), logger)

    try:
        publisher = RecommendationsPublisher()
        publisher.run()
    except Exception as e:
        logger.exception(e)

        # 發送錯誤通知
        notifier.send_error(
            task_name="Stock Recommendations Publisher",
            error_message=str(e),
            error_traceback=traceback.format_exc()
        )
