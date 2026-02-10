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
from googleapiclient.http import MediaIoBaseDownload
from utils.config_loader import ConfigLoader
from utils.logger_manager import LoggerManager
from utils.notifier import create_notification_manager

logger = logging.getLogger(__name__)

# Google Drive API 權限範圍
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


class DriveFetcher:
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

    def _download_task(self, task_name, task_config):
        folder_id = task_config.get('drive_folder_id')
        local_dir = task_config.get('local_dir')

        if not folder_id or not local_dir:
            logger.warning(f"Skipping task '{task_name}': Missing drive_folder_id or local_dir in config.")
            return

        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        logger.info(f"Checking Drive folder for task: {task_name} ({folder_id})...")

        download_count = 0
        page_token = None

        while True:
            try:
                query = (
                    f"'{folder_id}' in parents and trashed = false and mimeType != 'application/vnd.google-apps.folder'"
                )

                response = (
                    self.service.files()
                    .list(
                        q=query,
                        fields="nextPageToken, files(id, name)",
                        pageSize=100,
                        pageToken=page_token,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                    .execute()
                )

                items = response.get('files', [])
                page_token = response.get('nextPageToken')

                for item in items:
                    file_id = item['id']
                    file_name = item['name']
                    local_path = os.path.join(local_dir, file_name)

                    if os.path.exists(local_path):
                        continue

                    logger.info(f"Downloading: {file_name}")
                    try:
                        request = self.service.files().get_media(fileId=file_id)
                        fh = io.BytesIO()
                        downloader = MediaIoBaseDownload(fh, request)
                        done = False
                        while done is False:
                            status, done = downloader.next_chunk()

                        with open(local_path, "wb") as f:
                            f.write(fh.getbuffer())
                        download_count += 1
                    except Exception as e:
                        logger.error(f"Download failed {file_name}: {e}")

                if not page_token:
                    break

            except Exception as e:
                logger.error(f"Error listing files for task {task_name}: {e}")
                break

        if download_count > 0:
            logger.info(f"Downloaded {download_count} new files to {local_dir}")
        else:
            logger.info(f"No new files for {task_name}.")

    def run(self):
        if not self.tasks_config:
            logger.warning("No recommendation_tasks found in config.yaml.")
            return

        for task_name, task_config in self.tasks_config.items():
            self._download_task(task_name, task_config)


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    parser = argparse.ArgumentParser(description="Run Google Drive File Fetcher")
    args = parser.parse_args()

    # 初始化通知管理器
    config_loader = ConfigLoader(os.path.join(root_dir, "config.yaml"))
    notifier = create_notification_manager(config_loader.config.get('notification', {}), logger)

    try:
        fetcher = DriveFetcher()
        fetcher.run()
    except Exception as e:
        logger.exception(e)

        # 發送錯誤通知
        notifier.send_error(
            task_name="Google Drive Fetcher", error_message=str(e), error_traceback=traceback.format_exc()
        )
