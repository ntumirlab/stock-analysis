import os
import json
import re
import time
import argparse
import logging
import traceback
import google.genai as genai
from google.genai import types
from datetime import datetime
from zoneinfo import ZoneInfo
from dao.recommendation_dao import RecommendationDAO, RecommendationRecord, Stock
from utils.config_loader import ConfigLoader
from utils.logger_manager import LoggerManager
from utils.notifier import create_notification_manager

logger = logging.getLogger(__name__)


class RecommendationsParser:
    def __init__(self, task_name, config_path="config.yaml", base_log_directory="logs"):
        self.task_name = task_name
        self.timestamp = datetime.now(ZoneInfo("Asia/Taipei"))

        self.logger_manager = LoggerManager(
            base_log_directory=base_log_directory,
            current_datetime=self.timestamp,
        )
        self.log_file = self.logger_manager.setup_logging()

        self.config_loader = ConfigLoader(config_path)
        self.config_loader.load_global_env_vars()

        self.llm_config = self.config_loader.config.get('llm_settings', {})
        self.model_name = self.llm_config.get('model_name', "gemini-2.0-flash")
        self.api_rate_sleep = self.llm_config.get('api_rate_limit_sleep', 30)
        self.max_retries = self.llm_config.get('max_retries', 3)

        prompt_file_path = self.llm_config.get('prompt_file_path')

        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
            logger.info(f"Loaded prompt from: {prompt_file_path}")
        except Exception as e:
            logger.error(f"Failed to load prompt file from {prompt_file_path}: {e}")
            raise

        task_config = self.config_loader.config.get('recommendation_tasks', {}).get(task_name)
        if not task_config:
            raise ValueError(f"Task '{task_name}' not found in config.yaml under 'recommendation_tasks'")

        self.input_folder = task_config.get('local_dir')

        if not self.input_folder:
            raise ValueError(f"Missing local_dir for task '{task_name}'")

        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise EnvironmentError("Missing environment variable: GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

    def _extract_date(self, filename):
        try:
            match_new = re.match(r"^(\d{8})_\d{6}_", filename)
            if match_new:
                date_str = match_new.group(1)
                return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")

            match_old = re.search(r"recommendation_(\d{8})_", filename)
            if match_old:
                date_str = match_old.group(1)
                return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            return None
        return None

    def _call_gemini(self, content, date_str):
        if not self.prompt_template:
            logger.error("Prompt template is empty!")
            return None

        prompt = self.prompt_template.format(date_str=date_str, content=content)
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json"),
                )

                # Parse JSON response
                parsed_json = json.loads(response.text)

                # Validate stocks field exists
                if 'stocks' not in parsed_json:
                    logger.warning(f"JSON missing 'stocks' field in {date_str}")
                    return None

                # Convert raw dicts to Stock objects
                stocks = [Stock(**stock_dict) for stock_dict in parsed_json.get('stocks', [])]

                # Create RecommendationRecord with Stock objects
                record = RecommendationRecord(date=date_str, stocks=stocks)

                logger.info(f"[{date_str}] Parsed {len(stocks)} stocks: {[s.id for s in stocks]}")
                return record

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    logger.warning(
                        f"Rate Limit (429) on {date_str}. Sleeping {self.api_rate_sleep}s... ({attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(self.api_rate_sleep)
                else:
                    logger.error(f"Gemini Error processing {date_str}: {e}")
                    logger.error(f"Response text: {response.text if 'response' in locals() else 'N/A'}")
                    return None

        logger.error(f"Failed to process {date_str} after {self.max_retries} attempts.")
        return None

    def run(self):
        if not os.path.exists(self.input_folder):
            os.makedirs(self.input_folder)
            logger.info(f"Created input folder: {self.input_folder}")
            return

        # Use DAO to load existing data (filtered by task frequency)
        dao = RecommendationDAO(frequency=self.task_name)
        existing_records = dao.load()
        existing_dates = {record.date for record in existing_records}

        candidates = {}

        files = os.listdir(self.input_folder)
        for f in files:
            if not f.endswith(".md"):
                continue

            f_date = self._extract_date(f)

            if not f_date or f_date in existing_dates:
                continue

            # 比較同日期的檔案，只保留時間較晚的 (較新的)
            if f_date not in candidates:
                candidates[f_date] = f
            else:
                current_best = candidates[f_date]
                if f > current_best:
                    candidates[f_date] = f

        sorted_dates = sorted(candidates.keys())

        if not sorted_dates:
            logger.info("No new dates to process.")
        else:
            logger.info(f"Found {len(sorted_dates)} new dates to process: {sorted_dates}")

            for f_date in sorted_dates:
                f_name = candidates[f_date]
                logger.info(f"Parsing new file for {f_date}: {f_name}")

                file_path = os.path.join(self.input_folder, f_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    record = self._call_gemini(file.read(), f_date)

                    if record:
                        dao.add_record(record)
                        logger.info(f"Saved {len(record.stocks)} stocks for {f_date}")
                        time.sleep(self.api_rate_sleep)  # 避免 Rate Limit

            logger.info(f"Updated {len(sorted_dates)} new records to database")


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    parser = argparse.ArgumentParser(description="Run AI Stock Recommendations Parser")
    parser.add_argument("--task", required=True, help="Task name (e.g., weekly, monthly) as defined in config.yaml")
    args = parser.parse_args()

    # 初始化通知管理器
    config_loader = ConfigLoader(os.path.join(root_dir, "config.yaml"))
    notifier = create_notification_manager(config_loader.config.get('notification', {}), logger)

    try:
        recommendations_parser = RecommendationsParser(task_name=args.task)
        recommendations_parser.run()
    except Exception as e:
        logger.exception(e)

        # 發送錯誤通知
        notifier.send_error(
            task_name=f"Stock Recommendations Parser ({args.task})",
            error_message=str(e),
            error_traceback=traceback.format_exc(),
        )
