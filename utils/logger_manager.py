import os
import logging
import re


class LoggerManager:
    def __init__(self, base_log_directory, current_datetime):
        self.base_log_directory = base_log_directory
        self.current_datetime = current_datetime

    def setup_logging(self):
        log_directory = self.base_log_directory
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        log_filename = f'{self.current_datetime.strftime("%Y-%m-%d_%H-%M-%S")}.log'
        log_filepath = os.path.join(log_directory, log_filename)

        logger = logging.getLogger()

        # 清除已有的處理程序（避免重複日誌）
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return log_filepath

    def extract_order_logs(self, log_filepath):
        order_logs = []
        pattern = re.compile(
            r"(?P<action>BUY|SELL)\s+(?P<stock_id>\S+)\s+X\s+(?P<quantity>[\d\.]+)\s+@\s+(?P<limit_price>[\d\.]+)"
            r"(?:\s+with extra bid\s+(?P<extra_bid_pct>[\d\.]+)%){0,1}\s+(?P<order_condition>\S+)"
        )
        with open(log_filepath, "r", encoding="utf-8") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    d = match.groupdict()
                    d["quantity"] = float(d["quantity"])
                    d["limit_price"] = float(d["limit_price"])
                    d["extra_bid_pct"] = float(d["extra_bid_pct"]) / 100 if d["extra_bid_pct"] is not None else 0.0
                    order_logs.append(d)
        return order_logs

    def extract_alerting_stocks(self, log_filepath):
        """
        從 log 檔案中提取警示股資訊

        預期格式:
        買入 8101 0.429 張 - 總價約         2672.67
        賣出 2330 1.500 張 - 總價約        45000.00

        Returns:
            list: 警示股資訊列表，每個元素包含 action, stock_id, quantity, total_amount
        """
        alerting_stocks = []
        pattern = re.compile(
            r"(?P<action>買入|賣出)\s+(?P<stock_id>\d{4,6})\s+(?P<quantity>[\d\.]+)\s+張\s+-\s+總價約\s+(?P<total_amount>[\d\.]+)"
        )
        with open(log_filepath, "r", encoding="utf-8") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    d = match.groupdict()
                    d["quantity"] = float(d["quantity"])
                    d["total_amount"] = float(d["total_amount"])
                    alerting_stocks.append(d)
        return alerting_stocks


if __name__ == "__main__":
    from datetime import datetime
    from zoneinfo import ZoneInfo

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)
    config = LoggerManager(
        base_log_directory=os.path.join(root_dir, "logs"),
        current_datetime=datetime.now(ZoneInfo("Asia/Taipei")),
    )
    log_path = config.setup_logging()
    logging.info(f"Log file created at {log_path}")
