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
        """從 log 抽出 finlab 印出的委託行；換算後不足 1 股的「幽靈單」不列入。

        finlab 合併多策略目標部位時會退化成浮點相加（Position.op 兩邊型別不一致就
        fallback 成 float），與券商回報的 Decimal 部位相減後留下 ~1e-18 的殘差。
        finlab 只丟棄「恰好為 0」的委託，殘差因此會被印成一行 `X 0.0` 的委託 log，
        但它換算張數與股數都是 0、`create_order()` 不會被呼叫，實際沒有送出委託。
        此處照 finlab 的換算方式濾掉，避免幽靈單進 order_history 與下單摘要通知。
        """
        order_logs = []
        skipped = 0
        pattern = re.compile(
            r"(?P<action>BUY|SELL)\s+(?P<stock_id>\S+)\s+X\s+(?P<quantity>[\d\.]+)\s+@\s+(?P<limit_price>[\d\.]+|HIGHEST|LOWEST)"
            r"(?:\s+with extra bid\s+(?P<extra_bid_pct>[\d\.]+)%){0,1}\s+(?P<order_condition>\S+)"
        )
        with open(log_filepath, "r", encoding="utf-8") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    d = match.groupdict()
                    d["quantity"] = float(d["quantity"])
                    if round(d["quantity"] * 1000) == 0:  # 同 finlab 的股數換算
                        skipped += 1
                        continue
                    d["limit_price"] = float(d["limit_price"]) if d["limit_price"] not in ("HIGHEST", "LOWEST") else None
                    d["extra_bid_pct"] = float(d["extra_bid_pct"]) / 100 if d["extra_bid_pct"] is not None else 0.0
                    order_logs.append(d)
        if skipped:
            logging.getLogger(__name__).info(f"略過 {skipped} 筆換算後為 0 股的委託 log（finlab 浮點殘差，未實際送單）")
        return order_logs

    def extract_alerting_stocks(self, log_filepath):
        """
        從 log 檔案中提取警示股資訊

        預期格式:
        買入 8101  0.429 張 - 總價約         2672.67
        賣出 2330  1.500 張 - 總價約        45000.00
        賣出 2492 -0.004 張 - 總價約        -1497.60  (finlab 賣出時數量/金額為負值)

        Returns:
            list: 警示股資訊列表，每個元素包含 action, stock_id, quantity, total_amount
        """
        alerting_stocks = []
        pattern = re.compile(
            r"(?P<action>買入|賣出)\s+(?P<stock_id>\d{4,6})\s+(?P<quantity>-?\d+(?:\.\d+)?)\s+張\s+-\s+總價約\s+(?P<total_amount>-?\d+(?:\.\d+)?)"
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
    config = LoggerManager(base_log_directory=os.path.join(root_dir, "logs"),
                          current_datetime=datetime.now(ZoneInfo("Asia/Taipei")),)
    log_path = config.setup_logging()
    logging.info(f"Log file created at {log_path}")
