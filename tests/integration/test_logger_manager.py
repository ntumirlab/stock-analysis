import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger_manager import LoggerManager
from datetime import datetime
from zoneinfo import ZoneInfo


def _make_manager():
    return LoggerManager(
        base_log_directory="/tmp/test_logs",
        current_datetime=datetime.now(ZoneInfo("Asia/Taipei")),
    )


def _write_log(content: str) -> str:
    """Write content to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".log", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


class TestExtractAlertingStocks(unittest.TestCase):

    def setUp(self):
        self.mgr = _make_manager()

    def _extract(self, log_text):
        path = _write_log(log_text)
        try:
            return self.mgr.extract_alerting_stocks(path)
        finally:
            os.unlink(path)

    # ------------------------------------------------------------------ buy --
    def test_buy_positive_values_matched(self):
        log = "2026-06-01 10:00:00 - 買入 2330  0.429 張 - 總價約         2672.67\n"
        result = self._extract(log)
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row["action"], "買入")
        self.assertEqual(row["stock_id"], "2330")
        self.assertAlmostEqual(row["quantity"], 0.429)
        self.assertAlmostEqual(row["total_amount"], 2672.67)

    # ----------------------------------------------------------------- sell --
    def test_sell_negative_values_matched(self):
        """賣出時 finlab 印出負值，regex 必須能 match。"""
        log = "2026-06-01 16:02:39 - 賣出 2492 -0.004 張 - 總價約        -1497.60\n"
        result = self._extract(log)
        self.assertEqual(len(result), 1, "賣出的負值數量/金額應被 regex 正確解析")
        row = result[0]
        self.assertEqual(row["action"], "賣出")
        self.assertEqual(row["stock_id"], "2492")
        self.assertAlmostEqual(row["quantity"], -0.004)
        self.assertAlmostEqual(row["total_amount"], -1497.60)

    def test_sell_negative_quantity_abs_is_nonzero(self):
        """下游 reservation_handler 使用 abs(quantity)，確認負值轉換後非零。"""
        log = "2026-06-01 16:02:39 - 賣出 2492 -0.004 張 - 總價約        -1497.60\n"
        result = self._extract(log)
        self.assertGreater(abs(result[0]["quantity"]) * 1000, 0)

    # ----------------------------------------------------------- mixed lines --
    def test_mixed_buy_and_sell_in_same_log(self):
        log = (
            "2026-06-01 10:00:00 - 買入 8101  1.500 張 - 總價約        45000.00\n"
            "2026-06-01 16:02:39 - 賣出 2492 -0.004 張 - 總價約        -1497.60\n"
            "2026-06-01 16:02:39 - 賣出 3035 -2.000 張 - 總價約       -60000.00\n"
        )
        result = self._extract(log)
        self.assertEqual(len(result), 3)
        actions = [r["action"] for r in result]
        self.assertEqual(actions, ["買入", "賣出", "賣出"])

    def test_irrelevant_lines_not_matched(self):
        log = (
            "2026-06-01 16:02:39 - 無警示股，跳過圈存流程\n"
            "2026-06-01 16:02:39 - INFO some other log line\n"
        )
        result = self._extract(log)
        self.assertEqual(len(result), 0)

    def test_empty_log_returns_empty_list(self):
        result = self._extract("")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
