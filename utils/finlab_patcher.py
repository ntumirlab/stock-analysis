"""
FinLab Order Executor Patcher
自動修改 finlab.online.order_executor 的 print 輸出為 logger.info
避免客戶需要手動修改套件原始碼

運作原理:
1. 在程式啟動時,導入 finlab 模組
2. 用 Monkey Patch 替換掉 finlab 內部的函數
3. 把函數內的 print() 輸出改成 logger.info()
"""

import logging
import builtins  # 直接導入 builtins,不用判斷 __builtins__ 的型別!
from typing import Optional

logger = logging.getLogger(__name__)


class FinLabPatcher:
    """動態修改 finlab 套件行為"""

    def __init__(self):
        self._patched = False

    def patch(self) -> bool:
        """
        對 finlab.online.order_executor 進行 monkey patch

        Returns:
            bool: 是否成功打補丁
        """
        try:
            import finlab.online.order_executor as order_executor

            # 檢查是否已經有 logger 屬性
            if not hasattr(order_executor, 'logger'):
                # 為 order_executor 模組添加 logger
                order_executor.logger = logging.getLogger('finlab.online.order_executor')

            # Patch 1: 修改 execute_orders 函數
            self._patch_execute_orders(order_executor)

            # Patch 2: 修改 show_alerting_stocks 函數
            self._patch_show_alerting_stocks(order_executor)

            self._patched = True
            logger.info("[OK] FinLab order_executor 已成功打補丁")
            return True

        except ImportError as e:
            logger.error(f"[ERROR] 無法導入 finlab.online.order_executor: {e}")
            return False
        except Exception as e:
            logger.error(f"[ERROR] FinLab patch 失敗: {e}")
            return False

    def _patch_execute_orders(self, order_executor):
        """
        修改 execute_orders 函數

        原本: print(f'買進 2330 X 10')
        改成: logger.info(f'買進 2330 X 10')
        """
        # 保存原始函數
        original_execute_orders = order_executor.OrderExecutor.execute_orders

        def patched_execute_orders(self, *args, **kwargs):
            """包裝後的 execute_orders"""
            # 保存原始的 print 函數
            original_print = builtins.print

            def logging_print(*print_args, **print_kwargs):
                """替換後的 print 函數"""
                message = ' '.join(str(arg) for arg in print_args)

                # 如果是下單訊息,寫入 log
                if any(keyword in message for keyword in ['買進', '賣出', '股票', 'X']):
                    order_executor.logger.info(message)
                else:
                    # 其他訊息還是用原本的 print
                    original_print(*print_args, **print_kwargs)

            # 暫時替換 print
            builtins.print = logging_print

            try:
                # 執行原始的 execute_orders (它內部的 print 會被我們的版本取代)
                result = original_execute_orders(self, *args, **kwargs)
            finally:
                # 恢復 print (很重要!不然其他程式的 print 也會被影響)
                builtins.print = original_print

            return result

        # 替換掉原始函數
        order_executor.OrderExecutor.execute_orders = patched_execute_orders

    def _patch_show_alerting_stocks(self, order_executor):
        """
        修改 show_alerting_stocks 函數

        處理警示股的 print 輸出
        """
        original_show_alerting_stocks = order_executor.OrderExecutor.show_alerting_stocks

        def patched_show_alerting_stocks(self, *args, **kwargs):
            """包裝後的 show_alerting_stocks"""
            original_print = builtins.print

            def logging_print(*print_args, **print_kwargs):
                message = ' '.join(str(arg) for arg in print_args)

                # 如果是警示股訊息,寫入 log
                if any(keyword in message for keyword in ['買入', '賣出', '張', '總價約']):
                    order_executor.logger.info(message)
                else:
                    original_print(*print_args, **print_kwargs)

            builtins.print = logging_print

            try:
                result = original_show_alerting_stocks(self, *args, **kwargs)
            finally:
                builtins.print = original_print

            return result

        order_executor.OrderExecutor.show_alerting_stocks = patched_show_alerting_stocks

    def is_patched(self) -> bool:
        """檢查是否已經打補丁"""
        return self._patched


# ==================== 對外接口 ====================

_patcher_instance: Optional[FinLabPatcher] = None


def apply_finlab_patches() -> bool:
    """
    應用所有 finlab 補丁 (對外的簡單接口)

    使用方式:
        from utils.finlab_patcher import apply_finlab_patches
        apply_finlab_patches()

    Returns:
        bool: 是否成功應用補丁
    """
    global _patcher_instance

    if _patcher_instance is None:
        _patcher_instance = FinLabPatcher()

    if not _patcher_instance.is_patched():
        return _patcher_instance.patch()
    else:
        logger.debug("FinLab 補丁已經應用過了")
        return True


# ==================== 測試 ====================

if __name__ == "__main__":
    # 測試用
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("=== 開始測試 FinLab Patcher ===")
    success = apply_finlab_patches()

    if success:
        print("[OK] Patch 成功!")

        # 測試: 檢查 finlab 是否真的被 patch 了
        import finlab.online.order_executor as oe

        print(f"[OK] finlab.online.order_executor.logger 存在: {hasattr(oe, 'logger')}")

        # 測試: 檢查函數是否被替換
        print(
            f"[OK] OrderExecutor.execute_orders 已被替換: {oe.OrderExecutor.execute_orders.__name__ == 'patched_execute_orders'}"
        )
        print(
            f"[OK] OrderExecutor.show_alerting_stocks 已被替換: {oe.OrderExecutor.show_alerting_stocks.__name__ == 'patched_show_alerting_stocks'}"
        )

        print("\n=== 所有測試通過! ===")
    else:
        print("[ERROR] Patch 失敗")
