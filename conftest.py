import os
import sys

# 讓測試不論從哪裡執行都能 import 專案模組（core、dao、utils、jobs...）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
