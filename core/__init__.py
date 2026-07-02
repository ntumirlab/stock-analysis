# core: 純邏輯模組，供 CI 單元測試直接執行。
# 鐵律：此套件內不得 import finlab、utils、jobs（它們的 package __init__
# 會連鎖載入 finlab 等重依賴）。只允許 stdlib 與 dao（純 sqlite）。
