# Lite image 專用的 strategy_class/__init__.py（由 lite/Dockerfile 覆蓋原檔）。
# 主 repo 的 __init__ 會 eager import 所有策略；Lite 只含 GoldenAI 的
# adapter 與 base，order path 均以 `from strategy_class.<module> import ...`
# 直接路徑載入，套件層不需要 re-export。
