# GoldenAI Lite 交付用 image（client 自跑下單，不含 repo 與其他策略 IP）
# 由 repo 根目錄建置: docker build -f lite/Dockerfile -t goldenai-lite:vX.Y.Z .
FROM continuumio/miniconda3:latest AS builder

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml && \
    conda clean -afy

# 最終 runtime image
FROM continuumio/miniconda3:latest

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    curl \
    cron \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/conda/envs/stock-analysis /opt/conda/envs/stock-analysis

# 程式碼採白名單逐路徑 COPY（而非 COPY . + 排除法）：
# 沒列出的檔案（Alan/PeterWu/Oscar 等策略）物理上不存在於 image
COPY jobs/__init__.py jobs/scheduler.py jobs/inventory_fetcher.py jobs/balance_fetcher.py \
     jobs/order_executor.py jobs/recommendations_fetcher.py jobs/profit_loss_fetcher.py \
     jobs/shioaji_activation_test.py /app/jobs/
COPY core/__init__.py core/trading_cycles.py core/notification_formats.py \
     core/recommendation_fetching.py core/recommendation_publishing.py core/price_frames.py \
     core/profit_loss_range.py /app/core/
COPY dao/__init__.py dao/order_dao.py dao/account_dao.py dao/inventory_dao.py \
     dao/balance_dao.py dao/recommendation_dao.py dao/golden_ai_backtest_metrics_dao.py \
     dao/profit_loss_dao.py /app/dao/
COPY utils/ /app/utils/
COPY service/ /app/service/
COPY tabs/ /app/tabs/
# markets 為 namespace package（無 __init__.py）
COPY markets/target_weekday_tw_market.py /app/markets/
COPY strategy_class/golden_ai_order_adapter.py strategy_class/golden_ai_tw_strategy_base.py /app/strategy_class/
# 主 repo 的 strategy_class/__init__.py 會 eager import 所有策略，
# Lite 只放 GoldenAI 兩檔，改用極簡版 __init__（order path 均為直接路徑 import）
COPY lite/strategy_class_init.py /app/strategy_class/__init__.py
COPY lite_dashboard.py /app/

# IP 守門：strategy_class/ 出現白名單以外的檔案就讓 build 失敗
RUN test "$(ls /app/strategy_class | sort)" = "$(printf '__init__.py\ngolden_ai_order_adapter.py\ngolden_ai_tw_strategy_base.py')"

# IP 守門：開發方模組（回測、報告解析/清單發布、主 dashboard）不得混入 image
RUN for f in jobs/backtest_executor.py jobs/recommendations_parser.py \
        jobs/recommendations_publisher.py jobs/drive_fetcher.py \
        core/recommendation_parsing.py core/report_rewrap.py dashboard.py; do \
        test ! -e "/app/$f" || exit 1; \
    done

# 白名單完整性檢查：把排程與 dashboard 會執行的模組全部 import 一次，
# 白名單 COPY 漏了任何被依賴的檔案，在 build 期即失敗（而非 client 機的 cron 才爆）。
# 下單路徑的 adapter 是 lazy import（order_executor.load_strategies 內），須明列才會被檢查。
# import 的副作用檔（lite_dashboard 建空 DB、shioaji 建空 log）同一層 rm 掉不留進 image
RUN /opt/conda/envs/stock-analysis/bin/python -c "import jobs.order_executor, jobs.scheduler, \
        jobs.recommendations_fetcher, jobs.inventory_fetcher, jobs.balance_fetcher, \
        jobs.shioaji_activation_test, \
        strategy_class.golden_ai_order_adapter, lite_dashboard" && \
    rm -f /app/data_prod.db /app/shioaji.log

# crontab 烘進 image（排程不開放 client 端修改；改排程 = 發新版）
COPY lite/crontab /etc/cron.d/lite-cron
RUN dos2unix /etc/cron.d/lite-cron

# 註: config.yaml, .env, credentials/, data_prod.db, logs/, finlab_db
#     由 docker-compose 的 volumes 掛載（見 lite/docker-compose.yml）

ENV PATH=/opt/conda/envs/stock-analysis/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Taipei

SHELL ["/bin/bash", "-c"]
RUN echo "source activate stock-analysis" > ~/.bashrc

# lite dashboard 端口
EXPOSE 8060

# 預設啟動排程（dashboard service 在 compose 以 command 覆寫）。
# 啟動時讀 config.yaml 的 users key（名稱由 client 自訂）代入 crontab 的 {{USER_NAME}}；
# config 缺漏、使用者不是恰好一位、或名稱不合法時啟動即失敗（compose ps 立即可見），
# 不留到排程時段才在 log 裡爆
CMD ["bash", "-c", "user_name=$(python -c \"import re, yaml; us = list(yaml.safe_load(open('/app/config.yaml'))['users']); assert len(us) == 1, f'config.yaml 的 users 底下必須恰好一位使用者: {us}'; n = str(us[0]); assert re.fullmatch(r'[A-Za-z0-9_]+', n), f'使用者名稱僅限英文、數字、底線: {n}'; print(n)\") && sed \"s/{{USER_NAME}}/$user_name/g\" /etc/cron.d/lite-cron | crontab - && cron -f"]
