# Stock Analysis

此專案利用 FinLab 提供的即時財經資料，設計股票交易策略並進行歷史回測。透過自動化流程，每日定時執行回測與下單，並將結果以表格與圖表形式展示於 Dashboard 上。Dashboard 內容包含詳細下單資訊、帳戶資金水位變化圖及每月實際回報率圖表。

## 功能
+ 策略開發與回測
    + 利用 FinLab 的即時資料進行股票策略設計與歷史回測。
+ 多用戶與多券商支援
    + 支援多使用者，下單走永豐 Shioaji API，相關參數可於 config.yaml 中調整。
      （玉山富果為 legacy 路徑：finlab 2.x 依賴的 esun_trade 套件未公開發行，目前停用）
+ GoldenAI 推薦清單管線
    + 每週日深夜：`drive_fetcher` 從 Google Drive 抓取推薦報告 → `recommendations_parser` 以 Gemini 解析入庫
    + 每天 07:30：`recommendations_publisher` 將解析結果以 JSON 發布回 Google Drive（供下游系統使用）
+ 批次任務與 Dashboard（排程見 `docker/crontab`，直接呼叫 `python -m jobs.*`）
    + 每天 08:10 - `order_executor` 下單: 推薦清單 → 滾動 tranche 持倉計算 → 下單 → 紀錄 → 資料庫
    + 每天 20:30 - `scheduler` 帳務抓取: 庫存明細、銀行餘額、交割款 → 資料庫
    + 每晚 21:50 起 - `backtest_executor` 各策略回測: 產生報告 → assets/ 與資料庫
    + Web Dashboard: 帳戶頁（:5000，下單紀錄／庫存／資金水位）、GoldenAI 回測儀表板（:8051）
+ Telegram 通知
    + 任務失敗（🚨）、下單摘要（✅）、週日清單解析結果與缺席警告（⚠️）

<img width="1915" height="1077" alt="image" src="https://github.com/user-attachments/assets/9bf3fc18-3ab3-4662-b18f-9f3f04259341" />



## 依賴

+ 資料與交易相關
    + finlab：取得即時財經資料與回測分析（僅能用 pip 安裝）
    + shioaji[speed]：永豐證券下單（僅能用 pip 安裝）
    + fugle-trade：玉山證券下單（僅能用 pip 安裝）
    + keyring：管理與儲存敏感資訊

    + openpyxl：處理 Excel 文件
    + ta-lib：技術分析庫，提供多種股票技術指標
    + lxml：HTML 解析工具（用於 FinLab 中 order_executer.show_alerting_stocks() 依賴 pd.read_html(res.text)）
    + pandas：資料處理與分析

+ Web 與 Dashboard
    + dash：建立交互式 Web 應用
    + dash-bootstrap-components：搭配 Bootstrap 的 UI 組件
    + Flask-AutoIndex：自動生成目錄列表（僅能用 pip 安裝）
    + gunicorn：WSGI HTTP 伺服器（僅適用於 Unix 系統）

+ 其他
    + IPython：互動式 Python 介面
    + pyyaml：讀取配置文件

## 測試

+ `tests/unit/`：純邏輯 + SQLite 測試，不需要 finlab，任何機器可跑；CI 會在每個 PR 與部署前自動執行
    ```bash
    pip install -r requirements-dev.txt
    pytest tests/unit
    ```
+ `tests/integration/`：需要完整環境（finlab 等），用容器現成環境跑（bind mount 蓋掉 image 內的舊 code，不用 rebuild）
    ```bash
    docker compose run --rm -v $(pwd):/app stock-analysis pytest tests
    ```
+ `research/`：策略研究腳本（backfill、grid search、bayesian optimize），手動執行，不是測試

## 手冊

### Docker 部署 (推薦)

> 🚀 最簡單的部署方式,無需安裝 Python 環境,只需 Docker!

#### 前置需求

| 作業系統 | 下載連結 |
|---------|---------|
| **Windows/Mac** | [Docker Desktop](https://www.docker.com/products/docker-desktop/) |
| **Linux** | 執行指令: `curl -fsSL https://get.docker.com \| sh` |

#### 快速開始 (3 步)

**1. 下載專案**
```bash
git clone https://github.com/ntumirlab/stock-analysis.git
cd stock-analysis
```

**2. 準備配置檔**
```bash
# 複製 .env 範本並填入你的 API Keys
cp .env.example .env
nano .env

# 參考 .env.example 了解所有環境變數
# 參考 config.yaml（根目錄）了解配置結構
```

**3. 啟動服務**
```bash
docker compose up -d --build

# 查看狀態
docker compose ps

# 訪問 Dashboard: http://localhost:5000
```

#### 配置架構 (三層系統)

```
.env (敏感值) → config.yaml (${VAR_NAME} 引用) → ConfigLoader (解析)
```

- **Layer 1**: `.env` - 實際敏感值 (在 .gitignore,永不提交)
- **Layer 2**: `config.yaml` - 模板 (${VAR_NAME} 引用,可安全提交)
- **Layer 3**: ConfigLoader - 自動解析變數

詳細文件請見 [DOCKER_SETUP.md](./docs/DOCKER_SETUP.md)
