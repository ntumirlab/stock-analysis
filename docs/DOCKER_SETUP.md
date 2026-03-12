# Stock Analysis - Docker 部署指南 (客戶版)

> 🚀 這是最簡單的部署方式,無需安裝 Python 環境,只需 Docker!

---

## 📋 目錄
- [前置需求](#前置需求)
- [快速開始](#快速開始)
- [配置說明](#配置說明)
- [排程設定](#排程設定)
- [常用指令](#常用指令)
- [疑難排解](#疑難排解)
- [附錄](#附錄)

---

## ⚙️ 配置架構說明

### 三層配置系統

本系統採用 **三層配置架構**,確保敏感資訊的安全:

```
┌─────────────────────────────────────────────┐
│  第 1 層: .env (敏感資訊 - ⚠️ 不提交到 Git)  │
│  ├─ FINLAB_API_TOKEN=PG323UEltzZ...        │
│  ├─ GOOGLE_API_KEY=AIzaSyDGFlM8...         │
│  └─ SHIOAJI_CERT_PASSWORD=A123456789       │
└────────────────┬────────────────────────────┘
                 │ (env vars 載入到 os.environ)
                 ↓
┌─────────────────────────────────────────────┐
│  第 2 層: config.yaml (配置模板)             │
│  ├─ env:                                    │
│  │  ├─ FINLAB_API_TOKEN: "${FINLAB_API_TOKEN}" │
│  │  └─ ...                                  │
│  └─ users:                                  │
│     └─ junting:                             │
│        └─ shioaji:                          │
│           └─ env:                           │
│              └─ SHIOAJI_CERT_PASSWORD: "${SHIOAJI_CERT_PASSWORD}" │
└────────────────┬────────────────────────────┘
                 │ (解析 ${VAR_NAME} 引用)
                 ↓
┌─────────────────────────────────────────────┐
│  第 3 層: ConfigLoader (變數解析)            │
│  ├─ 載入 .env 到 os.environ               │
│  ├─ 解析 config.yaml 中的 ${VAR_NAME}      │
│  └─ 提供給應用程式實際的配置值              │
└─────────────────────────────────────────────┘
```

**流程說明:**
1. **Docker 啟動** → 掛載 `.env` 檔案
2. **ConfigLoader 初始化** → 讀取 `.env` 到環境變數
3. **應用程式啟動** → ConfigLoader 解析 `config.yaml` 中的 `${VAR_NAME}` 引用
4. **獲得最終配置** → `.env` 優先於 `config.yaml` 中的預設值

**安全優勢:**
- ✅ `.env` 包含實際敏感值,已在 `.gitignore` 中,永不提交
- ✅ `config.yaml` 只有 `${VAR_NAME}` 引用,可安全提交
- ✅ 開發和生產環境共用同一 `config.yaml`,只需調整 `.env`
- ✅ 易於版本控制和團隊協作

---

## 前置需求

### 1. 安裝 Docker Desktop

| 作業系統 | 下載連結 |
|---------|---------|
| **Windows/Mac** | [Docker Desktop](https://www.docker.com/products/docker-desktop/) |
| **Linux** | 執行以下指令: |

```bash
# Linux 安裝 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
# sudo sh get-docker.sh --version 28.1 # for ubuntu 20.04
sudo systemctl start docker
sudo systemctl enable docker
```

### 2. 驗證安裝

```bash
docker --version

docker compose version
```

### 3. 準備憑證檔案

準備永豐金證券的憑證檔案:

| 券商 | 憑證格式 | 取得方式 |
|------|---------|---------|
| **永豐金證券 (Shioaji)** | `.pfx` | [金鑰與憑證申請](https://sinotrade.github.io/zh/tutor/prepare/token/) |

---

## 快速開始 (3 步驟)

### 步驟 1: 下載專案

```bash
# 使用 Git 下載
git clone https://github.com/JunTingLin/stock-analysis.git
cd stock-analysis

```

---

### 步驟 2: 準備配置檔

#### 2.1 設定環境變數 (.env)

**⚠️ 重要:** 所有敏感資訊 (API Keys、憑證密碼等) 應存放在 `.env` 檔案,不應寫在 `config.yaml`!

```bash
# 複製範本
cp .env.example .env

# 編輯 .env 填入你的實際值
nano .env
```

請參考 `.env.example` 檔案了解所有可用的環境變數及其說明。

#### 2.2 編輯 `config/config.yaml` (非敏感設定)

`config.yaml` 只存放非敏感的配置,所有敏感值都以 `${VAR_NAME}` 格式引用自 `.env`。

請參考 `config/config.yaml` 檔案了解配置結構。主要設定包括:
- `env`: 全域環境變數參考
- `users`: 使用者和券商設定
- `llm_settings`: LLM 模型配置
- `notification`: 通知設定
- `recommendation_tasks`: 推薦清單任務設定

**重要:** `.env` 已在 `.gitignore` 中,不會被提交到版本控制。

#### 2.3 放入憑證檔案

在 `config/credentials/` 目錄放入憑證檔案:

```bash
# Linux/Mac
mkdir -p config/credentials
cp /path/to/your_cert.pfx config/credentials/

# Windows (PowerShell)
mkdir -Force config/credentials
copy C:\path\to\your_cert.pfx config\credentials\
```

**目錄結構:**

```
config/credentials/
├── 你的憑證.pfx          # Shioaji 憑證 (必需)
├── google_token.json      # Google Drive API Token (可選)
└── fugle_config.json      # Fugle 設定檔 (可選)
```

**⚠️ 重要提醒:**
1. 憑證檔名須與 `.env` 中的 `SHIOAJI_CERT_PATH` 一致
   - 例如: `.env` 設 `SHIOAJI_CERT_PATH=./config/credentials/junting_Sinopac.pfx` 
   - 則實際檔案應為 `config/credentials/junting_Sinopac.pfx`

2. `.env` 中的路徑使用本地路徑 (`./config/credentials/...`)
   - Docker 容器內自動轉換為 `/app/config/credentials/...`

3. 憑證檔案必須從 [永豐金證券官網](https://sinopac.com.tw) 申請取得

#### 2.4 驗證配置 ✓

```bash
# 驗證 .env 中的環境變數是否存在
grep -E "^(FINLAB_API_TOKEN|SHIOAJI_API_KEY|SHIOAJI_SECRET_KEY)=" .env

# 驗證 .env 中是否有空值
grep "=\s*$" .env
```

---

### 步驟 3: 啟動服務

```bash
# 啟動所有服務 (Dashboard + 排程)
docker compose up -d --build

# 查看啟動狀態
docker compose ps
```

**預期輸出:**
```
NAME                 IMAGE                  STATUS        PORTS
stock-analysis-app   stock-analysis:latest  Up (healthy)  0.0.0.0:5000->5000/tcp
stock-scheduler      stock-analysis:latest  Up
```

- **Dashboard主頁**: http://localhost:5000
- **回測報告瀏覽**: http://localhost:5000/assets/

---

## Optuna 多進程優化（PostgreSQL）

`single_stock_bayesian` 已固定使用 PostgreSQL 作為 Optuna storage（不再提供 DB backend 切換選項）。

### 1. 啟動 PostgreSQL

```bash
docker compose up -d optuna-postgres
docker compose ps optuna-postgres
```

### 2. 以 PostgreSQL storage 執行 Bayesian

```bash
python -m tests.oscar_tw_strategy.single_stock_bayesian_composite \
  --stock_id 2330 \
  --process_workers 8
```

### 3. 套件需求

若使用 PostgreSQL URL，Python 環境需有 PostgreSQL driver（例如 `psycopg2` 或 `psycopg2-binary`）。

```bash
pip install psycopg2-binary
```

---

## 配置說明

### 目錄結構

```
stock-analysis/
├── config/
│   ├── config.yaml          ← 📝 你需要編輯這個
│   ├── credentials/         ← 🔐 憑證資料夾
│   │   ├── your_cert.pfx    ← 永豐金憑證
│   │   └── google_token.json← ☁️ Google Drive 憑證
│   └── prompts/             ← 🧠 股票推薦清單 parser LLM 提示詞腳本
├── logs/                    ← 📊 日誌輸出位置
│   ├── order.log           # 下單日誌
│   ├── fetch.log           # 抓取日誌
│   └── backtest.log        # 回測日誌
├── data_prod.db             ← 💾 資料庫 (自動建立)
├── assets/                  ← 📈 回測報告 HTML
├── finlab_db/               ← 🗄️ FinLab 資料快取 (自動建立)
│   └── workspace/          # 持倉快照 (pm.to_local)
├── docker-compose.yml       ← ⚙️ Docker 配置
└── Dockerfile
```

### 服務說明

| 服務名稱 | 用途 | 端口 |
|---------|------|------|
| `stock-analysis-app` | Dashboard 網頁介面 | 5000 |
| `stock-scheduler` | 定時排程執行器 | - |

### Volume 掛載說明

| 本地路徑 | 容器路徑 | 用途 | 模式 |
|---------|---------|------|------|
| `./.env` | `/app/.env` | 環境變數 (必需) | 只讀 `:ro` |
| `./config/` | `/app/config/` | 配置檔和憑證 (含 token 更新) | 讀寫 |
| `./config.yaml` | `/app/config.yaml` | 主配置檔 (模板) | 只讀 `:ro` |
| `./logs/` | `/app/logs/` | 日誌輸出 | 讀寫 |
| `./data_prod.db` | `/app/data_prod.db` | SQLite 資料庫 | 讀寫 |
| `./assets/` | `/app/assets/` | 回測報告 HTML 與推薦清單輸出 | 讀寫 |
| `./finlab_db/` | `/root/finlab_db/` | FinLab 資料快取 | 讀寫 |
| `./docker/crontab` | `/etc/cron.d/stock-cron` | 排程設定 (僅 scheduler) | 只讀 `:ro` |

**重要說明:**
- **`.env` 必需**: 包含所有敏感資訊 (API Keys、憑證密碼等)
- **`config.yaml` 只是模板**: 現在只包含 `${VAR_NAME}` 參考,實際值從 `.env` 讀取
- **ConfigLoader 自動解析**: 啟動時會自動讀取 `.env` 並解析 YAML 中的 `${VAR_NAME}` 模式
- **`config/` 現為讀寫**: 允許 Google token 自動更新 (原為只讀)
- **推薦清單輸出**: 所有推薦清單相關資料現存放於 `assets/` 目錄
- `finlab_db/` 目錄用於存放 FinLab 的持倉快照和資料快取,會自動建立

---

## 排程設定

預設排程內容 (`docker/crontab`):

```bash
# 1. 每天 20:30 - 抓取當日持股和帳戶資訊
30 20 * * * cd /app && python -m jobs.scheduler --user_name=junting --broker_name=shioaji

# 2. 每天 20:00 - 執行回測
0 20 * * * cd /app && python -m jobs.backtest_executor --strategy_class_name=AlanTWStrategyACE

# 3. 每天 08:00 - 早盤下單
0 8 * * * cd /app && python -m jobs.order_executor --user_name=junting --broker_name=shioaji

# 4. 每天 13:00 - 尾盤下單 (加價 1%)
0 13 * * * cd /app && python -m jobs.order_executor --user_name=junting --broker_name=shioaji --extra_bid_pct=0.01
```

**排程中的參數值來源:**
- `--user_name`, `--broker_name`: 來自 `config.yaml` 的 `users` 節點
- `--strategy_class_name`: 來自 `config.yaml` 中使用者的 `constant.strategy_class_name`
- 所有敏感資訊 (API Key、憑證密碼): 從 `.env` 自動讀取，無需在 crontab 中指定

### 排程參數說明

#### `jobs.scheduler` - 抓取帳務資料

| 參數 | 必需 | 預設值 | 說明 |
|------|------|--------|------|
| `--user_name` | ✅ | 無 | 使用者名稱 (需與 `config.yaml` 一致) |
| `--broker_name` | ✅ | 無 | 券商名稱 (`shioaji`) |

**範例:**
```bash
python -m jobs.scheduler --user_name=alan --broker_name=shioaji
```

#### `jobs.backtest_executor` - 執行回測

| 參數 | 必需 | 預設值 | 說明 |
|------|------|--------|------|
| `--strategy_class_name` | ✅ | 無 | 策略類別名稱 (見 [附錄 A](#附錄-a-可用的策略類別)) |

**範例:**
```bash
python -m jobs.backtest_executor --strategy_class_name=PrisonRabbitStrategy
```

#### `jobs.order_executor` - 執行下單

| 參數 | 必需 | 預設值 | 說明 |
|------|------|--------|------|
| `--user_name` | ✅ | 無 | 使用者名稱 (需與 `config.yaml` 一致) |
| `--broker_name` | ✅ | 無 | 券商名稱 (`shioaji`) |
| `--extra_bid_pct` | ❌ | `0` | 額外加價百分比 (例如 `0.01` = 加價 1%) |
| `--view_only` | ❌ | `false` | 僅查看模式,不實際下單 |

**範例:**
```bash
# 一般下單
python -m jobs.order_executor --user_name=junting --broker_name=shioaji

# 加價 1% 下單 (尾盤)
python -m jobs.order_executor --user_name=junting --broker_name=shioaji --extra_bid_pct=0.01

# 只看不下單 (測試模式)
python -m jobs.order_executor --user_name=junting --broker_name=shioaji --view_only
```

---

## 常用指令

### 服務管理

```bash
# 啟動服務
docker compose up -d --build

# 停止服務
docker compose down

# 重新啟動服務
docker compose restart

# 查看服務狀態
docker compose ps

# 查看資源使用
docker stats stock-analysis-app stock-scheduler
```

### 日誌查看

```bash
# 查看所有日誌 (即時)
docker compose logs -f

# 只看 Dashboard 日誌
docker compose logs -f stock-analysis

# 只看排程日誌
docker compose logs -f stock-scheduler

# 查看最近 100 行
docker compose logs --tail=100

# 查看本地日誌檔案
tail -f logs/order.log
tail -f logs/fetch.log
tail -f logs/backtest.log
```

### 手動執行指令

```bash
# 手動執行下單 (測試模式)
docker exec -it stock-scheduler python -m jobs.order_executor \
  --user_name=junting \
  --broker_name=shioaji \
  --view_only

# 手動執行回測
docker exec -it stock-scheduler python -m jobs.backtest_executor \
  --strategy_class_name=AlanTWStrategyACE

# 手動抓取帳務資料
docker exec -it stock-scheduler python -m jobs.scheduler \
  --user_name=junting \
  --broker_name=shioaji
```

### 更新程式

```bash
# 1. 拉取最新程式碼
git pull

# 2. 重新建立並啟動
docker compose up -d --build

# 3. 確認更新成功
docker compose ps
docker compose logs --tail=50
```

### 清理資源

```bash
# 停止並移除容器
docker compose down

# 移除容器 + 未使用的映像
docker compose down --rmi local

# 清理所有未使用的 Docker 資源 (謹慎使用!)
docker system prune -a
```

---


## 附錄

### 附錄 A: 可用的策略類別

在 `config.yaml` 的 `strategy_class_name` 欄位可使用以下策略:

| 策略類別名稱 | 檔案位置 | 說明 | 來源 |
|-------------|---------|------|------|
| `AlanTWStrategyACE` | [alan_tw_strategy_ACE.py](strategy_class/alan_tw_strategy_ACE.py) | Alan 策略 (A\|C\|E) | 自訂 |
| `PeterWuStrategy` | [peterwu_tw_strategy.py](strategy_class/peterwu_tw_strategy.py) | Peter Wu 策略 | 自訂 |
| `RAndDManagementStrategy` | [r_and_d_management_strategy.py](strategy_class/r_and_d_management_strategy.py) | 研發管理大亂鬥 | FinLab 官方 |
| `RevenuePriceStrategy` | [tibetanmastiff_tw_strategy.py](strategy_class/tibetanmastiff_tw_strategy.py) | 藏敖策略 | FinLab 官方 |

**範例配置:**

```yaml
users:
  junting:
    shioaji:
      constant:
        strategy_class_name: "AlanTWStrategyACE"  # 使用 Alan 策略 ACE 組合
```

---