# GoldenAI Lite 部署手冊

GoldenAI Lite 是自動下單系統：每天從 Google Drive 同步 GoldenAI 週推薦清單，
於每週一買進、週五賣出（永豐金 Shioaji），並提供本機 Dashboard 與 Telegram 通知。

## 機器需求

- 可跑 Docker 的機器（Linux 或 Windows + Docker Desktop），記憶體 8GB 以上
- **排程時段（每日 07:50、08:10、20:30）機器必須開機且未睡眠**，錯過的排程不會補跑
- 系統時鐘正確（Windows + WSL2 睡眠喚醒後時鐘可能漂移，異常時重啟 Docker Desktop 或 `wsl --shutdown`）
- Docker 設定為開機自動啟動，容器 `restart: unless-stopped` 會自動回復

## 安裝步驟

1. **載入 image**（開發方交付的 tar 檔）：

   ```bash
   docker load -i goldenai-lite_v1.0.0.tar
   ```

2. **建立部署目錄**，放入本資料夾提供的檔案：

   ```
   goldenai-lite/
   ├── docker-compose.yml
   ├── config.yaml        # 由 config.yaml.example 複製後填寫
   ├── .env               # 由 .env.example 複製後填寫
   ├── credentials/
   │   ├── Sinopac.pfx        # 永豐憑證（自備）
   │   └── google_token.json  # Drive token（開發方協助取得）
   └── data_prod.db       # 空白檔案，首次啟動自動建表
                          #（Linux: touch data_prod.db；Windows PowerShell: New-Item data_prod.db）
   ```

3. **填寫設定**：
   - `.env`：Shioaji 憑證五項（自備）、Telegram token、開發方提供的 `FINLAB_API_TOKEN` 與 `WEEKLY_FETCH_FOLDER_ID`
   - `config.yaml`：`cycle_start_date` 填第一個買入日（**必須是未來的週一**），`invest_ratio` 依需求調整

4. **啟動**：

   ```bash
   docker compose up -d
   ```

## 驗證安裝

```bash
docker compose ps                          # 兩個容器都應為 running
docker exec goldenai-lite-scheduler date   # 時間應為台北時間
docker exec goldenai-lite-scheduler /opt/conda/envs/stock-analysis/bin/python -m jobs.recommendations_fetcher
```

最後一行手動執行清單同步：首次執行會收到 Telegram ✅ 通知，並可在 Dashboard
（瀏覽器開 <http://localhost:8060>）的「推薦清單」頁看到本週清單。
（清單已是最新時再次執行不會有通知，屬正常。）

## 日常行為

| 時間 | 動作 | 通知 |
|---|---|---|
| 每日 07:50 | 從 Drive 同步推薦清單 | 有新清單發 ✅；檔案異常發 ⚠️ |
| 週一–五 08:10 | 下單（週一買、週五賣） | 委託摘要 ✅；異常 🚨 |
| 每日 20:30 | 抓取持股與帳戶餘額 | 失敗才通知 |

安全設計：當週清單未同步成功時，系統**寧可不下單**（Telegram 會收到 🚨 錯誤通知），
不會拿舊清單交易。

## 更新版本

```bash
docker load -i goldenai-lite_vX.Y.Z.tar
# 修改 .env 的 GOLDENAI_LITE_TAG=vX.Y.Z
docker compose up -d
```

- 請於**非排程時段**執行（避開 07:50、08:10、20:30 前後，建議週末）
- 設定與歷史資料都存在本資料夾，更新不會影響
- 新版異常時，把 `GOLDENAI_LITE_TAG` 改回舊版本再 `docker compose up -d` 即可還原

## 故障排除

- 排程日誌：`./logs/` 目錄（`recommendations_fetcher.log`、`order.log`、`fetch.log`）
- 容器日誌：`docker compose logs -f lite-scheduler`
- 沒收到任何通知：檢查 `.env` 的 Telegram 設定；容器是否在跑（`docker compose ps`）
- 收到 🚨 錯誤通知：先自查常見原因——機器是否在排程時段睡眠、系統時間是否正確、
  `.env` 憑證是否填錯或過期。無法排除時，將通知內容與 `./logs/` 內對應的日誌檔
  一併回報開發方協助排查
