# GoldenAI Lite 部署手冊 (客戶版)

GoldenAI Lite 是自動下單系統，安裝完成後自動運作，平常不需人工操作：

- 每天早上從 Google Drive 下載最新的 GoldenAI 推薦清單（週清單與月清單）
- 每週一早上買進清單股票、每週五早上全數賣出（永豐金證券）
- 每天傍晚記錄帳戶持股與餘額
- 各動作完成或異常時發 Telegram 通知
- 提供網頁報表（Dashboard）：下單紀錄、持股、推薦清單

請依本手冊順序安裝，每一步都附有預期結果可供核對。

## 目錄

- [開始之前：需要準備的東西](#開始之前需要準備的東西)
- [基本操作：如何執行指令](#基本操作如何執行指令)
- [步驟 1：安裝 Docker Desktop](#步驟-1安裝-docker-desktop)
- [步驟 2：建立系統資料夾](#步驟-2建立系統資料夾)
- [步驟 3：填寫設定檔](#步驟-3填寫設定檔)
- [步驟 4：啟動系統](#步驟-4啟動系統)
- [步驟 5：確認一切正常](#步驟-5確認一切正常)
- [日常運作](#日常運作)
- [更新版本](#更新版本)
- [疑難排解](#疑難排解)
- [附錄 A：永豐金證券 API 憑證申請](#附錄-a永豐金證券-api-憑證申請)
- [附錄 B：Linux 使用者](#附錄-blinux-使用者)

---

## 開始之前：需要準備的東西

### 電腦需求

- Windows 10 / 11，記憶體 8GB 以上（Linux 指令見文末附錄）
- **每天 07:50、08:10、20:30 電腦必須開機且未睡眠**。系統在這三個時間點
  自動工作，錯過的排程不會補執行——例如週一早上電腦睡眠，當週就不會買進。
  建議使用一台不關機的電腦。
- 系統日期時間正確

### 開發方提供


| 項目                                                        | 說明                                       |
| ----------------------------------------------------------- | ------------------------------------------ |
| `goldenai-lite_v1.0.0.tar`                                  | 系統主程式（版本號可能不同）               |
| `docker-compose.yml`、`config.yaml.example`、`.env.example` | 設定檔與範本                               |
| FinLab API Token                                            | 填入設定檔                                 |
| Google Drive 資料夾 ID                                      | 填入設定檔（週清單必要；月清單選用）       |
| `google_token.json`                                         | Google Drive 憑證檔（開發方協助取得）      |
| Telegram bot token 與群組 chat id                           | 填入設定檔；通知發送到開發方建立的專用群組 |

### 使用者自備


| 項目                | 說明                                                                                |
| ------------------- | ----------------------------------------------------------------------------------- |
| 永豐金證券 API 憑證 | 五項：API Key、Secret Key、身分證字號、憑證檔（.pfx）、憑證密碼。申請步驟見附錄 A。 |
| Telegram 帳號       | 手機安裝 Telegram 註冊後，加入開發方提供的通知群組即可                              |

---

## 基本操作：如何執行指令

安裝過程需要執行指令，方法如下：

1. 用「檔案總管」進入系統資料夾（即步驟 2 建立的 `goldenai-lite` 資料夾）
2. 點檔案總管上方的網址列，輸入 `powershell`，按 Enter，會開啟指令視窗
3. 手冊中灰底框內的文字即指令：整行複製、在視窗內按右鍵貼上、按 Enter 執行

> ⚠️ 請勿使用「命令提示字元（CMD）」，本手冊指令僅適用 PowerShell。
> 依上述方法開啟的即是 PowerShell（提示字元以 `PS` 開頭）。

建議先開啟副檔名顯示，避免認錯檔案：
檔案總管「檢視」→「顯示」→ 勾選「副檔名」。

---

## 步驟 1：安裝 Docker Desktop

Docker 是執行本系統的基礎軟體（免費）。

1. 至 [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/) 下載 Windows 版並安裝，
   全部使用預設選項（可能要求重新開機）
2. 開啟 Docker Desktop，首次詢問登入可按 Skip 跳過
3. 設定開機自動啟動：右上角齒輪（Settings）→ General →
   勾選「Start Docker Desktop when you sign in to your computer」
4. 左下角狀態顯示綠色（Engine running）即安裝完成

同時將電腦設為不自動睡眠：
「設定」→「系統」→「電源與電池」→「螢幕與睡眠」→ 睡眠改為「永不」
（螢幕可關閉，電腦不可睡眠）。

---

## 步驟 2：建立系統資料夾

1. 建立資料夾 `goldenai-lite`（位置自選；本手冊所稱「系統資料夾」即此資料夾），
   放入開發方提供的四個檔案：
   tar 檔、`docker-compose.yml`、`config.yaml.example`、`.env.example`
2. 在其中建立子資料夾 `credentials`，放入兩個憑證檔：

   - 永豐憑證檔（附錄 A 取得），**改名為 `Sinopac.pfx`**
   - `google_token.json`（開發方提供）
3. 在系統資料夾開啟指令視窗，逐行執行：

   ```powershell
   Copy-Item .env.example .env
   Copy-Item config.yaml.example config.yaml
   New-Item data_prod.db -ItemType File
   ```

   （複製兩份設定檔、建立空白資料庫檔）

完成後的資料夾內容：

```
goldenai-lite/
├── goldenai-lite_v1.0.0.tar   ← 系統主程式
├── docker-compose.yml
├── config.yaml.example        ← 範本（保留不動）
├── config.yaml                ← 步驟 3 填寫
├── .env.example               ← 範本（保留不動）
├── .env                       ← 步驟 3 填寫
├── data_prod.db               ← 空白資料庫
└── credentials/
    ├── Sinopac.pfx
    └── google_token.json
```

> ⚠️ **重要**：此資料夾內所有檔案為系統運作必需，之後請勿移動、改名或刪除。
> 下單紀錄與帳戶資料都存放於此，備份此資料夾即備份全部。

---

## 步驟 3：填寫設定檔

需填寫兩個檔案：`.env`（帳號密碼類）與 `config.yaml`（策略參數）。

> ⚠️ **填寫時輸入法請切換為英文**。中文輸入法的全形引號、全形空格
> 會使系統無法讀取設定，是最常見的填錯原因。
>
> ⚠️ `.env` 含帳戶密碼，**請勿提供給任何人**（回報問題時亦不需提供）。

### 3-1. 填寫 .env

執行 `notepad .env` 以記事本開啟，逐項填寫（等號右邊貼上對應的值，前後不留空格）：


| 欄位                      | 內容                                    | 來源                                 |
| ------------------------- | --------------------------------------- | ------------------------------------ |
| `GOLDENAI_LITE_TAG`       | 版本號（如`v1.0.0`），須與 tar 檔名一致 | 已預填（更新版本時才需修改）         |
| `FINLAB_API_TOKEN`        | FinLab token                            | 開發方提供                           |
| `WEEKLY_FETCH_FOLDER_ID`  | 週清單的 Google Drive 資料夾 ID         | 開發方提供                           |
| `MONTHLY_FETCH_FOLDER_ID` | 月清單的 Google Drive 資料夾 ID         | 開發方提供；選填，留空則不同步月清單 |
| `GOOGLE_TOKEN_PATH`       | Google 憑證檔路徑                       | 已預填，不用改                       |
| `TELEGRAM_BOT_TOKEN`      | Telegram 機器人 token                   | 開發方提供                           |
| `TELEGRAM_CHAT_ID`        | 通知群組編號                            | 開發方提供                           |
| `SHIOAJI_API_KEY`         | 永豐 API Key                            | 附錄 A                               |
| `SHIOAJI_SECRET_KEY`      | 永豐 Secret Key                         | 附錄 A                               |
| `SHIOAJI_CERT_PERSON_ID`  | 身分證字號                              | 自行填寫                             |
| `SHIOAJI_CERT_PATH`       | 永豐憑證檔路徑                          | 已預填，不用改                       |
| `SHIOAJI_CERT_PASSWORD`   | 永豐憑證密碼                            | 預設為身分證字號（附錄 A）           |

填寫完成後存檔（Ctrl+S）關閉。

系統通知會發送到開發方建立的 Telegram 專用群組（開發方也在群組內，
異常訊息雙方都看得到）。需要接收通知的人員請開發方邀請加入即可。

### 3-2. 填寫 config.yaml

執行 `notepad config.yaml`。必填欄位只有一個：

- `cycle_start_date`：第一個買入日，**必須是未來的週一**，
  格式如 `"2026-07-20"`（保留引號）

可依需求調整（不改可直接使用）：

- `users:` 底下的使用者名稱（範本為 `username`）：可改為自己偏好的名稱，
  限英文、數字、底線。僅用於系統內部紀錄；**開始交易後請勿再更改**。
  系統僅支援一位使用者，請勿在 `users:` 底下新增其他名稱
- `invest_ratio`：投入資金比例上限，預設 `0.7`（最多以總資產七成買股，其餘留現金）
- `excluded_stocks`：排除的股票代號，如 `["2330"]`

其餘為策略參數，與開發方系統一致，**請勿修改**。

填寫完成後存檔（Ctrl+S）關閉。

> ⚠️ 系統上線後若要調整 `invest_ratio` 或 `cycle_start_date`，
> 請於**無持股時**修改（正常情況為週五賣出後至下週一開盤前）。
> 持股期間修改會使系統立即按新設定調整持股，產生非預期買賣。

---

## 步驟 4：啟動系統

在系統資料夾（`goldenai-lite`）的指令視窗依序執行：

1. 載入系統主程式（首次與每次更新版本時執行；指令中的檔名請對照
   實際收到的 tar 檔名修改，版本號可能不同）：

   ```powershell
   docker load -i goldenai-lite_v1.0.0.tar
   ```

   此步驟需執行數分鐘，**期間畫面不會有任何輸出，屬正常現象**，
   請耐心等候至出現以下結果，切勿關閉視窗。

   **預期輸出**（版本號同檔名）：

   ```
   Loaded image: goldenai-lite:v1.0.0
   ```
2. 啟動：

   ```powershell
   docker compose up -d
   ```

   **預期輸出**（首次啟動會多一行 `Network ... Created`，日後重新啟動則無）：

   ```
   [+] Running 3/3
    ✔ Network goldenai-lite_default      Created
    ✔ Container goldenai-lite-scheduler  Started
    ✔ Container goldenai-lite-dashboard  Started
   ```

之後電腦重開機、Docker Desktop 啟動時，系統會自動恢復運作，無須重新執行。

---

## 步驟 5：確認一切正常

依序執行以下檢查：

1. **兩個服務都在執行**：

   ```powershell
   docker compose ps
   ```

   **預期輸出**（節錄，實際欄位較多；重點為兩行 STATUS 均是 `Up`，
   dashboard 約一分鐘後會多顯示 `healthy`）：

   ```
   NAME                      IMAGE                  STATUS         PORTS
   goldenai-lite-dashboard   goldenai-lite:v1.0.0   Up (healthy)   127.0.0.1:8060->8060/tcp
   goldenai-lite-scheduler   goldenai-lite:v1.0.0   Up
   ```
2. **系統時間正確**：

   ```powershell
   docker exec goldenai-lite-scheduler date
   ```

   **預期輸出**（台北時間的現在時刻，結尾 CST）：

   ```
   Mon Jul 20 09:30:00 CST 2026
   ```
3. **手動同步一次推薦清單**（正式運作時系統每天早上自動執行）：

   ```powershell
   docker exec goldenai-lite-scheduler /opt/conda/envs/stock-analysis/bin/python -m jobs.recommendations_fetcher
   ```

   成功時 Telegram 通知群組會收到 ✅ 開頭、標題含 `[Lite]` 的通知。
   安裝後**首次執行必定會有通知**；未收到即為設定有誤，請檢查 `.env` 的
   Telegram 欄位與資料夾 ID 是否正確。
   （日後清單已是最新時，再次執行不會有通知，屬正常。）
4. **Dashboard 正常**：瀏覽器開啟 [http://localhost:8060](http://localhost:8060)，
   「推薦清單」分頁應顯示最新的週清單與月清單
   （未設定 `MONTHLY_FETCH_FOLDER_ID` 時，月清單區塊顯示「尚無資料」屬正常）。
   （「訂單」「持股」等分頁在開始交易前為空，屬正常。）

四項通過即安裝完成。系統將於 `cycle_start_date` 當天（週一）自動開始交易。

---

## 日常運作


| 時間             | 動作                         | Telegram 通知                  |
| ---------------- | ---------------------------- | ------------------------------ |
| 每天 07:50       | 從 Google Drive 同步推薦清單 | 有新清單發 ✅；檔案異常發 ⚠️ |
| 週一～週五 08:10 | 下單（週一買、週五賣）       | 委託摘要 ✅；異常 🚨           |
| 每天 20:30       | 記錄持股與帳戶餘額           | 失敗才通知                     |

安全設計：當週新清單未同步成功時，系統**寧可不下單**（發 🚨 通知），
不會使用舊清單交易。

日常確認：每週一、五早上 08:10 後應收到下單通知，每週日至週一早上
應收到新清單 ✅ 通知。連續數日無任何通知時，依「疑難排解」檢查。

---

## 更新版本

開發方發布新版時會提供新 tar 檔（如 `goldenai-lite_v1.1.0.tar`）：

1. **於非排程時段操作**（避開 07:50、08:10、20:30 前後，建議週末），
   將新 tar 檔放入系統資料夾（`goldenai-lite`）
2. 在系統資料夾開啟指令視窗（方法見「基本操作」），先停止系統：

   ```powershell
   docker compose down
   ```
3. 載入新版主程式（檔名對照實際收到的 tar 檔；執行需數分鐘，期間無輸出屬正常，請等候至出現 `Loaded image:`）：

   ```powershell
   docker load -i goldenai-lite_v1.1.0.tar
   ```
4. 執行 `notepad .env` 開啟設定檔，找到 `GOLDENAI_LITE_TAG` 開頭的那一行，將等號右邊改為新版本號（須與新 tar 檔名一致），存檔（Ctrl+S）關閉。
   例如新檔案是 `goldenai-lite_v1.1.0.tar`：

   ```
   修改前：GOLDENAI_LITE_TAG=v1.0.0
   修改後：GOLDENAI_LITE_TAG=v1.1.0
   ```
5. 重新啟動：

   ```powershell
   docker compose up -d
   ```

   啟動後依「步驟 5：確認一切正常」的第 1 項，確認兩個服務均為 `Up`。

設定與歷史資料均存於系統資料夾，更新不受影響。

退回舊版：依第 2 步停止系統，將 `GOLDENAI_LITE_TAG` 改回舊版本號，
再執行 `docker compose up -d`。

---

## 疑難排解

回報開發方前請先自查：

**完全沒有收到任何通知（包含 ✅）**

1. Docker Desktop 是否在執行（左下角綠色 Engine running）
2. 服務是否在執行：`docker compose ps` 兩行 STATUS 均應為 `Up`，
   否則執行 `docker compose up -d` 重新啟動
3. `.env` 的 Telegram 兩個欄位是否與開發方提供的值一致（可請開發方核對）

**只有 `goldenai-lite-scheduler` 反覆重啟（dashboard 正常）**

1. 通常為 `config.yaml` 填寫有誤：格式壞掉（常見為全形引號、引號不成對）、
   使用者名稱含英文、數字、底線以外的字元，或設定檔未建立
2. 執行 `docker compose logs lite-scheduler`，最後幾行會顯示錯誤原因；
   依訊息修正 `config.yaml` 存檔後，執行 `docker compose up -d`

**啟動後服務反覆重啟，或日誌出現 `unable to open database file`**

1. 檢查系統資料夾中的 `data_prod.db` 是否變成了**資料夾**
   （Docker 在該檔案不存在時啟動，會自動建立同名資料夾）
2. 是的話：執行 `docker compose down` 停止服務，刪除該資料夾，
   依步驟 2 重新建立空白的 `data_prod.db` 檔案，再執行 `docker compose up -d`

**收到 🚨 錯誤通知**

1. 排程時間電腦是否睡眠（最常見原因）
2. 電腦時間是否正確——睡眠喚醒後偶有時間漂移，
   重啟 Docker Desktop 可解（工作列鯨魚圖示右鍵 → Restart）
3. `.env` 憑證是否填錯或過期（永豐憑證與 API Key 皆有效期限，到期需
   重新下載／申請並更新 `.env`，見附錄 A）

**無法排除時回報開發方**，並附上：

- Telegram 通知完整內容（截圖即可）
- 系統資料夾內 `logs` 子資料夾中對應的日誌檔：清單問題附 `recommendations_fetcher.log`、
  下單問題附 `order.log`、帳戶資料問題附 `fetch.log`
- （進階）容器即時日誌：`docker compose logs -f lite-scheduler`，Ctrl+C 離開

---

## 附錄 A：永豐金證券 API 憑證申請

前置條件：已有永豐金證券帳戶並開通電子交易。以下均在永豐「新理財網」
（[https://www.sinotrade.com.tw](https://www.sinotrade.com.tw)）登入後操作，為一次性設定。
操作上有疑問時，可洽永豐營業員或客服。

> ⚠️ 本附錄通常在安裝前完成，此時 `.env` 尚未建立。
> 過程中取得的五項資料請先抄存於安全處（Secret Key 只顯示一次），
> 於步驟 3 填寫 `.env` 時使用。

1. **簽署 API 條款**：至簽署中心「證券 API 簽署」頁完成簽署：
   [https://www.sinotrade.com.tw/newweb/signCenter/S_openAPI/](https://www.sinotrade.com.tw/newweb/signCenter/S_openAPI/)
2. **申請 API Key**：至「API 管理頁面」
   [https://www.sinotrade.com.tw/newweb/PythonAPIKey/](https://www.sinotrade.com.tw/newweb/PythonAPIKey/)，
   完成雙因子驗證（手機或信箱）後新增 API Key：

   - 權限依頁面勾選（帳務、交易；「正式環境」需通過第 4 步測試審核才可使用）
   - 效期自行設定，請記下到期日，到期前需重新申請並更新 `.env`
   - IP 限制建議不啟用（家用網路 IP 會變動，啟用後會無法登入）
   - 建立完成後顯示 API Key 與 Secret Key（即 `SHIOAJI_API_KEY`、
     `SHIOAJI_SECRET_KEY` 兩欄的值）。
     **Secret Key 只顯示這一次**，請立即複製抄存。
3. **下載憑證**：在同一個 API 管理頁面點「下載憑證」，取得 `.pfx` 檔
   並保存（步驟 2 會放入 `credentials` 資料夾並改名為 `Sinopac.pfx`）。
   憑證密碼預設為身分證字號（若曾自行變更則用變更後的密碼），
   即 `SHIOAJI_CERT_PASSWORD` 欄的值。憑證效期約一年，到期需重新下載。
4. **測試審核**：依主管機關規定，首次使用 API 須先在模擬環境完成
   登入與下單測試並通過審核，正式環境的下單權限才會開通。
   此步驟需以程式執行，請洽開發方協助完成。

## 附錄 B：Linux 使用者

流程相同，僅兩處指令不同：

- 步驟 2 的三行改為：

  ```bash
  cp .env.example .env
  cp config.yaml.example config.yaml
  touch data_prod.db
  ```
- 編輯設定檔以慣用編輯器（如 `nano .env`）取代 `notepad`

機器需求相同：排程時段不可睡眠、時鐘正確、Docker 開機自啟
（`systemctl enable docker`）。
