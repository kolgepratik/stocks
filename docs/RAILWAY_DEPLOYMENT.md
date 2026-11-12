# Deploying Antigravity Options AI to Railway

This application is ready to deploy on **[Railway](https://railway.com)** using the root `Dockerfile`, `railway.toml`, and a managed **Railway PostgreSQL** database.

---

## 1. Railway Project Setup (5 Minutes)

1. **Create a Project & Add PostgreSQL:**
   * In Railway Dashboard, create a **New Project** → **Deploy from GitHub repo** (or `railway up` from CLI).
   * Inside the same project, click **+ New** → **Database** → **Add PostgreSQL**.
2. **Link `DATABASE_URL` to the App Service:**
   * Open your web service → **Variables** → add `DATABASE_URL` referencing `${{Postgres.DATABASE_URL}}` (Railway auto-creates tables on first boot via `init_db()`).
   * The app automatically normalizes Railway's `postgres://` / `postgresql://` string into `postgresql+psycopg://`.
3. **Set Required Environment Variables:**
   ```ini
   ANTHROPIC_API_KEY="sk-ant-..."
   ANTHROPIC_MODEL="claude-haiku-5-5"
   LLM_PROVIDER="anthropic"
   ENABLE_LLM_FALLBACK="false"

   # Protect all /api/* routes (sent via X-API-Key or Authorization: Bearer)
   API_KEY="replace-with-long-random-secret"
   CRON_SECRET="replace-with-cron-secret"

   # Notifications (optional)
   NOTIFICATION_CHANNEL="TELEGRAM"
   TELEGRAM_BOT_TOKEN=""
   TELEGRAM_CHAT_ID=""
   TELEGRAM_WEBHOOK_SECRET="replace-with-telegram-secret"
   ```
4. **Deploy:**
   * Railway reads `railway.toml`, builds `Dockerfile`, binds dynamically to `$PORT`, and verifies `/health`.
   * Once live, open `https://<your-app>.up.railway.app/docs` for interactive Swagger/OpenAPI docs.

---

## 2. Optional Post-Deploy Steps

### A. Copy Existing Local SQLite Data to Railway Postgres
Copy your public `DATABASE_URL` from Railway Postgres → **Connect**, then run locally:
```bash
DATABASE_URL="postgresql://postgres:..." .venv/bin/python scripts/migrate_sqlite_to_postgres.py
```

### B. Use Telegram Interactive Buttons Without a Second Container
Instead of running a separate `start_bot.py` service, register your Railway endpoint as the Telegram webhook once:
```bash
curl -X POST "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://<your-app>.up.railway.app/api/webhooks/telegram",
    "secret_token": "<TELEGRAM_WEBHOOK_SECRET>"
  }'
```

### C. Schedule Automated Scans (Railway Cron)
Option 1 — **Railway Cron Service** (same repo image, custom command + cron schedule like `30 13 * * 1` UTC = 8:30 AM ET Monday):
```bash
python scripts/run_weekly_scan.py
```
Option 2 — **HTTP Cron Trigger** hitting your API directly:
```bash
curl -X POST "https://<your-app>.up.railway.app/api/cron/weekly-scan" \
  -H "X-Cron-Secret: <CRON_SECRET>"
```

---

## 3. REST API Reference

Pass `X-API-Key: <API_KEY>` (or `Authorization: Bearer <API_KEY>`) on `/api/*` requests when `API_KEY` is set.

| Operation | Method & Path | Purpose |
| :--- | :--- | :--- |
| Health / DB Check | `GET /health` | Liveness & PostgreSQL connectivity probe |
| Start Scan (Async) | `POST /api/scan/start` | Queue DB-backed scan job (`ScanJob`), returns `job_id` |
| Scan Job Status | `GET /api/scan/status/{job_id}` | Poll stage, progress, and recommendations |
| Latest / History Jobs | `GET /api/scan/latest`, `GET /api/scan/jobs` | Fetch recent DB-persisted pipeline runs |
| Run Scan (Sync) | `POST /api/scan` | Execute pipeline synchronously & return results |
| Scheduled Cron Scan | `POST /api/cron/weekly-scan` | Trigger scan with notifications enabled |
| List Recommendations | `GET /api/recommendations` | Grouped by date (`timeframe`, `status` filters) |
| Recommendation Actions | `GET /api/recommendations/{id}`<br>`POST /api/recommendations/{id}/approve`<br>`POST /api/recommendations/{id}/reject`<br>`POST /api/recommendations/{id}/execute` | Inspect, approve, reject, or submit order to `ALPACA` / `ROBINHOOD` / `PAPER` |
| Placed Orders | `GET /api/orders`, `GET /api/orders/{id}`, `POST /api/orders` | List grouped by date, fetch single, or record fill |
| Track / PnL / Close | `PATCH /api/orders/{id}/tracking`<br>`POST /api/orders/{id}/update-pnl`<br>`POST /api/orders/{id}/close` | Toggle `TRACKED`/`UNTRACKED`, update PnL, or close trade |
| Live Position Monitor | `POST /api/orders/monitor` | Refresh live quotes (incl. two-leg spreads) & check alerts |
| Market Research | `GET /api/market/sectors`<br>`GET /api/market/{symbol}/quote`<br>`GET /api/market/{symbol}/options` | Standalone sector dossiers, technicals & options chains |
| Conversational Agent | `POST /api/chat` | Prompt the Claude Haiku 5.5 trading co-pilot |
| Portfolio Stats | `GET /api/stats` | Budget, active orders, unrealized/realized PnL, DB dialect |
| Runtime Settings | `GET /api/settings`, `POST /api/settings` | Read/persist runtime settings in PostgreSQL `app_settings` |
| Notification Test & Bot | `POST /api/notifications/test`<br>`POST /api/webhooks/telegram` | Test Telegram/WhatsApp alerts & handle Telegram callbacks |
