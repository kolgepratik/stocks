# Antigravity Options AI — Advisory, Execution & Order Tracking Engine

An institutional-grade, multi-agent options trading system built on **LangGraph**, **Claude Haiku 5.5**, and **FastAPI**. Screens the market across 3 thematic macro sectors, structures strictly defined-risk call strategies, enforces zero-margin guardrails, and provides an interactive **Web UX & Chat Assistant** to track placed orders and monitor live PnL over time.

---

## ⚡ Core Capabilities

- **Zero-Margin & Zero-Share Policy:** Exclusively generates pure buyer positions (`LONG_CALL` and `CALL_DEBIT_SPREAD`). All single-leg puts and margin-dependent strategies are permanently banned.
- **Independent Per-Trade Budget Cap:** Each trade is sized independently under your budget limit (default \$500.00). Qualifiers are ranked by conviction and expected profit likelihood so you can choose your preferred setup.
- **Placed Order Tracking & Live Monitoring:** Tell the agent which orders you executed in your broker. The app records them in SQLite, tracks performance over time, toggles `TRACKED` vs `UNTRACKED` states, updates PnL, and monitors against profit targets, stop losses, and time-stop rules.
- **Interactive Web UX & API:** Modern dashboard built with FastAPI:
  - 💬 **Agent Prompting & Chat:** Conversational co-pilot powered by **Claude Haiku 5.5**.
  - 📋 **Past Recommendations:** Filterable by timeframe (`7d`, `14d`, `1m`, `1q`, `1y`, `all`) and grouped by dates, with a 1-click *"I Placed This Order"* logger.
  - 💼 **Placed Orders & PnL:** Filterable by timeframe and tracking state (`TRACKED` / `UNTRACKED`), grouped by dates with live PnL updates.
  - ⚙️ **Settings Management:** Live runtime controls for budget, models, and notifications.
- **Dual Mobile Alerts:** Instant trade notifications via **WhatsApp** (CallMeBot) and interactive inline approval buttons via **Telegram**.

---

## 🚀 Quick Start (Local)

### 1. Install Dependencies
```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

cd trading_agents
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Configure Environment (`.env`)
```bash
cp .env.example .env
```
Key settings:
```ini
ANTHROPIC_API_KEY="sk-ant-..."
ANTHROPIC_MODEL="claude-haiku-5-5"
LLM_PROVIDER="anthropic"
ENABLE_LLM_FALLBACK=false

WEEKLY_BUDGET_USD=500.0
MAX_COST_PER_TRADE_USD=500.0
NOTIFICATION_CHANNEL="WHATSAPP"
```

### 3. Launch Web Dashboard
```bash
python scripts/start_web.py --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

### 4. Run CLI Weekly Scan
```bash
# Scan default thematic universe ($500 budget)
python scripts/run_weekly_scan.py

# Custom symbols & budget override
python scripts/run_weekly_scan.py --symbols NVDA OKLO XOM --budget 500
```

---

## 🐳 Docker Containerization

Run the entire application stack in Docker:

```bash
# Build and start the Web Dashboard service (port 8000)
docker compose up --build web

# Run a one-off scan inside Docker
docker compose run --rm weekly-scan
```

---

## 📚 Comprehensive Documentation

For complete architectural diagrams, database schema details, local setup, and Railway cloud deployment:
- 👉 **[Architecture, Installation & Configuration Guide](./docs/ARCHITECTURE_AND_SETUP.md)**
- 🚂 **[Railway Deployment & REST API Guide](./docs/RAILWAY_DEPLOYMENT.md)**
