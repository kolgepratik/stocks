"""
Tests for Railway Cloud Readiness, PostgreSQL URL Handling, DB-Backed Settings/Jobs,
API Authentication, and Full Operation REST Endpoints.
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from config.settings import get_settings, normalize_database_url
from src.database.connection import get_db, reset_engine
from src.database.models import (
    OptionType,
    ScanJob,
    TradeRecommendation,
    TradeStatus,
)
from src.tools.market_data import CandidateTicker, OptionChainResult, OptionContract, TechnicalSnapshot
from src.agents.sector_subagents import SectorResearchDossier
from src.web.app import app


@pytest.fixture
def isolated_client(tmp_path, monkeypatch):
    """Run FastAPI TestClient against an isolated temp directory & SQLite database."""
    monkeypatch.chdir(tmp_path)
    db_file = tmp_path / "test_railway.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("CRON_SECRET", "")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    get_settings.cache_clear()
    reset_engine()

    with TestClient(app) as client:
        yield client

    get_settings.cache_clear()
    reset_engine()


def test_normalize_database_url():
    assert normalize_database_url("") == "sqlite:///./data/trading.db"
    assert normalize_database_url(None) == "sqlite:///./data/trading.db"
    assert (
        normalize_database_url("postgres://user:pass@host:5432/railway")
        == "postgresql+psycopg://user:pass@host:5432/railway"
    )
    assert (
        normalize_database_url("postgresql://user:pass@host:5432/railway")
        == "postgresql+psycopg://user:pass@host:5432/railway"
    )
    assert (
        normalize_database_url("postgresql+psycopg://user:pass@host:5432/railway")
        == "postgresql+psycopg://user:pass@host:5432/railway"
    )
    assert normalize_database_url("sqlite:///./data/trading.db") == "sqlite:///./data/trading.db"


def test_health_and_root(isolated_client: TestClient):
    r_health = isolated_client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json() == {
        "status": "ok",
        "database": "connected",
        "dialect": "sqlite",
    }

    r_root = isolated_client.get("/")
    assert r_root.status_code == 200
    assert "Antigravity Options AI" in r_root.text
    assert "apiFetch" in r_root.text
    assert "modal-api-auth" in r_root.text
    assert "modal-close-order" in r_root.text
    assert "modal-execute-rec" in r_root.text
    assert "view-research" in r_root.text
    assert "/api/market/" in r_root.text
    assert "/api/scan/jobs" in r_root.text
    assert "/api/notifications/test" in r_root.text
    assert "window.prompt" not in r_root.text
    assert "API_AUTH_REQUIRED = false;" in r_root.text


def test_api_key_and_cron_auth(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_file = tmp_path / "auth.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("API_KEY", "super-secret-api-key")
    monkeypatch.setenv("CRON_SECRET", "super-secret-cron-key")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "tg-secret")
    get_settings.cache_clear()
    reset_engine()

    with TestClient(app) as client:
        # Public routes stay open
        assert client.get("/health").status_code == 200
        r_auth_root = client.get("/")
        assert r_auth_root.status_code == 200
        assert "API_AUTH_REQUIRED = true;" in r_auth_root.text

        # Protected API routes reject missing/wrong key
        assert client.get("/api/stats").status_code == 401
        assert client.get("/api/stats", headers={"X-API-Key": "wrong"}).status_code == 401

        # Valid X-API-Key and Bearer succeed
        assert client.get("/api/stats", headers={"X-API-Key": "super-secret-api-key"}).status_code == 200
        assert (
            client.get(
                "/api/stats",
                headers={"Authorization": "Bearer super-secret-api-key"},
            ).status_code
            == 200
        )

        # Cron endpoint requires CRON_SECRET
        assert client.post("/api/cron/weekly-scan").status_code == 401
        with patch("src.web.app.run_pipeline") as mock_pipe:
            mock_pipe.return_value = {
                "final_recommendations": [],
                "macro_political_context": {},
                "saved_recommendation_ids": [],
            }
            res = client.post(
                "/api/cron/weekly-scan?async_mode=false",
                headers={"X-Cron-Secret": "super-secret-cron-key"},
            )
            assert res.status_code == 200
            assert mock_pipe.call_args.kwargs["notify"] is True

        # Telegram webhook validates secret
        assert client.post("/api/webhooks/telegram", json={"update_id": 1}).status_code == 401

    get_settings.cache_clear()
    reset_engine()


def test_db_backed_settings_persist_across_cache_clear(isolated_client: TestClient):
    res = isolated_client.post(
        "/api/settings",
        json={
            "weekly_budget_usd": 850.0,
            "max_cost_per_trade_usd": 425.0,
            "notification_channel": "TELEGRAM",
            "scan_universe": "NVDA,OKLO,XOM",
        },
    )
    assert res.status_code == 200

    # Simulate fresh container boot / cache invalidation with NO .env file on disk
    get_settings.cache_clear()
    s = isolated_client.get("/api/settings").json()
    assert s["weekly_budget_usd"] == 850.0
    assert s["max_cost_per_trade_usd"] == 425.0
    assert s["notification_channel"] == "TELEGRAM"
    assert s["scan_universe"] == "NVDA,OKLO,XOM"


def test_db_backed_scan_jobs_and_restart_recovery(isolated_client: TestClient):
    fake_rec = {
        "symbol": "NVDA",
        "option_type": "CALL",
        "structure": "LONG_CALL",
        "strike": 140.0,
        "expiration": (date.today() + timedelta(days=7)).isoformat(),
        "entry_price": 2.50,
        "target_exit_price": 3.75,
        "stop_loss_price": 1.88,
    }
    with patch("src.web.app.run_pipeline") as mock_pipe:
        mock_pipe.return_value = {
            "final_recommendations": [fake_rec],
            "macro_political_context": {"us_political_macro_summary": "Supportive AI spend"},
            "saved_recommendation_ids": [1],
        }
        start_res = isolated_client.post(
            "/api/scan/start",
            json={"symbols": ["nvda"], "budget": 500.0, "notify": False},
        )
        assert start_res.status_code == 200
        job_id = start_res.json()["job_id"]

        # Poll DB-backed job until completed
        status_payload = {}
        for _ in range(30):
            r = isolated_client.get(f"/api/scan/status/{job_id}")
            assert r.status_code == 200
            status_payload = r.json()
            if status_payload["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert status_payload["status"] == "completed"
        assert status_payload["recommendations_count"] == 1
        assert status_payload["symbols"] == ["NVDA"]

    # Latest & list endpoints read directly from DB
    latest = isolated_client.get("/api/scan/latest").json()
    assert latest["job_id"] == job_id
    jobs = isolated_client.get("/api/scan/jobs").json()
    assert jobs["count"] == 1

    # Simulate an interrupted job surviving in DB -> startup marks it failed
    with get_db() as session:
        session.add(
            ScanJob(
                job_id="orphaned-job-1",
                status="running",
                stage="2/4 Structuring...",
                notify=False,
                started_at=time.time() - 60,
            )
        )

    from src.web.app import _mark_orphaned_jobs_failed

    _mark_orphaned_jobs_failed()
    recovered = isolated_client.get("/api/scan/status/orphaned-job-1").json()
    assert recovered["status"] == "failed"
    assert "restart" in recovered["error"].lower()


def test_recommendation_lifecycle_and_execution_apis(isolated_client: TestClient):
    exp = date.today() + timedelta(days=7)
    with get_db() as session:
        rec = TradeRecommendation(
            symbol="OKLO",
            underlying_price=24.0,
            option_type=OptionType.CALL,
            structure="CALL_DEBIT_SPREAD",
            strike_price=24.0,
            short_strike_price=26.0,
            expiration_date=exp,
            contract_symbol="OKLO_SPREAD_24_26",
            entry_price_estimate=1.20,
            target_exit_price=1.80,
            stop_loss_price=0.90,
            max_contracts=2,
            max_cost=240.0,
            thesis="Data center power momentum",
            status=TradeStatus.PENDING_APPROVAL,
        )
        session.add(rec)
        session.flush()
        rec_id = rec.id

    # Verify Enum serialized cleanly (not 'OptionType.CALL' / 'TradeStatus.PENDING_APPROVAL')
    r_get = isolated_client.get(f"/api/recommendations/{rec_id}")
    assert r_get.status_code == 200
    assert r_get.json()["recommendation"]["option_type"] == "CALL"
    assert r_get.json()["recommendation"]["status"] == "PENDING_APPROVAL"

    # Approve -> Reject -> Execute via PAPER broker
    r_app = isolated_client.post(f"/api/recommendations/{rec_id}/approve")
    assert r_app.status_code == 200
    assert r_app.json()["recommendation"]["status"] == "APPROVED"

    r_exec = isolated_client.post(
        f"/api/recommendations/{rec_id}/execute",
        json={"broker": "PAPER", "record_placed_order": True},
    )
    assert r_exec.status_code == 200
    exec_data = r_exec.json()
    assert exec_data["recommendation"]["status"] == "EXECUTED"
    assert exec_data["placed_order"]["symbol"] == "OKLO"
    assert exec_data["placed_order"]["tracking_status"] == "TRACKED"

    order_id = exec_data["placed_order"]["id"]

    # Fetch single order, update tracking, update PnL, close
    assert isolated_client.get(f"/api/orders/{order_id}").status_code == 200
    assert (
        isolated_client.patch(
            f"/api/orders/{order_id}/tracking",
            json={"tracking_status": "UNTRACKED"},
        ).json()["order"]["tracking_status"]
        == "UNTRACKED"
    )
    assert (
        isolated_client.post(
            f"/api/orders/{order_id}/update-pnl",
            json={"current_price": 1.50},
        ).json()["order"]["pnl"]
        == 60.0
    )
    closed = isolated_client.post(
        f"/api/orders/{order_id}/close",
        json={"exit_price": 1.65},
    ).json()["order"]
    assert closed["status"] == "CLOSED"
    assert closed["pnl"] == 90.0

    # Reject a second recommendation
    with get_db() as session:
        rec2 = TradeRecommendation(
            symbol="XOM",
            underlying_price=120.0,
            option_type=OptionType.CALL,
            structure="LONG_CALL",
            strike_price=122.0,
            expiration_date=exp,
            entry_price_estimate=2.0,
            target_exit_price=3.0,
            stop_loss_price=1.5,
            thesis="Energy hedge",
            status=TradeStatus.PENDING_APPROVAL,
        )
        session.add(rec2)
        session.flush()
        rec2_id = rec2.id

    r_rej = isolated_client.post(f"/api/recommendations/{rec2_id}/reject")
    assert r_rej.status_code == 200
    assert r_rej.json()["recommendation"]["status"] == "REJECTED"


def test_market_research_and_notification_apis(isolated_client: TestClient):
    snap = TechnicalSnapshot(symbol="NVDA", price=138.5, change_7d_pct=4.2, rsi_14=58.0, iv_rank=42.0)
    cand = CandidateTicker(symbol="NVDA", price=138.5, score=9.1, rationale="Leader", technicals=snap)
    dossier = SectorResearchDossier(
        sector_id="AI_CHIPS_INFRA",
        sector_name="AI & Chips",
        benchmark_etf="SMH",
        sector_7d_change_pct=3.5,
        top_candidates=[cand],
    )
    chain = OptionChainResult(
        symbol="NVDA",
        underlying_price=138.5,
        expiration=date.today() + timedelta(days=7),
        dte=7,
        calls=[
            OptionContract(
                contract_symbol="NVDA_C_140",
                option_type="CALL",
                strike=140.0,
                expiration=date.today() + timedelta(days=7),
                dte=7,
                bid=2.40,
                ask=2.60,
                mid=2.50,
            )
        ],
        puts=[],
    )

    with (
        patch("src.web.app.run_all_sector_subagents", return_value=([cand], [dossier])),
        patch("src.web.app.get_technical_snapshot", return_value=snap),
        patch("src.web.app.get_option_chain", return_value=chain),
    ):
        sec_res = isolated_client.get("/api/market/sectors?sectors=AI_CHIPS_INFRA")
        assert sec_res.status_code == 200
        assert sec_res.json()["candidates_count"] == 1

        q_res = isolated_client.get("/api/market/nvda/quote")
        assert q_res.status_code == 200
        assert q_res.json()["quote"]["price"] == 138.5

        opt_res = isolated_client.get("/api/market/nvda/options?option_type=CALL")
        assert opt_res.status_code == 200
        assert len(opt_res.json()["chain"]["calls"]) == 1

    notif_res = isolated_client.post("/api/notifications/test", json={"channel": "TELEGRAM"})
    assert notif_res.status_code == 200
    assert "skipped" in notif_res.json()["results"]["telegram"]


def test_postgres_dialect_engine_and_ddl():
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateTable

    from src.database.connection import _build_engine
    from src.database.models import Base

    engine = _build_engine("postgres://postgres:secret@pg.railway.internal:5432/railway")
    try:
        assert engine.dialect.name == "postgresql"
        assert engine.dialect.driver == "psycopg"
        pg_dialect = postgresql.dialect()
        for table in Base.metadata.sorted_tables:
            ddl = str(CreateTable(table).compile(dialect=pg_dialect))
            assert f"CREATE TABLE {table.name}" in ddl
    finally:
        engine.dispose()


def test_migrate_sqlite_data_idempotent(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from scripts.migrate_sqlite_to_postgres import migrate_data
    from src.database.models import Base, PlacedOrder

    src_url = f"sqlite:///{tmp_path / 'src.db'}"
    dst_url = f"sqlite:///{tmp_path / 'dst.db'}"

    src_engine = create_engine(src_url)
    Base.metadata.create_all(bind=src_engine)
    SrcSession = sessionmaker(bind=src_engine)
    with SrcSession() as s:
        s.add(
            TradeRecommendation(
                symbol="NVDA",
                underlying_price=140.0,
                option_type=OptionType.CALL,
                structure="LONG_CALL",
                strike_price=142.0,
                expiration_date=date.today() + timedelta(days=7),
                entry_price_estimate=2.5,
                target_exit_price=3.75,
                stop_loss_price=1.88,
                thesis="AI demand",
                status=TradeStatus.PENDING_APPROVAL,
            )
        )
        s.add(
            PlacedOrder(
                symbol="OKLO",
                structure="LONG_CALL",
                option_type="CALL",
                strike_price=24.0,
                expiration_date=date.today() + timedelta(days=7),
                contracts=1,
                fill_price=2.0,
                total_cost=200.0,
            )
        )
        s.commit()
    src_engine.dispose()

    first = migrate_data(src_url, dst_url)
    assert first == {"trade_recommendations": 1, "execution_logs": 0, "placed_orders": 1}

    second = migrate_data(src_url, dst_url)
    assert second == {"trade_recommendations": 0, "execution_logs": 0, "placed_orders": 0}


def test_options_math_returns_native_python_numbers_and_msgpack_serializable():
    import dataclasses

    import numpy as np
    import ormsgpack

    from src.tools.options_math import (
        black_scholes_greeks,
        breakeven,
        pnl_at_expiry,
        score_contract,
        size_position,
    )

    greeks = black_scholes_greeks(
        np.float64(173.73),
        np.float64(175.0),
        7 / 365,
        0.045,
        np.float64(0.46),
        "CALL",
    )
    be = breakeven(np.float64(175.0), np.float64(3.25), "CALL")
    curve = pnl_at_expiry(
        "CALL",
        np.float64(175.0),
        np.float64(3.25),
        underlying_price=np.float64(173.73),
    )
    pos = size_position(np.float64(500.0), 100.0, np.float64(3.25))
    scored = score_contract(
        contract_mid=np.float64(3.25),
        strike=np.float64(175.0),
        dte=np.int64(7),  # type: ignore[arg-type]
        underlying_price=np.float64(173.73),
        option_type="CALL",
        contract_symbol="NVDA_C_175",
        iv_override=np.float64(0.46),
    )

    assert type(be) is float
    assert scored is not None
    payload = {
        "greeks": dataclasses.asdict(greeks),
        "breakeven": be,
        "curve": dataclasses.asdict(curve),
        "position": dataclasses.asdict(pos),
        "scored": dataclasses.asdict(scored),
    }

    def _assert_no_numpy(obj):
        assert not isinstance(obj, np.generic)
        if isinstance(obj, dict):
            for v in obj.values():
                _assert_no_numpy(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _assert_no_numpy(v)

    _assert_no_numpy(payload)
    packed = ormsgpack.packb(payload)
    assert isinstance(packed, bytes)


def test_langgraph_checkpointer_handles_numpy_scalars():
    import numpy as np

    from src.workflow.graph import build_graph

    def fake_scanner(_state):
        return {
            "scanner_proposals": [
                {"symbol": "NVDA", "score": np.float64(9.1), "rank": np.int64(1)}
            ],
            "macro_political_context": {"risk_on": np.bool_(True)},
        }

    def fake_structurer(_state):
        return {
            "structured_trades": [
                {
                    "symbol": "NVDA",
                    "delta": np.float64(0.44),
                    "levels": np.array([170.0, 175.0]),
                }
            ]
        }

    def fake_risk(_state):
        return {
            "risk_reviews": [{"symbol": "NVDA", "approved": np.bool_(True)}],
            "final_recommendations": [
                {"symbol": "NVDA", "entry_price": np.float64(3.25)}
            ],
            "review_feedback": None,
        }

    def fake_persistence(_state):
        return {"saved_recommendation_ids": [np.int64(42)]}

    with (
        patch("src.workflow.graph.scanner_node", fake_scanner),
        patch("src.workflow.graph.profitability_node", fake_structurer),
        patch("src.workflow.graph.risk_node", fake_risk),
        patch("src.workflow.graph.persistence_node", fake_persistence),
    ):
        compiled = build_graph(use_checkpointer=True)
        out = compiled.invoke(
            {
                "target_symbols": ["NVDA"],
                "weekly_budget_usd": 500.0,
                "max_cost_per_trade_usd": 500.0,
                "scanner_proposals": [],
                "structured_trades": [],
                "risk_reviews": [],
                "final_recommendations": [],
                "iteration_count": 0,
                "review_feedback": None,
                "workflow_error": None,
                "run_id": "test-np-msgpack",
                "notify": False,
            },
            config={"configurable": {"thread_id": "test-np-msgpack"}},
        )

    assert type(out["scanner_proposals"][0]["score"]) is float
    assert type(out["scanner_proposals"][0]["rank"]) is int
    assert type(out["macro_political_context"]["risk_on"]) is bool
    assert out["structured_trades"][0]["levels"] == [170.0, 175.0]
    assert out["saved_recommendation_ids"] == [42]
    assert type(out["saved_recommendation_ids"][0]) is int
