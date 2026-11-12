"""
FastAPI Web Application & API Backend
======================================
Cloud-ready REST API & Web UX for Antigravity Options AI (Railway + PostgreSQL / local SQLite).

Exposes every pipeline & portfolio operation as a REST endpoint:
- Health & readiness: GET /health
- Pipeline scans (DB-backed jobs + sync + cron): /api/scan/*, /api/cron/weekly-scan
- Recommendations lifecycle: GET/POST /api/recommendations/* (/approve, /reject, /execute)
- Placed orders & live monitoring: /api/orders/*
- Market research (sectors, quotes, option chains): /api/market/*
- Agent chat, settings (DB-persisted), stats & notifications: /api/chat, /api/settings, /api/stats, /api/notifications/test
- Telegram webhook (single-container bot handling): POST /api/webhooks/telegram
"""
from __future__ import annotations

import hmac
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Literal, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import func

from config.settings import get_settings, save_runtime_settings
from src.agents.chat_agent import run_chat_turn
from src.agents.sector_subagents import SECTOR_CONFIGS, run_all_sector_subagents
from src.database.connection import get_db, get_db_dialect, health_check, init_db
from src.database.models import (
    BrokerName,
    ExecutionLog,
    PlacedOrder,
    ScanJob,
    TradeRecommendation,
    TradeStatus,
)
from src.tools.broker import execute_option_order
from src.tools.market_data import get_option_chain, get_technical_snapshot
from src.tools.order_tracker import (
    close_placed_order,
    get_placed_order_by_id,
    get_placed_orders_grouped_by_date,
    get_recommendation_by_id,
    get_recommendations_grouped_by_date,
    monitor_placed_orders,
    record_placed_order,
    set_order_tracking_state,
    update_order_pnl,
    update_recommendation_status,
)
from src.workflow.graph import run_pipeline


# ── DB-Backed Scan Job Helpers ────────────────────────────────────────────────

_STAGE_LABELS = {
    "scanner_node": "2/4 Structuring defined-risk CALL strategies & Greeks...",
    "structurer_node": "3/4 Running Risk Sentinel compliance & budget checks...",
    "persistence_node": "Finalizing ranked recommendations...",
}


def _mark_orphaned_jobs_failed() -> None:
    """Mark any `running` scan jobs left over from a prior container crash/redeploy as failed."""
    try:
        with get_db() as session:
            stale = session.query(ScanJob).filter(ScanJob.status == "running").all()
            now = time.time()
            for job in stale:
                job.status = "failed"
                job.stage = "Scan failed"
                job.completed_at = now
                job.error = "Interrupted by server restart"
            if stale:
                logger.warning("Marked {} interrupted scan job(s) as failed on startup", len(stale))
    except Exception as exc:
        logger.debug("Skipping stale job cleanup: {}", exc)


def _create_scan_job_record(
    job_id: str,
    symbols: list[str] | None,
    budget: float | None,
    max_trade_cost: float | None,
    notify: bool,
    stage: str,
) -> dict[str, Any]:
    now = time.time()
    with get_db() as session:
        job = ScanJob(
            job_id=job_id,
            status="running",
            stage=stage,
            symbols_json=json.dumps(symbols) if symbols else None,
            budget=budget,
            max_trade_cost=max_trade_cost,
            notify=notify,
            started_at=now,
        )
        session.add(job)
        session.flush()
        return job.to_snapshot()


def _update_scan_job_record(job_id: str, **fields: Any) -> None:
    with get_db() as session:
        job = session.query(ScanJob).filter(ScanJob.job_id == job_id).first()
        if not job:
            return
        for key, value in fields.items():
            setattr(job, key, value)


# ── Auth & CORS Helpers ───────────────────────────────────────────────────────

def _extract_provided_secret(request: Request, alt_header: str | None = None) -> str:
    if alt_header:
        alt_val = request.headers.get(alt_header, "").strip()
        if alt_val:
            return alt_val
    api_key_hdr = request.headers.get("X-API-Key", "").strip()
    if api_key_hdr:
        return api_key_hdr
    auth_hdr = request.headers.get("Authorization", "").strip()
    if auth_hdr.lower().startswith("bearer "):
        return auth_hdr[7:].strip()
    return ""


def require_api_auth(request: Request) -> None:
    """Enforce API key auth on /api/* endpoints whenever API_KEY is configured."""
    expected = get_settings().api_key.strip()
    if not expected:
        return
    provided = _extract_provided_secret(request)
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: provide a valid X-API-Key or Authorization: Bearer token",
        )


def require_cron_auth(request: Request) -> None:
    """Enforce CRON_SECRET (or fallback API_KEY) on scheduled cron endpoints."""
    settings = get_settings()
    expected = (settings.cron_secret or settings.api_key).strip()
    if not expected:
        return
    provided = _extract_provided_secret(request, alt_header="X-Cron-Secret")
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Unauthorized cron request")


# ── App Lifespan & Initialization ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    _mark_orphaned_jobs_failed()
    logger.info("🚀 Web application started ({})", get_db_dialect())
    yield


app = FastAPI(
    title="Antigravity Options Trading AI",
    description="Multi-Agent Options Advisory & Execution Engine with Cloud PostgreSQL & REST API",
    version="2.1.0",
    lifespan=lifespan,
)

_initial_settings = get_settings()
_cors_origins = _initial_settings.cors_origins_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

api_router = APIRouter(prefix="/api", dependencies=[Depends(require_api_auth)])
cron_router = APIRouter(prefix="/api/cron", dependencies=[Depends(require_cron_auth)])


# ── Public UI & Health Endpoints ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    settings = get_settings()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "weekly_budget": settings.weekly_budget_usd,
            "max_trade_cost": settings.max_cost_per_trade_usd,
            "llm_provider": settings.llm_provider,
            "anthropic_model": settings.anthropic_model,
            "notification_channel": settings.notification_channel,
            "auth_required": bool(settings.api_key.strip()),
        },
    )


@app.get("/health")
def health_endpoint():
    """Health & DB readiness endpoint for Railway / load balancers."""
    db_ok = health_check()
    dialect = "unknown"
    if db_ok:
        try:
            dialect = get_db_dialect()
        except Exception:
            pass
    payload = {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "unreachable",
        "dialect": dialect,
    }
    return JSONResponse(content=payload, status_code=200 if db_ok else 503)


# ── Pydantic Request Schemas ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="User prompt or instruction")
    history: Optional[List[dict[str, str]]] = Field(default=None, description="Previous conversation turns")


class RecordOrderRequest(BaseModel):
    symbol: str
    fill_price: float
    contracts: int = 1
    strike_price: Optional[float] = None
    short_strike_price: Optional[float] = None
    expiration_date: Optional[str] = None
    structure: str = "LONG_CALL"
    option_type: str = "CALL"
    recommendation_id: Optional[int] = None
    notes: Optional[str] = None
    tracking_status: str = "TRACKED"


class UpdateTrackingRequest(BaseModel):
    tracking_status: str = Field(..., description="'TRACKED' or 'UNTRACKED'")


class UpdatePnLRequest(BaseModel):
    pnl: Optional[float] = None
    current_price: Optional[float] = None
    notes: Optional[str] = None


class CloseOrderRequest(BaseModel):
    exit_price: float
    notes: Optional[str] = None


class ExecuteRecommendationRequest(BaseModel):
    broker: Literal["ALPACA", "ROBINHOOD", "PAPER"] = "PAPER"
    contracts: Optional[int] = None
    limit_price: Optional[float] = None
    record_placed_order: bool = True
    notes: Optional[str] = None


class SettingsUpdateRequest(BaseModel):
    weekly_budget_usd: Optional[float] = None
    max_cost_per_trade_usd: Optional[float] = None
    target_gain_percent: Optional[float] = None
    stop_loss_percent: Optional[float] = None
    llm_provider: Optional[str] = None
    anthropic_model: Optional[str] = None
    gemini_model: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    notification_channel: Optional[str] = None
    scan_universe: Optional[str] = None
    strategy_mode: Optional[str] = None
    target_dte: Optional[int] = None
    max_candidates: Optional[int] = None


class ScanTriggerRequest(BaseModel):
    symbols: Optional[List[str]] = None
    budget: Optional[float] = None
    max_trade_cost: Optional[float] = None
    notify: bool = False


class NotificationTestRequest(BaseModel):
    channel: Optional[Literal["TELEGRAM", "WHATSAPP", "BOTH"]] = None
    message: Optional[str] = None


# ── Chat API ──────────────────────────────────────────────────────────────────

@api_router.post("/chat")
def chat_endpoint(req: ChatRequest):
    try:
        reply = run_chat_turn(user_message=req.message, chat_history=req.history)
        return {"response": reply}
    except Exception as exc:
        logger.error("Chat turn failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Recommendations API ───────────────────────────────────────────────────────

@api_router.get("/recommendations")
def get_recommendations_endpoint(
    timeframe: str = Query("all", description="7d | 14d | 1m | 1q | 1y | all"),
    status: str = Query("ALL", description="ALL | PENDING_APPROVAL | APPROVED | REJECTED | EXECUTED"),
):
    try:
        groups = get_recommendations_grouped_by_date(timeframe=timeframe, status=status)
        return {"timeframe": timeframe, "status": status, "groups": groups}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to fetch recommendations: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@api_router.get("/recommendations/{rec_id}")
def get_recommendation_endpoint(rec_id: int):
    try:
        rec = get_recommendation_by_id(rec_id)
        return {"recommendation": rec}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to fetch recommendation #{}: {}", rec_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@api_router.post("/recommendations/{rec_id}/approve")
def approve_recommendation_endpoint(rec_id: int):
    try:
        rec = update_recommendation_status(rec_id, TradeStatus.APPROVED.value)
        return {"status": "success", "recommendation": rec}
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to approve recommendation #{}: {}", rec_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@api_router.post("/recommendations/{rec_id}/reject")
def reject_recommendation_endpoint(rec_id: int):
    try:
        rec = update_recommendation_status(rec_id, TradeStatus.REJECTED.value)
        return {"status": "success", "recommendation": rec}
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to reject recommendation #{}: {}", rec_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@api_router.post("/recommendations/{rec_id}/execute")
async def execute_recommendation_endpoint(rec_id: int, req: ExecuteRecommendationRequest):
    try:
        rec_data = get_recommendation_by_id(rec_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    qty = req.contracts or rec_data.get("max_contracts") or 1
    limit_px = req.limit_price if req.limit_price is not None else rec_data["entry_price_estimate"]

    try:
        result = await execute_option_order(
            broker=req.broker,
            symbol=rec_data["symbol"],
            option_type=rec_data["option_type"],
            strike=rec_data["strike_price"],
            expiration=rec_data["expiration_date"],
            quantity=qty,
            limit_price=limit_px,
            contract_symbol=rec_data.get("contract_symbol") or "",
        )

        with get_db() as session:
            rec_obj = session.get(TradeRecommendation, rec_id)
            if rec_obj is None:
                raise HTTPException(status_code=404, detail=f"Recommendation #{rec_id} not found")
            session.add(
                ExecutionLog(
                    recommendation_id=rec_id,
                    broker=BrokerName(req.broker),
                    order_id=str(result.get("order_id", "unknown")),
                    status=str(result.get("status", "submitted")),
                    filled_avg_price=result.get("filled_price"),
                    quantity=qty,
                    raw_response=str(result),
                )
            )
            if result.get("status") not in ("error", "rejected"):
                rec_obj.status = TradeStatus.EXECUTED

        placed_order = None
        if req.record_placed_order and result.get("status") not in ("error", "rejected"):
            fill = float(result.get("filled_price") or limit_px)
            placed_order = record_placed_order(
                symbol=rec_data["symbol"],
                fill_price=fill,
                contracts=qty,
                strike_price=rec_data["strike_price"],
                short_strike_price=rec_data.get("short_strike_price"),
                expiration_date=rec_data["expiration_date"],
                structure=rec_data.get("structure") or "LONG_CALL",
                option_type=rec_data["option_type"],
                recommendation_id=rec_id,
                notes=req.notes or f"Executed via {req.broker} API (order_id={result.get('order_id')})",
            )

        return {
            "status": "success",
            "execution": result,
            "recommendation": get_recommendation_by_id(rec_id),
            "placed_order": placed_order,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to execute recommendation #{}: {}", rec_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Placed Orders API ─────────────────────────────────────────────────────────

@api_router.get("/orders")
def get_orders_endpoint(
    timeframe: str = Query("all", description="7d | 14d | 1m | 1q | 1y | all"),
    tracking_status: str = Query("ALL", description="ALL | TRACKED | UNTRACKED"),
    status: str = Query("ALL", description="ALL | OPEN | CLOSED"),
):
    try:
        groups = get_placed_orders_grouped_by_date(
            timeframe=timeframe,
            tracking_status=tracking_status,
            status=status,
        )
        return {
            "timeframe": timeframe,
            "tracking_status": tracking_status,
            "status": status,
            "groups": groups,
        }
    except Exception as exc:
        logger.error("Failed to fetch orders: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@api_router.get("/orders/{order_id}")
def get_order_endpoint(order_id: int):
    try:
        return {"order": get_placed_order_by_id(order_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to fetch order #{}: {}", order_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@api_router.post("/orders")
def record_order_endpoint(req: RecordOrderRequest):
    try:
        order = record_placed_order(
            symbol=req.symbol,
            fill_price=req.fill_price,
            contracts=req.contracts,
            strike_price=req.strike_price,
            short_strike_price=req.short_strike_price,
            expiration_date=req.expiration_date,
            structure=req.structure,
            option_type=req.option_type,
            recommendation_id=req.recommendation_id,
            notes=req.notes,
            tracking_status=req.tracking_status,
        )
        return {"status": "success", "order": order}
    except Exception as exc:
        logger.error("Failed to record order: {}", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@api_router.patch("/orders/{order_id}/tracking")
def set_tracking_endpoint(order_id: int, req: UpdateTrackingRequest):
    try:
        order = set_order_tracking_state(order_id=order_id, tracking_status=req.tracking_status)
        return {"status": "success", "order": order}
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to set tracking state: {}", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@api_router.post("/orders/{order_id}/update-pnl")
def update_pnl_endpoint(order_id: int, req: UpdatePnLRequest):
    try:
        order = update_order_pnl(
            order_id=order_id,
            pnl=req.pnl,
            current_price=req.current_price,
            notes=req.notes,
        )
        return {"status": "success", "order": order}
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to update PnL: {}", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@api_router.post("/orders/{order_id}/close")
def close_order_endpoint(order_id: int, req: CloseOrderRequest):
    try:
        order = close_placed_order(
            order_id=order_id,
            exit_price=req.exit_price,
            notes=req.notes,
        )
        return {"status": "success", "order": order}
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to close order: {}", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@api_router.post("/orders/monitor")
def monitor_orders_endpoint(tracking_only: bool = Query(True)):
    try:
        report = monitor_placed_orders(tracking_only=tracking_only)
        return {"status": "success", "report": report}
    except Exception as exc:
        logger.error("Failed to monitor orders: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Scan Pipeline API (DB-Backed) ─────────────────────────────────────────────

def _clean_symbols_list(symbols: Optional[List[str]]) -> list[str] | None:
    if not symbols:
        return None
    cleaned = [s.strip().upper() for s in symbols if s and s.strip()]
    return cleaned or None


def _run_scan_and_persist_job(
    job_id: str,
    symbols: list[str] | None,
    budget: float | None,
    max_trade_cost: float | None,
    notify: bool,
) -> dict[str, Any]:
    def _on_node(node_name: str, state: dict[str, Any]) -> None:
        if node_name == "risk_node":
            if state.get("review_feedback") and state.get("iteration_count", 0) <= 2:
                stage = f"Refining trade structures (Risk Sentinel revision loop {state.get('iteration_count', 1)})..."
            else:
                stage = "4/4 Ranking & persisting approved recommendations..."
        elif node_name in _STAGE_LABELS:
            stage = _STAGE_LABELS[node_name]
        else:
            return
        _update_scan_job_record(job_id, stage=stage)

    try:
        final_state = run_pipeline(
            symbols=symbols,
            weekly_budget_usd=budget,
            max_cost_per_trade_usd=max_trade_cost,
            notify=notify,
            progress_callback=_on_node,
        )
        recs = final_state.get("final_recommendations") or []
        macro = final_state.get("macro_political_context") or {}
        saved_ids = final_state.get("saved_recommendation_ids") or []
        _update_scan_job_record(
            job_id,
            status="completed",
            stage=f"Completed ({len(recs)} approved)",
            completed_at=time.time(),
            recommendations_count=len(recs),
            recommendations_json=json.dumps(recs, default=str),
            saved_ids_json=json.dumps(saved_ids),
            macro_summary=macro.get("us_political_macro_summary"),
            political_catalysts_json=json.dumps(macro.get("political_catalysts", [])),
        )
        return {
            "job_id": job_id,
            "status": "success",
            "recommendations_count": len(recs),
            "recommendations": recs,
            "saved_ids": saved_ids,
            "macro_summary": macro.get("us_political_macro_summary"),
            "political_catalysts": macro.get("political_catalysts", []),
        }
    except Exception as exc:
        logger.error("Scan job {} failed: {}", job_id, exc)
        _update_scan_job_record(
            job_id,
            status="failed",
            stage="Scan failed",
            completed_at=time.time(),
            error=str(exc),
        )
        raise


def _execute_scan_job(
    job_id: str,
    symbols: list[str] | None,
    budget: float | None,
    max_trade_cost: float | None,
    notify: bool,
) -> None:
    try:
        _run_scan_and_persist_job(job_id, symbols, budget, max_trade_cost, notify)
    except Exception:
        # Failure details already persisted on ScanJob row
        pass


@api_router.post("/scan/start")
def start_scan_job_endpoint(req: ScanTriggerRequest):
    symbols = _clean_symbols_list(req.symbols)
    job_id = str(uuid.uuid4())
    initial_stage = "1/4 Screening thematic sectors, news feeds & momentum..."
    snapshot = _create_scan_job_record(
        job_id=job_id,
        symbols=symbols,
        budget=req.budget,
        max_trade_cost=req.max_trade_cost,
        notify=req.notify,
        stage=initial_stage,
    )

    t = threading.Thread(
        target=_execute_scan_job,
        args=(job_id, symbols, req.budget, req.max_trade_cost, req.notify),
        daemon=True,
    )
    t.start()
    logger.info("Started background scan job {} (symbols={}, budget={})", job_id, symbols, req.budget)
    return {"job_id": job_id, "status": "running", "stage": snapshot["stage"]}


@api_router.get("/scan/status/{job_id}")
def get_scan_status_endpoint(job_id: str):
    with get_db() as session:
        job = session.query(ScanJob).filter(ScanJob.job_id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Scan job not found")
        return job.to_snapshot()


@api_router.get("/scan/latest")
def get_latest_scan_endpoint():
    with get_db() as session:
        job = session.query(ScanJob).order_by(ScanJob.started_at.desc()).first()
        if not job:
            return {"status": "none"}
        return job.to_snapshot()


@api_router.get("/scan/jobs")
def list_scan_jobs_endpoint(limit: int = Query(20, ge=1, le=100)):
    with get_db() as session:
        jobs = session.query(ScanJob).order_by(ScanJob.started_at.desc()).limit(limit).all()
        return {"count": len(jobs), "jobs": [j.to_snapshot() for j in jobs]}


@api_router.post("/scan")
def trigger_scan_endpoint(req: ScanTriggerRequest):
    symbols = _clean_symbols_list(req.symbols)
    job_id = str(uuid.uuid4())
    _create_scan_job_record(
        job_id=job_id,
        symbols=symbols,
        budget=req.budget,
        max_trade_cost=req.max_trade_cost,
        notify=req.notify,
        stage="1/4 Running synchronous pipeline scan...",
    )
    try:
        logger.info("Triggering synchronous scan via Web API: symbols={} budget={}", symbols, req.budget)
        return _run_scan_and_persist_job(job_id, symbols, req.budget, req.max_trade_cost, req.notify)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@cron_router.post("/weekly-scan")
def cron_weekly_scan_endpoint(
    req: Optional[ScanTriggerRequest] = None,
    async_mode: bool = Query(True, description="Run in background and return job_id immediately"),
):
    """Scheduled scan endpoint for Railway Cron / external schedulers (notifies by default)."""
    payload = req or ScanTriggerRequest(notify=True)
    symbols = _clean_symbols_list(payload.symbols)
    notify = True if req is None else payload.notify
    job_id = str(uuid.uuid4())

    if async_mode:
        snapshot = _create_scan_job_record(
            job_id=job_id,
            symbols=symbols,
            budget=payload.budget,
            max_trade_cost=payload.max_trade_cost,
            notify=notify,
            stage="1/4 Scheduled cron scan queued...",
        )
        t = threading.Thread(
            target=_execute_scan_job,
            args=(job_id, symbols, payload.budget, payload.max_trade_cost, notify),
            daemon=True,
        )
        t.start()
        return {"job_id": job_id, "status": "running", "stage": snapshot["stage"], "notify": notify}

    _create_scan_job_record(
        job_id=job_id,
        symbols=symbols,
        budget=payload.budget,
        max_trade_cost=payload.max_trade_cost,
        notify=notify,
        stage="1/4 Running scheduled cron scan...",
    )
    try:
        return _run_scan_and_persist_job(job_id, symbols, payload.budget, payload.max_trade_cost, notify)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Market Research APIs ──────────────────────────────────────────────────────

@api_router.get("/market/sectors")
def get_market_sectors_endpoint(
    sectors: Optional[str] = Query(None, description="Comma-separated sector IDs (e.g. AI_CHIPS_INFRA,WAR_ENERGY_GAS)"),
    picks_per_sector: int = Query(2, ge=1, le=5),
):
    try:
        target_list = None
        if sectors:
            target_list = [s.strip().upper() for s in sectors.split(",") if s.strip()]
            invalid = [s for s in target_list if s not in SECTOR_CONFIGS]
            if invalid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown sector(s): {invalid}. Allowed: {list(SECTOR_CONFIGS.keys())}",
                )
        candidates, dossiers = run_all_sector_subagents(
            target_sectors=target_list,
            picks_per_sector=picks_per_sector,
        )
        return {
            "sectors": [d.model_dump() for d in dossiers],
            "candidates_count": len(candidates),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Sector research API failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@api_router.get("/market/{symbol}/quote")
def get_market_quote_endpoint(
    symbol: str,
    period: str = Query("1y", description="Lookback period (e.g. 3mo, 6mo, 1y)"),
):
    try:
        snap = get_technical_snapshot(symbol.upper().strip(), period=period)
        if not snap:
            raise HTTPException(status_code=404, detail=f"Market data unavailable for {symbol.upper()}")
        return {"quote": snap.model_dump()}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Market quote API failed for {}: {}", symbol, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@api_router.get("/market/{symbol}/options")
def get_market_options_endpoint(
    symbol: str,
    target_dte: int = Query(7, ge=0, le=365),
    expiration: Optional[str] = Query(None, description="Exact expiry YYYY-MM-DD"),
    option_type: Literal["ALL", "CALL", "PUT"] = Query("ALL"),
):
    try:
        chain = get_option_chain(
            symbol.upper().strip(),
            target_dte=target_dte,
            exact_expiration=expiration,
        )
        if not chain:
            raise HTTPException(status_code=404, detail=f"Options chain unavailable for {symbol.upper()}")
        data = chain.model_dump()
        if option_type == "CALL":
            data["puts"] = []
        elif option_type == "PUT":
            data["calls"] = []
        return {"chain": data}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Options chain API failed for {}: {}", symbol, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Settings API ──────────────────────────────────────────────────────────────

@api_router.get("/settings")
def get_settings_endpoint():
    settings = get_settings()
    ant_key = settings.anthropic_api_key
    gem_key = settings.gemini_api_key
    masked_ant = f"...{ant_key[-6:]}" if len(ant_key) > 6 else ("Configured" if ant_key else "")
    masked_gem = f"...{gem_key[-6:]}" if len(gem_key) > 6 else ("Configured" if gem_key else "")

    return {
        "weekly_budget_usd": settings.weekly_budget_usd,
        "max_cost_per_trade_usd": settings.max_cost_per_trade_usd,
        "target_gain_percent": settings.target_gain_percent,
        "stop_loss_percent": settings.stop_loss_percent,
        "llm_provider": settings.llm_provider,
        "anthropic_model": settings.anthropic_model,
        "gemini_model": settings.gemini_model,
        "notification_channel": settings.notification_channel,
        "scan_universe": settings.scan_universe,
        "strategy_mode": settings.strategy_mode,
        "target_dte": settings.target_dte,
        "max_candidates": settings.max_candidates,
        "anthropic_api_key_configured": bool(ant_key),
        "anthropic_api_key_masked": masked_ant,
        "gemini_api_key_configured": bool(gem_key),
        "gemini_api_key_masked": masked_gem,
        "telegram_configured": bool(settings.telegram_bot_token and settings.telegram_chat_id),
        "whatsapp_configured": bool(settings.whatsapp_phone and settings.whatsapp_api_key),
        "auth_enabled": bool(settings.api_key.strip()),
    }


@api_router.post("/settings")
def update_settings_endpoint(req: SettingsUpdateRequest):
    try:
        updates = req.model_dump(exclude_unset=True)
        save_runtime_settings(updates)
        logger.info("Runtime settings persisted and reloaded")
        return {"status": "success", "message": "Settings updated successfully"}
    except Exception as exc:
        logger.error("Failed to update settings: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Dashboard Stats API ───────────────────────────────────────────────────────

@api_router.get("/stats")
def get_stats_endpoint():
    settings = get_settings()
    with get_db() as session:
        total_recs = session.query(func.count(TradeRecommendation.id)).scalar() or 0
        tracked_orders = session.query(PlacedOrder).filter(
            PlacedOrder.status == "OPEN",
            PlacedOrder.tracking_status == "TRACKED",
        ).all()

        untracked_orders = session.query(PlacedOrder).filter(
            PlacedOrder.status == "OPEN",
            PlacedOrder.tracking_status == "UNTRACKED",
        ).all()

        closed_orders = session.query(PlacedOrder).filter(PlacedOrder.status == "CLOSED").all()

        total_tracked_cost = sum(o.total_cost for o in tracked_orders)
        total_tracked_val = sum(o.current_value or o.total_cost for o in tracked_orders)
        unrealized_pnl = sum(o.pnl for o in tracked_orders)
        realized_pnl = sum(o.pnl for o in closed_orders)

        return {
            "weekly_budget": settings.weekly_budget_usd,
            "max_cost_per_trade": settings.max_cost_per_trade_usd,
            "active_tracked_orders_count": len(tracked_orders),
            "untracked_orders_count": len(untracked_orders),
            "total_invested_tracked": round(total_tracked_cost, 2),
            "current_portfolio_value": round(total_tracked_val, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "realized_pnl": round(realized_pnl, 2),
            "total_recommendations": total_recs,
            "active_model": settings.anthropic_model if settings.llm_provider == "anthropic" else settings.gemini_model,
            "provider": settings.llm_provider,
            "database_dialect": get_db_dialect(),
        }


# ── Notifications & Telegram Webhook ──────────────────────────────────────────

@api_router.post("/notifications/test")
async def test_notification_endpoint(req: NotificationTestRequest):
    """Send a test alert to Telegram and/or WhatsApp using active config."""
    settings = get_settings()
    channel = (req.channel or settings.notification_channel).upper()
    msg = req.message or "Antigravity Options AI: cloud notification test OK"
    results: dict[str, str] = {}

    if channel in ("TELEGRAM", "BOTH"):
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            results["telegram"] = "skipped (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing)"
        else:
            from telegram import Bot
            await Bot(token=settings.telegram_bot_token).send_message(
                chat_id=settings.telegram_chat_id,
                text=msg,
            )
            results["telegram"] = "sent"

    if channel in ("WHATSAPP", "BOTH"):
        if not settings.whatsapp_phone or not settings.whatsapp_api_key:
            results["whatsapp"] = "skipped (WHATSAPP_PHONE or WHATSAPP_API_KEY missing)"
        else:
            from src.notifications.whatsapp_bot import send_whatsapp_message
            ok = await send_whatsapp_message(msg)
            results["whatsapp"] = "sent" if ok else "failed"

    if not results:
        results["channel"] = f"skipped (channel={channel})"

    return {"status": "success", "channel": channel, "results": results}


@app.post("/api/webhooks/telegram")
async def telegram_webhook_endpoint(request: Request):
    """
    Receive Telegram Bot API webhook updates in the main FastAPI process
    (eliminates the need for a separate polling container on Railway).
    """
    settings = get_settings()
    expected_secret = settings.telegram_webhook_secret.strip()
    if expected_secret:
        header_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "").strip()
        if not header_secret or not hmac.compare_digest(header_secret, expected_secret):
            raise HTTPException(status_code=401, detail="Invalid Telegram webhook secret")

    if not settings.telegram_bot_token:
        raise HTTPException(status_code=503, detail="TELEGRAM_BOT_TOKEN is not configured")

    try:
        payload = await request.json()
        from src.notifications.telegram_bot import process_webhook_update
        await process_webhook_update(payload)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Telegram webhook processing failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# Mount authenticated routers
app.include_router(api_router)
app.include_router(cron_router)
