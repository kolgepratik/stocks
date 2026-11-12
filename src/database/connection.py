"""
Database Connection – SQLAlchemy engine, session factory, and table initializer.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from loguru import logger
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_settings, normalize_database_url
from src.database.models import AppSetting, Base

_db_lock = threading.RLock()
_engine = None
_engine_url: str | None = None
_SessionLocal = None


def _build_engine(db_url: str):
    normalized = normalize_database_url(db_url)
    is_sqlite = normalized.startswith("sqlite")

    if is_sqlite:
        db_path = normalized.replace("sqlite:///", "", 1).replace("sqlite://", "", 1)
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            normalized,
            connect_args={"check_same_thread": False},
            echo=False,
        )

        @event.listens_for(engine, "connect")
        def set_wal(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

        return engine

    return create_engine(
        normalized,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=5,
        max_overflow=10,
        echo=False,
    )


def get_engine_for_url(db_url: str):
    """Return a cached SQLAlchemy Engine bound to `db_url` (never calls get_settings)."""
    global _engine, _engine_url, _SessionLocal
    normalized = normalize_database_url(db_url)
    with _db_lock:
        if _engine is None or _engine_url != normalized:
            if _engine is not None:
                try:
                    _engine.dispose()
                except Exception:
                    pass
            _engine = _build_engine(normalized)
            _engine_url = normalized
            _SessionLocal = None
        return _engine


def reset_engine() -> None:
    """Dispose the active SQLAlchemy engine and session factory (for tests/reloads)."""
    global _engine, _engine_url, _SessionLocal
    with _db_lock:
        if _engine is not None:
            try:
                _engine.dispose()
            except Exception:
                pass
        _engine = None
        _engine_url = None
        _SessionLocal = None


def get_engine():
    settings = get_settings()
    return get_engine_for_url(settings.database_url)


def get_session_factory() -> sessionmaker:
    global _SessionLocal
    engine = get_engine()
    with _db_lock:
        if _SessionLocal is None:
            _SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        return _SessionLocal


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Provide a transactional database session as a context manager."""
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def read_runtime_overrides(db_url: str) -> dict[str, str]:
    """Read DB-persisted setting overrides from `app_settings` without invoking get_settings()."""
    try:
        engine = get_engine_for_url(db_url)
        if not inspect(engine).has_table(AppSetting.__tablename__):
            return {}
        with engine.connect() as conn:
            rows = conn.execute(text(f"SELECT key, value FROM {AppSetting.__tablename__}")).fetchall()
        return {str(row[0]): str(row[1]) for row in rows}
    except Exception:
        return {}


def upsert_runtime_overrides(db_url: str, updates: dict[str, str]) -> None:
    """Upsert key-value runtime settings into the `app_settings` table."""
    if not updates:
        return
    engine = get_engine_for_url(db_url)
    AppSetting.__table__.create(bind=engine, checkfirst=True)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = factory()
    try:
        for key, val in updates.items():
            existing = session.get(AppSetting, key)
            if existing is not None:
                existing.value = str(val)
            else:
                session.add(AppSetting(key=key, value=str(val)))
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables if they don't already exist and apply lightweight schema migrations."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)

    dialect = engine.dialect.name
    if dialect == "sqlite":
        with engine.connect() as conn:
            for table_name, table in Base.metadata.tables.items():
                existing = [row[1] for row in conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()]
                for col in table.columns:
                    if col.name not in existing:
                        col_type = col.type.compile(engine.dialect)
                        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col.name} {col_type}"))
            conn.commit()
    elif dialect == "postgresql":
        with engine.connect() as conn:
            for table_name, table in Base.metadata.tables.items():
                rows = conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = current_schema() AND table_name = :tbl"
                    ),
                    {"tbl": table_name},
                ).fetchall()
                existing = {row[0] for row in rows}
                for col in table.columns:
                    if col.name not in existing:
                        col_type = col.type.compile(engine.dialect)
                        conn.execute(
                            text(f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS "{col.name}" {col_type}')
                        )
            conn.commit()

    get_settings.cache_clear()
    db_display = normalize_database_url(get_settings().database_url)
    if "@" in db_display:
        # Mask credentials in startup logs
        prefix, rest = db_display.split("://", 1)
        host_part = rest.split("@", 1)[1]
        db_display = f"{prefix}://***@{host_part}"
    logger.info("Database initialized ({}): {}", dialect, db_display)


def get_db_dialect() -> str:
    """Return active SQLAlchemy dialect name (e.g. 'postgresql', 'sqlite')."""
    return get_engine().dialect.name


def health_check() -> bool:
    """Verify the database is reachable."""
    try:
        with get_db() as session:
            session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("DB health check failed: {}", exc)
        return False
