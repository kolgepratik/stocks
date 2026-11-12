"""
Migrate Local SQLite Data to Railway PostgreSQL
================================================
One-time helper script to copy existing recommendations, execution logs,
and placed orders from local `sqlite:///./data/trading.db` into your
cloud Railway PostgreSQL database (`DATABASE_URL`).

Usage:
    python scripts/migrate_sqlite_to_postgres.py --target "postgresql://postgres:...@host:port/railway"
    # Or using DATABASE_URL env var:
    DATABASE_URL="postgresql://..." python scripts/migrate_sqlite_to_postgres.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config.settings import normalize_database_url
from src.database.models import Base, ExecutionLog, PlacedOrder, TradeRecommendation


def migrate_data(source_url: str, target_url: str) -> dict[str, int]:
    """Copy rows from source SQLite DB to target PostgreSQL DB idempotently."""
    src_engine = create_engine(normalize_database_url(source_url))
    dst_engine = create_engine(normalize_database_url(target_url), pool_pre_ping=True)

    # Ensure target schema exists
    Base.metadata.create_all(bind=dst_engine)

    SrcSession = sessionmaker(bind=src_engine)
    DstSession = sessionmaker(bind=dst_engine)

    counts = {"trade_recommendations": 0, "execution_logs": 0, "placed_orders": 0}
    models = [TradeRecommendation, ExecutionLog, PlacedOrder]

    with SrcSession() as src, DstSession() as dst:
        for model in models:
            rows = src.query(model).order_by(model.id.asc()).all()
            cols = [c.name for c in model.__table__.columns]
            inserted = 0
            for row in rows:
                if dst.get(model, row.id) is not None:
                    continue
                payload = {col: getattr(row, col) for col in cols}
                dst.add(model(**payload))
                inserted += 1
            dst.commit()
            counts[model.__tablename__] = inserted
            logger.info("Migrated {} new row(s) into {}", inserted, model.__tablename__)

        # Advance PostgreSQL sequences past max imported primary keys
        if dst_engine.dialect.name == "postgresql":
            for model in models:
                tbl = model.__tablename__
                dst.execute(
                    text(
                        f"SELECT setval(pg_get_serial_sequence('{tbl}', 'id'), "
                        f"GREATEST(COALESCE((SELECT MAX(id) FROM {tbl}), 1), 1), "
                        f"(SELECT COUNT(*) > 0 FROM {tbl}))"
                    )
                )
            dst.commit()

    src_engine.dispose()
    dst_engine.dispose()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate local SQLite data to Railway PostgreSQL")
    parser.add_argument(
        "--source",
        default="sqlite:///./data/trading.db",
        help="Source SQLite URL (default: sqlite:///./data/trading.db)",
    )
    parser.add_argument(
        "--target",
        default=os.environ.get("DATABASE_URL", ""),
        help="Target PostgreSQL URL (defaults to DATABASE_URL env var)",
    )
    args = parser.parse_args()

    if not args.target or args.target.startswith("sqlite"):
        logger.error("Provide a PostgreSQL URL via --target or DATABASE_URL")
        sys.exit(1)

    counts = migrate_data(args.source, args.target)
    logger.info("Migration complete: {}", counts)


if __name__ == "__main__":
    main()
