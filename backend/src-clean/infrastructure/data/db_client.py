import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from backend.src.config.infrastructure import DATABASE_URL, DB_MAX_OVERFLOW, DB_POOL_SIZE
from backend.src.infrastructure.data.models import Base

logger = logging.getLogger(__name__)


def _build_engine():
    db_url = DATABASE_URL
    engine_kwargs: dict[str, Any] = {"pool_pre_ping": True}

    if db_url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    else:
        engine_kwargs["pool_size"] = DB_POOL_SIZE
        engine_kwargs["max_overflow"] = DB_MAX_OVERFLOW

    return create_engine(db_url, **engine_kwargs)


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


@contextmanager
def get_db_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database_connection() -> bool:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True


def initialize_database() -> None:
    """Initialize persistent chat tables if they do not exist yet."""
    check_database_connection()
    Base.metadata.create_all(bind=engine)
    _ensure_message_sources_column()
    _ensure_summary_column()
    logger.info("Database tables initialized")


def _ensure_message_sources_column() -> None:
    """Backfill schema changes for environments that already have existing tables."""
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("messages")}
    if "sources_json" in columns:
        return

    logger.warning("messages.sources_json column missing; applying compatibility ALTER TABLE")
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE messages ADD COLUMN sources_json TEXT"))
    logger.info("messages.sources_json column added")


def _ensure_summary_column() -> None:
    """Backfill schema changes for conversations.summary column."""
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("conversations")}
    if "summary" in columns:
        return

    logger.warning("conversations.summary column missing; applying compatibility ALTER TABLE")
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE conversations ADD COLUMN summary TEXT"))
    logger.info("conversations.summary column added")


def create_postgres_checkpointer() -> Optional[Any]:
    """Create PostgresSaver checkpointer for LangGraph state persistence."""
    from backend.src.config.conversation import CHECKPOINTER_ENABLED

    if not CHECKPOINTER_ENABLED:
        return None

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        import psycopg
    except ImportError:
        return None

    try:
        db_url = DATABASE_URL.replace("+psycopg", "")
        conn = psycopg.connect(db_url)
        return PostgresSaver(conn)
    except Exception as e:
        logger.warning("Could not create PostgresSaver checkpointer: %s", e)
        return None
