import logging
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from backend.src.config.infrastructure import DATABASE_URL, DB_MAX_OVERFLOW, DB_POOL_SIZE

logger = logging.getLogger(__name__)


def _ensure_psycopg_dialect(db_url: str) -> str:
    """Map bare Postgres URLs to the installed psycopg3 dialect.

    SQLAlchemy resolves a plain ``postgresql://`` (or ``postgres://``) scheme
    to the psycopg2 dialect, which is not installed — only psycopg3
    (``psycopg[binary]``) ships in this image. URLs already carrying a driver
    (e.g. ``postgresql+psycopg://``) are left untouched.
    """
    for plain, driver in (
        ("postgresql://", "postgresql+psycopg://"),
        ("postgres://", "postgres+psycopg://"),
    ):
        if db_url.startswith(plain):
            return driver + db_url[len(plain):]
    return db_url


def _prepare_db_url(db_url: str) -> str:
    """Normalize a raw DATABASE_URL for this stack.

    Ensures the psycopg3 dialect is selected and that a non-local Postgres URL
    requests an encrypted connection. Supabase pooler connections require
    sslmode; the URL in .env may already include it. Local/test hosts keep
    their existing behavior (plaintext).
    """
    db_url = _ensure_psycopg_dialect(db_url)
    url = make_url(db_url)
    if not url.drivername.startswith("postgres"):
        return db_url
    host = (url.host or "").lower()
    if host in ("localhost", "127.0.0.1", "::1"):
        return db_url
    query = dict(url.query)
    query.setdefault("sslmode", "require")
    return url.set(query=query).render_as_string(hide_password=False)


def _build_engine():
    db_url = _prepare_db_url(DATABASE_URL)
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
    """Verify the database is reachable.

    DDL lives in scripts/supabase_migration.sql and is applied from the
    Supabase dashboard, never at runtime, so startup only validates the
    connection.
    """
    check_database_connection()
    logger.info("Database connection verified")


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
        db_url = _prepare_db_url(DATABASE_URL).replace("+psycopg", "")
        conn = psycopg.connect(db_url)
        return PostgresSaver(conn)
    except Exception as e:
        logger.warning("Could not create PostgresSaver checkpointer: %s", e)
        return None


def purge_user_checkpoint_threads(user_id: uuid.UUID) -> bool:
    """Delete LangGraph checkpointer state for a user's thread (user_{user_id}).

    Called during account deletion: the PostgresSaver tables (checkpoints,
    checkpoint_blobs, checkpoint_writes) are not covered by the user FK
    cascade, so without this the persisted thread state would outlive the
    account. Best-effort: returns False (logged) if the purge fails.
    """
    from backend.src.config.conversation import CHECKPOINTER_ENABLED

    if not CHECKPOINTER_ENABLED:
        return True

    try:
        import psycopg
    except ImportError:
        return True

    thread_id = f"user_{user_id}"
    try:
        db_url = _prepare_db_url(DATABASE_URL).replace("+psycopg", "")
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                    cur.execute(f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,))
            conn.commit()
        logger.info("Purged checkpointer thread %r", thread_id)
        return True
    except Exception as e:
        logger.warning("Could not purge checkpointer thread %r: %s", thread_id, e)
        return False
