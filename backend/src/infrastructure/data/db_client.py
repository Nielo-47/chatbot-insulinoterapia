import logging
from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import create_engine, text
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
    logger.info("Database tables initialized")
