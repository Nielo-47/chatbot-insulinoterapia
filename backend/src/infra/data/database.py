# src/infrastructure/database.py
import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from langgraph.checkpoint.postgres import PostgresSaver

logger = logging.getLogger(__name__)


class DatabaseProvider:
    def __init__(self, db_url: str, pool_size: int = 5, max_overflow: int = 10):
        """Initialize the database engine and session factory."""
        self.db_url: str = db_url
        engine_kwargs: dict[str, Any] = {
            "pool_pre_ping": True,
            "pool_size": pool_size,
            "max_overflow": max_overflow,
        }

        self._engine = create_engine(db_url, **engine_kwargs)
        self._session_factory = sessionmaker(
            bind=self._engine, autocommit=False, autoflush=False, expire_on_commit=False
        )

        self.checkpointer: PostgresSaver = self._create_postgres_checkpointer()

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Provide a transactional scope around a series of operations."""
        session: Session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def check_connection(self) -> bool:
        """Verify the database is reachable."""
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error("Database connection failed: %s", e)
            return False

    def _create_postgres_checkpointer(self) -> Optional[Any]:
        """Create PostgresSaver checkpointer for LangGraph state persistence."""
        try:
            clean_url = self.db_url.replace("+psycopg", "")
            checkpointer = PostgresSaver.from_conn_string(clean_url)
            checkpointer.setup()

            return checkpointer
        except Exception as e:
            logger.warning("Could not create PostgresSaver checkpointer: %s", e)
            return None
