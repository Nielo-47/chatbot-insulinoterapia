import os
import uuid
from typing import Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import sessionmaker


ALLOWED_TEST_HOSTS = {"localhost", "127.0.0.1", "postgres"}


def require_test_database_url() -> str:
    db_url = os.getenv("TEST_DATABASE_URL")
    if not db_url:
        raise RuntimeError("TEST_DATABASE_URL must be set explicitly for integration tests.")

    parsed = make_url(db_url)
    if not parsed.drivername.startswith("postgresql"):
        raise RuntimeError("Integration tests only support PostgreSQL URLs.")

    db_name = (parsed.database or "").lower()
    if not db_name.endswith("_test"):
        raise RuntimeError("Integration tests must run against a database ending with '_test'.")

    host = (parsed.host or "").lower()
    if host not in ALLOWED_TEST_HOSTS:
        raise RuntimeError(
            f"Integration tests only allow local hosts: {', '.join(sorted(ALLOWED_TEST_HOSTS))}."
        )

    if any(token in db_name for token in ("prod", "production", "main")):
        raise RuntimeError("Integration tests refuse production-like database names.")

    return db_url


def create_isolated_test_engine() -> Tuple[Engine, str]:
    db_url = require_test_database_url()
    engine = create_engine(db_url, pool_pre_ping=True)
    schema_name = f"test_schema_{uuid.uuid4().hex[:8]}"

    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'))

    return engine, schema_name


def bind_session_to_schema(engine: Engine, schema_name: str) -> Engine:
    schema_engine = engine.execution_options(schema_translate_map={None: schema_name})

    import backend.src.db.session as db_session

    db_session.engine = schema_engine
    db_session.SessionLocal = sessionmaker(
        bind=schema_engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
    return schema_engine


def drop_isolated_schema(engine: Engine, schema_name: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))