import unittest
import os
import uuid

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from backend.src.infrastructure.data.models import Base


class DatabaseSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        db_url = os.getenv("TEST_DATABASE_URL")
        if not db_url:
            raise RuntimeError(
                "TEST_DATABASE_URL must be set explicitly for schema tests. "
                "Do not reuse DATABASE_URL."
            )

        parsed = make_url(db_url)
        if not parsed.drivername.startswith("postgresql"):
            raise RuntimeError("Schema tests only support PostgreSQL URLs.")

        db_name = (parsed.database or "").lower()
        if not db_name.endswith("_test"):
            raise RuntimeError(
                "Refusing to run schema tests on a non-test database. "
                "Database name must end with '_test'."
            )

        host = (parsed.host or "").lower()
        if host not in {"localhost", "127.0.0.1", "postgres"}:
            raise RuntimeError(
                "Refusing to run schema tests on non-local host. "
                "Allowed hosts: localhost, 127.0.0.1, postgres."
            )

        if any(token in db_name for token in ("prod", "production", "main")):
            raise RuntimeError("Refusing to run schema tests on production-like database names.")

        cls.engine = create_engine(db_url, pool_pre_ping=True)
        cls.test_schema = f"test_schema_{uuid.uuid4().hex[:8]}"
        try:
            with cls.engine.begin() as conn:
                conn.execute(text("SELECT 1"))
                conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{cls.test_schema}"'))
        except Exception as exc:
            raise RuntimeError(
                "Could not connect to PostgreSQL test database. "
                "Start the container first with 'docker compose up -d postgres'."
            ) from exc

    @classmethod
    def _schema_connection(cls, conn):
        return conn.execution_options(schema_translate_map={None: cls.test_schema})

    def setUp(self):
        with self.engine.begin() as conn:
            schema_conn = self._schema_connection(conn)
            Base.metadata.drop_all(bind=schema_conn)
            Base.metadata.create_all(bind=schema_conn)

    def tearDown(self):
        with self.engine.begin() as conn:
            schema_conn = self._schema_connection(conn)
            Base.metadata.drop_all(bind=schema_conn)

    @classmethod
    def tearDownClass(cls) -> None:
        with cls.engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{cls.test_schema}" CASCADE'))
        cls.engine.dispose()

    def test_create_all_creates_expected_tables(self) -> None:
        inspector = inspect(self.engine)
        table_names = set(inspector.get_table_names(schema=self.test_schema))

        self.assertIn("users", table_names)
        self.assertIn("conversations", table_names)
        self.assertIn("messages", table_names)

    def test_drop_all_removes_tables(self) -> None:
        with self.engine.begin() as conn:
            schema_conn = self._schema_connection(conn)
            Base.metadata.drop_all(bind=schema_conn)

        inspector = inspect(self.engine)
        table_names = set(inspector.get_table_names(schema=self.test_schema))

        self.assertNotIn("users", table_names)
        self.assertNotIn("conversations", table_names)
        self.assertNotIn("messages", table_names)

    def test_messages_table_has_expected_index(self) -> None:
        inspector = inspect(self.engine)
        indexes = inspector.get_indexes("messages", schema=self.test_schema)
        index_names = {index["name"] for index in indexes}

        self.assertIn("ix_messages_conversation_created", index_names)


if __name__ == "__main__":
    unittest.main()
