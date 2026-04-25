from sqlalchemy import create_engine

from backend.src.core.config.infrastructure import DATABASE_URL


class Database:
    def __init__(self) -> None:
        self.engine = create_engine(DATABASE_URL)
