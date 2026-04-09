import logging

from backend.src.db.models import Base
from backend.src.db.session import check_database_connection, engine

logger = logging.getLogger(__name__)


def initialize_database() -> None:
    """Initialize persistent chat tables if they do not exist yet."""
    check_database_connection()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized")
