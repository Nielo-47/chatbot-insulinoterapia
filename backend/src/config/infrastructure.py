import os


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_SITE_TITLE = os.getenv("OPENROUTER_SITE_TITLE", "")

# Persistent chat storage (PostgreSQL)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://chatbot:chatbot@localhost:5432/chatbot",
)
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))

# Conversation cache (Redis)
CHAT_CACHE_REDIS_URL = os.getenv("CHAT_CACHE_REDIS_URL", os.getenv("REDIS_URI", "redis://localhost:6379/1"))
CHAT_CACHE_TTL_SECONDS = int(os.getenv("CHAT_CACHE_TTL_SECONDS", "300"))
CHAT_CACHE_KEY_PREFIX = os.getenv("CHAT_CACHE_KEY_PREFIX", "chat:conv")
