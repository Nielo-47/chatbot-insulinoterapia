import os

from .env import require, require_int, get_int, get_str, get_bool

OPENROUTER_API_KEY = require("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = require("OPENROUTER_BASE_URL")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_SITE_TITLE = os.getenv("OPENROUTER_SITE_TITLE", "")

DATABASE_URL = require("DATABASE_URL")
DB_POOL_SIZE = require_int("DB_POOL_SIZE")
DB_MAX_OVERFLOW = require_int("DB_MAX_OVERFLOW")

CHAT_CACHE_REDIS_URL = require("CHAT_CACHE_REDIS_URL")
CHAT_CACHE_TTL_SECONDS = require_int("CHAT_CACHE_TTL_SECONDS")
CHAT_CACHE_KEY_PREFIX = require("CHAT_CACHE_KEY_PREFIX")

# API docs exposure. Disabled by default (fail closed) to avoid leaking the API
# schema for reconnaissance; opt in explicitly for development.
DOCS_ENABLED = get_bool("DOCS_ENABLED", False)

# LLM semantic cache (Redis). GLOBAL and keyed by prompt hash, NOT user-scoped,
# so a cached response containing PII could be served to another user. Disabled
# by default for that reason; enable only if responses can never contain
# user-specific data.
SEMANTIC_CACHE_ENABLED = get_bool("SEMANTIC_CACHE_ENABLED", False)