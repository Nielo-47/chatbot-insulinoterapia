import os

from .env import require, require_int, get_int, get_str, get_bool

OPENROUTER_API_KEY = require("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = require("OPENROUTER_BASE_URL")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_SITE_TITLE = os.getenv("OPENROUTER_SITE_TITLE", "")

CHAT_CACHE_REDIS_URL = require("CHAT_CACHE_REDIS_URL")
CHAT_CACHE_TTL_SECONDS = require_int("CHAT_CACHE_TTL_SECONDS")
CHAT_CACHE_KEY_PREFIX = require("CHAT_CACHE_KEY_PREFIX")

# LLM semantic cache (Redis). GLOBAL and keyed by prompt hash, NOT user-scoped,
# so a cached response containing PII could be served to another user. Disabled
# by default for that reason; enable only if responses can never contain
# user-specific data.
SEMANTIC_CACHE_ENABLED = get_bool("SEMANTIC_CACHE_ENABLED", False)