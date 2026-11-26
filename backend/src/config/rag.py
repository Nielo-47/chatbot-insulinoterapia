import os

from .env import get_float, get_str, require, require_float, require_int

RAG_WORKING_DIR = os.getenv("WORKING_DIR", "data/processed")

LLM_MODEL = require("LLM_MODEL")
LLM_FALLBACK_MODEL = require("LLM_MODEL_FALLBACK")
LLM_TIMEOUT_SECONDS = require_float("LLM_TIMEOUT_SECONDS")
LLM_PRIMARY_RETRIES = require_int("LLM_PRIMARY_RETRIES")

# Ingestion (knowledge-base builder) LLM configuration. Only used by
# backend/scripts/kb_builder.py; the runtime chatbot keeps using LLM_MODEL.
# Falls back to the runtime models when unset.
INGESTION_LLM_MODEL = get_str("INGESTION_LLM_MODEL", "") or LLM_MODEL
INGESTION_LLM_FALLBACK_MODEL = get_str("INGESTION_LLM_FALLBACK_MODEL", "") or LLM_FALLBACK_MODEL
INGESTION_LLM_TEMPERATURE = get_float("INGESTION_LLM_TEMPERATURE", 0.1)

EMBED_MODEL = require("EMBEDDING_MODEL")
EMBEDDING_FALLBACK_MODEL = require("EMBEDDING_FALLBACK_MODEL")
EMBEDDING_TIMEOUT_SECONDS = require_float("EMBEDDING_TIMEOUT_SECONDS")
EMBEDDING_PRIMARY_RETRIES = require_int("EMBEDDING_PRIMARY_RETRIES")
EMBEDDING_FALLBACK_RETRIES = require_int("EMBEDDING_FALLBACK_RETRIES")
EMBEDDING_DIM = require_int("EMBEDDING_DIM")
MAX_TOKENS = require_int("MAX_EMBED_TOKENS")

RAG_QUERY_TEMPERATURE = require_float("RAG_QUERY_TEMPERATURE")
RAG_QUERY_MAX_TOKENS = require_int("RAG_QUERY_MAX_TOKENS")