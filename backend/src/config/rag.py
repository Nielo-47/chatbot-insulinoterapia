import os


RAG_WORKING_DIR = os.getenv("WORKING_DIR", "data/processed")

# Storage configuration - LightRAG reads these from environment at init time
KV_STORAGE = os.getenv("KV_STORAGE", "JsonKVStorage")
VECTOR_STORAGE = os.getenv("VECTOR_STORAGE", "NanoVectorDBStorage")
GRAPH_STORAGE = os.getenv("GRAPH_STORAGE", "NetworkXStorage")

LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("LLM_MODEL_NAME", "openai/gpt-5.2"))
LLM_FALLBACK_MODEL = os.getenv("LLM_MODEL_FALLBACK", "anthropic/claude-3-haiku")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_PRIMARY_RETRIES = int(os.getenv("LLM_PRIMARY_RETRIES", "1"))

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"))
EMBEDDING_FALLBACK_MODEL = os.getenv("EMBEDDING_FALLBACK_MODEL", EMBED_MODEL)
EMBEDDING_TIMEOUT_SECONDS = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30"))
EMBEDDING_PRIMARY_RETRIES = int(os.getenv("EMBEDDING_PRIMARY_RETRIES", "1"))
EMBEDDING_FALLBACK_RETRIES = int(os.getenv("EMBEDDING_FALLBACK_RETRIES", "0"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
MAX_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "8192"))

RAG_QUERY_TEMPERATURE = float(os.getenv("RAG_QUERY_TEMPERATURE", "0.1"))
RAG_QUERY_MAX_TOKENS = int(os.getenv("RAG_QUERY_MAX_TOKENS", "500"))
