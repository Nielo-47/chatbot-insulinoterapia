import os


def _normalize_embed_host(raw_host: str) -> str:
    if raw_host.endswith("/v1"):
        return raw_host[:-3]
    return raw_host.rstrip("/")


RAG_WORKING_DIR = os.getenv("WORKING_DIR", "data/processed")

# Normalize embedding host (TEI) so callers can safely append /v1
_raw_embed = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:8000")
EMBED_HOST = _normalize_embed_host(_raw_embed)

LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("LLM_MODEL_NAME", "openai/gpt-5.2"))
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
MAX_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "8192"))
