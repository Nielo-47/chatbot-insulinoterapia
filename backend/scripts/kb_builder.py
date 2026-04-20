import os
import json
import asyncio
import requests
import time
import sys
import argparse
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Patch LightRAG's openai_embed BEFORE any LightRAG imports.
# OpenRouter occasionally returns response.data = None which causes
# "TypeError: 'NoneType' object is not iterable". We retry 5× and,
# on final failure, return zero vectors so the document can continue.
# ──────────────────────────────────────────────────────────────
try:
    import lightrag.llm.openai as _lightrag_openai

    _original_openai_embed = _lightrag_openai.openai_embed

    async def _patched_openai_embed(*args, **kwargs):
        for attempt in range(5):
            try:
                return await _original_openai_embed(*args, **kwargs)
            except TypeError as e:
                if "'NoneType' object is not iterable" in str(e):
                    if attempt < 4:
                        await asyncio.sleep(2**attempt)
                        continue
                    # Final attempt: return zero vectors so processing can continue
                    texts_arg = args[1] if len(args) > 1 else kwargs.get("texts", [])
                    texts = texts_arg if isinstance(texts_arg, list) else [texts_arg]
                    dim = kwargs.get("embedding_dim") or (args[2] if len(args) > 2 else 1024)
                    import numpy as np

                    return np.zeros((len(texts), int(dim)), dtype=np.float32)
                raise

    _lightrag_openai.openai_embed = _patched_openai_embed
    print("[kilo] Patched lightrag.llm.openai.openai_embed (5 retries, zero-vector fallback)")
except Exception as exc:
    print(f"[kilo] Warning: could not patch lightrag openai_embed: {exc}")

import nest_asyncio

nest_asyncio.apply()

# Now import LightRAG and other modules
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, setup_logger
from langchain_unstructured import UnstructuredLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.src.config.infrastructure import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from backend.src.config.rag import (
    EMBED_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_FALLBACK_MODEL,
    EMBEDDING_FALLBACK_RETRIES,
    EMBEDDING_PRIMARY_RETRIES,
    EMBEDDING_TIMEOUT_SECONDS,
)
from backend.src.infrastructure.rag.resilient_embeddings import (
    EmbeddingProviderConfig,
    build_embedding_callable,
)
from backend.src.config.env import require, require_int, require_float

WORKING_DIR = require("WORKING_DIR")
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)


async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
    # Use OpenRouter (OpenAI-compatible) for LLM completions
    model = require("KB_BUILD_LLM_MODEL")
    fallback_model = require("KB_BUILD_LLM_MODEL_FALLBACK")
    api_key = require("OPENROUTER_API_KEY")
    base_url = require("OPENROUTER_BASE_URL")

    # Rate limit and server error handling: wait-and-retry on 429/500 responses
    max_rate_retries = require_int("LLM_RATE_LIMIT_RETRIES")
    sleep_on_rate = require_int("LLM_RATE_LIMIT_SLEEP")

    max_server_retries = require_int("LLM_SERVER_ERROR_RETRIES")
    sleep_on_server = require_int("LLM_SERVER_ERROR_SLEEP")

    attempt_rate = 0
    attempt_server = 0
    use_fallback = False

    while True:
        try:
            return await openai_complete_if_cache(
                fallback_model if use_fallback else model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                temperature=require_float("LLM_TEMPERATURE"),
                **kwargs,
            )
        except Exception as e:
            msg = str(e)
            is_rate = False
            is_server = False
            try:
                import openai as _openai

                # Helper to safely get a class or None
                def safe_get_class(obj, attr):
                    val = getattr(obj, attr, None)
                    return val if isinstance(val, type) else None

                RateLimitError = safe_get_class(_openai, "RateLimitError")
                InternalServerError = safe_get_class(_openai, "InternalServerError")
                APIError = safe_get_class(_openai, "APIError")
                error_mod = getattr(_openai, "error", None)
                if error_mod:
                    if RateLimitError is None:
                        RateLimitError = safe_get_class(error_mod, "RateLimitError")
                    if InternalServerError is None:
                        InternalServerError = safe_get_class(error_mod, "InternalServerError")
                    if APIError is None:
                        APIError = safe_get_class(error_mod, "APIError")

                if RateLimitError is not None:
                    is_rate = isinstance(e, RateLimitError)
                if InternalServerError is not None:
                    is_server = isinstance(e, InternalServerError)
                if APIError is not None:
                    is_server = is_server or isinstance(e, APIError)
            except Exception:
                # Fallback to string matching if class check fails
                low = msg.lower()
                is_rate = "rate limit" in low or "rate_limit" in low or "rate limit exceeded" in low
                is_server = (
                    "internal server error" in low or "500" in low or "cloudflare" in low or "<!doctype html>" in low
                )

            if is_rate:
                attempt_rate += 1
                if attempt_rate > max_rate_retries:
                    if use_fallback:
                        print(f"OpenAI rate limit hit on fallback model and retries exhausted. Raising error.")
                        raise
                    print(f"OpenAI rate limit hit; switching to fallback model...")
                    use_fallback = True
                    attempt_rate = 0
                    continue

                print(
                    f"OpenAI rate limit hit; sleeping {sleep_on_rate}s before retrying (attempt {attempt_rate}/{max_rate_retries})..."
                )
                await asyncio.sleep(sleep_on_rate)
                continue

            if is_server:
                attempt_server += 1
                if attempt_server > max_server_retries:
                    if use_fallback:
                        print(f"OpenAI internal/server error on fallback model and retries exhausted. Raising error.")
                        raise
                    print(f"OpenAI internal/server error; switching to fallback model...")
                    use_fallback = True
                    attempt_server = 0
                    continue

                print(
                    f"OpenAI internal/server error; sleeping {sleep_on_server}s before retrying (attempt {attempt_server}/{max_server_retries})..."
                )
                await asyncio.sleep(sleep_on_server)
                continue

            # Not a retriable error -> re-raise
            raise


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=require_int("MAX_EMBED_TOKENS"),
            func=build_embedding_callable(
                primary=EmbeddingProviderConfig(
                    name="openrouter",
                    base_url=OPENROUTER_BASE_URL,
                    api_key=OPENROUTER_API_KEY,
                    model=EMBED_MODEL,
                ),
                fallback=EmbeddingProviderConfig(
                    name="openrouter",
                    base_url=OPENROUTER_BASE_URL,
                    api_key=OPENROUTER_API_KEY,
                    model=EMBEDDING_FALLBACK_MODEL,
                ),
                embedding_dim=EMBEDDING_DIM,
                timeout_seconds=EMBEDDING_TIMEOUT_SECONDS,
                primary_retries=EMBEDDING_PRIMARY_RETRIES,
                fallback_retries=EMBEDDING_FALLBACK_RETRIES,
            ),
        ),
    )

    # Initialize storages (this will create collections / databases if supported)
    await rag.initialize_storages()

    # Helpful logs about where data will be stored
    print("RAG Storage: using LightRAG defaults (local files)")

    return rag


def get_all_documents(root_dir):
    """
    Recursively find all PDF and DOCX files in the directory.
    Excludes Zone.Identifier files.
    """
    supported_extensions = {".pdf", ".docx"}
    documents = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = Path(root) / file
            # Skip Zone.Identifier files
            if ":Zone.Identifier" in file or file.endswith(".Identifier"):
                continue
            # Check if file has supported extension
            if file_path.suffix.lower() in supported_extensions:
                documents.append(file_path)

    return sorted(documents)


async def process_document(file_path, rag):
    """
    Process a single document and insert it into the RAG system.
    """
    try:
        print(f"\n{'='*80}")
        print(f"Reading file: {file_path}")
        print(f"{'='*80}")

        loader = UnstructuredLoader(str(file_path), languages=["pt", "en"])
        docs = loader.load()

        page_contents = []
        for doc in docs:
            page_num = doc.metadata.get("page_number", 1)
            content = doc.page_content.strip()
            if content:
                marked_content = f"[PAGE {page_num}]\n\n{content}"
                page_contents.append(marked_content)

        if not page_contents:
            print(f"⚠️  No content extracted from {file_path}")
            return False

        text = "\n\n".join(page_contents)

        # Insert with proper file_paths parameter for citation.
        # LightRAG will automatically check its internal kv_store_doc_status.json
        # to see if this exact content has already been processed.
        await rag.ainsert(input=text, file_paths=str(file_path))

        print(f"✓ File parsed and passed to LightRAG: {file_path.name}")
        return True

    except Exception as e:
        print(f"✗ Error processing {file_path}: {str(e)}")
        return False


def wait_for_service(url, timeout=60, interval=1):
    import urllib.parse

    parsed = urllib.parse.urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            # Try a simple GET to the service base to ensure TCP connection
            requests.get(base, timeout=2)
            return True
        except Exception:
            print(f"Waiting for embeddings service at {base} ...")
            time.sleep(interval)
    return False


async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Build the knowledge base by processing documents")
    parser.add_argument("--max-docs", type=int, default=0, help="Limit processing to first N documents (0 = no limit)")
    args = parser.parse_args()

    # Gather documents
    documents = get_all_documents(RAW_DATA_DIR)

    print(f"\nFound {len(documents)} total documents in directory.")
    print(f"{'='*80}\n")

    # Limit processing for testing if --max-docs CLI flag is set
    max_docs = args.max_docs
    if max_docs and max_docs > 0 and len(documents) > max_docs:
        print(f"⚠️  Limiting to first {max_docs} documents (out of {len(documents)}) for testing")
        documents = documents[:max_docs]

    if not documents:
        print("No documents found. Exiting.")
        return

    # Wait for core services to be reachable before initializing RAG
    service_wait_timeout = require_int("SERVICE_WAIT_TIMEOUT")

    embeddings_url = require("EMBEDDING_BINDING_HOST") + "/v1"
    print(f"Checking availability of embeddings at {embeddings_url}...")
    ok = await asyncio.get_event_loop().run_in_executor(
        None, wait_for_service, embeddings_url, service_wait_timeout, 1
    )
    if not ok:
        print(
            f"Warning: Embeddings service at {embeddings_url} not reachable after {service_wait_timeout}s; continuing anyway."
        )
    else:
        print(f"Embeddings service at {embeddings_url} is reachable.")

    # Initialize RAG
    rag = await initialize_rag()

    successful = 0
    failed = 0

    for doc_path in documents:
        success = await process_document(doc_path, rag)
        if success:
            successful += 1
        else:
            failed += 1

        await asyncio.sleep(0.5)

    # Summary
    print(f"\n{'='*80}")
    print(f"Script Execution Complete!")
    print(f"{'='*80}")
    print(f"✓ Read successfully: {successful} documents")
    print(f"✗ Failed to read: {failed} documents")
    print(f"Total attempted: {len(documents)} documents")
    print(
        "\nNote: LightRAG handles actual duplication internally. It will skip graph extraction for files it has already processed in 'kv_store_doc_status.json'."
    )

    # Run a test query
    print(f"\n{'='*80}")
    print("Testing query...")
    print(f"{'='*80}\n")
    try:
        result = await rag.aquery(
            "Quais são os tipos de insulina disponíveis?",
            param=QueryParam(mode="hybrid"),
        )
        print(f"Query result:\n{result}")
    except Exception as e:
        print(f"Query failed: {e}")

    await rag.finalize_storages()


if __name__ == "__main__":
    setup_logger(logger_name="kb_builder", level="DEBUG")
    asyncio.run(main())
