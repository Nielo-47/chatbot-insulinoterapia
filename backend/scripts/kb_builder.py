import os
import json
import asyncio
import requests
import time
import sys
import argparse
from pathlib import Path
from datetime import datetime

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
RAW_DATA_DIR = require("RAW_DATA_DIR")

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)


async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
    # Use OpenRouter (OpenAI-compatible) for LLM completions
    model = require("KB_BUILD_LLM_MODEL")
    fallback_model = require("KB_BUILD_LLM_MODEL_FALLBACK")
    api_key = require("OPENROUTER_API_KEY")
    base_url = require("OPENROUTER_BASE_URL")

    extra_headers = {}
    referer = require("OPENROUTER_HTTP_REFERER")
    title = require("OPENROUTER_SITE_TITLE")
    if referer:
        extra_headers["HTTP-Referer"] = referer
    if title:
        extra_headers["X-Title"] = title

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
                extra_headers=extra_headers if extra_headers else None,
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
    # Allow building directly into remote storages by setting env vars:
    # KV_STORAGE (JsonKVStorage|RedisKVStorage|PGKVStorage|MongoKVStorage)
    # VECTOR_STORAGE (NanoVectorDBStorage|QdrantVectorDBStorage|MilvusVectorDBStorage|...)
    # GRAPH_STORAGE (NetworkXStorage|Neo4JStorage|PGGraphStorage|AGEStorage)
    kv_storage = require("KV_STORAGE")
    vector_storage = require("VECTOR_STORAGE")
    graph_storage = require("GRAPH_STORAGE")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        kv_storage=kv_storage,
        vector_storage=vector_storage,
        graph_storage=graph_storage,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=require_int("MAX_EMBED_TOKENS"),
            func=build_embedding_callable(
                primary=EmbeddingProviderConfig(
                    name="tei",
                    base_url=require("EMBEDDING_BINDING_HOST") + "/v1",
                    api_key=require("EMBEDDING_API_KEY"),
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
    print(f"RAG Storage configuration: KV={kv_storage}, VECTOR={vector_storage}, GRAPH={graph_storage}")

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
        print(f"Processing: {file_path}")
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

        # Insert with proper file_paths parameter for citation
        await rag.ainsert(input=text, file_paths=str(file_path))

        print(f"✓ Successfully processed: {file_path.name}")
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
    parser.add_argument(
        "--force-reprocess", action="store_true", help="Reprocess all documents regardless of modification time"
    )
    parser.add_argument(
        "--run-on-startup", action="store_true", help="Allow builder to run (bypasses RUN_KB_ON_STARTUP env check)"
    )
    args = parser.parse_args()

    # Respect env var RUN_KB_ON_STARTUP (default false) to avoid running KB builder on every container start
    # Can be overridden with --run-on-startup flag
    run_on_startup = args.run_on_startup or require("RUN_KB_ON_STARTUP").lower() == "true"
    if not run_on_startup:
        print("RUN_KB_ON_STARTUP is not 'true' and --run-on-startup not set. Exiting without running KB builder.")
        return

    # Force reprocess: use flag first, then env var
    force_reprocess = args.force_reprocess or require("RUN_KB_FORCE_REPROCESS").lower() == "true"
    processed_index_file = Path(WORKING_DIR) / "processed_index.json"
    processed_index = {}

    # Load processed index if available
    if processed_index_file.exists():
        try:
            processed_index = json.loads(processed_index_file.read_text())
        except Exception:
            print("Warning: Unable to read processed index; will treat all files as new.")
            processed_index = {}

    # Gather documents and compute which need processing
    documents = get_all_documents(RAW_DATA_DIR)

    # Remove index entries for deleted files
    existing_paths = {str(p) for p in documents}
    for key in list(processed_index.keys()):
        if key not in existing_paths:
            del processed_index[key]

    to_process = []
    skipped = 0
    for doc in documents:
        doc_str = str(doc)
        try:
            mtime = os.path.getmtime(doc_str)
        except Exception:
            to_process.append(doc)
            continue

        prev = processed_index.get(doc_str)
        if force_reprocess:
            to_process.append(doc)
        else:
            # Allow a small tolerance for mtime differences (filesystem precision)
            if prev and abs(float(prev.get("mtime", 0)) - float(mtime)) < 1.0:
                skipped += 1
            else:
                to_process.append(doc)

    print(f"\nFound {len(documents)} documents ({len(to_process)} to process, {skipped} skipped)")
    print(f"{'='*80}\n")

    # Limit processing for testing if --max-docs CLI flag is set
    max_docs = args.max_docs
    if max_docs and max_docs > 0 and len(to_process) > max_docs:
        print(f"⚠️  Limiting to first {max_docs} documents (out of {len(to_process)}) for testing")
        to_process = to_process[:max_docs]

    # Nothing to do -> exit early to avoid reprocessing or LLM calls
    if not to_process:
        print("No new or modified documents to process. Exiting.")
        return

    # Wait for core services to be reachable before initializing RAG
    service_wait_timeout = require_int("SERVICE_WAIT_TIMEOUT")

    for name, url in {
        "embeddings": require("EMBEDDING_BINDING_HOST") + "/v1",
        "qdrant": require("QDRANT_URL"),
        "neo4j": require("NEO4J_URI"),
        "redis": require("REDIS_URI"),
    }.items():
        if not url:
            continue
        print(f"Checking availability of {name} at {url}...")
        # Skip HTTP check for non-HTTP schemes (e.g., bolt://, redis://) — LightRAG will handle connectivity
        if url.startswith(("bolt://", "redis://")):
            print(f"  → Skipping HTTP check for {url}; will be verified by RAG initialization.")
            ok = True
        else:
            ok = await asyncio.get_event_loop().run_in_executor(None, wait_for_service, url, service_wait_timeout, 1)
        if not ok:
            print(
                f"Warning: Service '{name}' at {url} not reachable after {service_wait_timeout}s; continuing anyway."
            )
        else:
            print(f"Service '{name}' at {url} is reachable.")

    # Initialize RAG and process only the files that need it
    rag = await initialize_rag()

    successful = 0
    failed = 0

    for doc_path in to_process:
        success = await process_document(doc_path, rag)
        if success:
            successful += 1
            processed_index[str(doc_path)] = {
                "mtime": os.path.getmtime(str(doc_path)),
                "processed_at": datetime.utcnow().isoformat(),
            }
        else:
            failed += 1

        await asyncio.sleep(0.5)

    # Persist processed index
    try:
        processed_index_file.write_text(json.dumps(processed_index, indent=2))
    except Exception as e:
        print(f"Warning: Could not write processed index: {e}")

    # Summary
    print(f"\n{'='*80}")
    print(f"Processing Complete!")
    print(f"{'='*80}")
    print(f"✓ Successfully processed: {successful} documents")
    print(f"✗ Failed: {failed} documents")
    print(f"Total: {len(to_process)} documents")

    # Run a test query only if we processed something (to avoid unnecessary LLM usage)
    if successful > 0:
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
