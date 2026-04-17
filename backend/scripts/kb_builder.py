import os
import json
import asyncio
import nest_asyncio
import requests
import time
import sys
from pathlib import Path
from datetime import datetime
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, setup_logger
from langchain_unstructured import UnstructuredLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
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

nest_asyncio.apply()

WORKING_DIR = "data/processed/"
RAW_DATA_DIR = "data/raw/"

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)


async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
    # Use OpenRouter (OpenAI-compatible) for LLM completions
    model = os.getenv("LLM_MODEL", os.getenv("OPENROUTER_MODEL"))
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    extra_headers = {}
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "")
    title = os.getenv("OPENROUTER_SITE_TITLE", "")
    if referer:
        extra_headers["HTTP-Referer"] = referer
    if title:
        extra_headers["X-Title"] = title

    # Rate limit and server error handling: wait-and-retry on 429/500 responses
    max_rate_retries = int(os.getenv("LLM_RATE_LIMIT_RETRIES", "1"))
    sleep_on_rate = int(os.getenv("LLM_RATE_LIMIT_SLEEP", "60"))  # seconds

    max_server_retries = int(os.getenv("LLM_SERVER_ERROR_RETRIES", "1"))
    sleep_on_server = int(os.getenv("LLM_SERVER_ERROR_SLEEP", "60"))  # seconds

    attempt_rate = 0
    attempt_server = 0

    while True:
        try:
            return await openai_complete_if_cache(
                model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                extra_headers=extra_headers if extra_headers else None,
                temperature=float(os.getenv("LLM_TEMPERATURE", 0.1)),
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
                    print(f"OpenAI rate limit hit and retries exhausted (attempts={attempt_rate}). Raising error.")
                    raise

                print(
                    f"OpenAI rate limit hit; sleeping {sleep_on_rate}s before retrying (attempt {attempt_rate}/{max_rate_retries})..."
                )
                await asyncio.sleep(sleep_on_rate)
                continue

            if is_server:
                attempt_server += 1
                if attempt_server > max_server_retries:
                    print(
                        f"OpenAI internal/server error and retries exhausted (attempts={attempt_server}). Raising error."
                    )
                    raise

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
    kv_storage = os.getenv("KV_STORAGE", "JsonKVStorage")
    vector_storage = os.getenv("VECTOR_STORAGE", "NanoVectorDBStorage")
    graph_storage = os.getenv("GRAPH_STORAGE", "NetworkXStorage")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        kv_storage=kv_storage,
        vector_storage=vector_storage,
        graph_storage=graph_storage,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=build_embedding_callable(
                primary=EmbeddingProviderConfig(
                    name="tei",
                    base_url=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:8000") + "/v1",
                    api_key=os.getenv("EMBEDDING_API_KEY", ""),
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

        pages = []
        for doc in docs:
            t = doc.page_content.strip()
            if t:
                pages.append(t)

        if not pages:
            print(f"⚠️  No content extracted from {file_path}")
            return False

        text = "\n\n".join(pages)

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
    # Respect env var RUN_KB_ON_STARTUP (default false) to avoid running KB builder on every container start
    run_on_startup = os.getenv("RUN_KB_ON_STARTUP", "false").lower() == "true"
    if not run_on_startup:
        print("RUN_KB_ON_STARTUP is not 'true'. Exiting without running KB builder.")
        return

    # Determine whether there are new or changed documents to process first (avoid heavy work if none)
    force_reprocess = os.getenv("RUN_KB_FORCE_REPROCESS", "false").lower() == "true"
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

    # Nothing to do -> exit early to avoid reprocessing or LLM calls
    if not to_process:
        print("No new or modified documents to process. Exiting.")
        return

    # Wait for core services to be reachable before initializing RAG
    service_wait_timeout = int(os.getenv("SERVICE_WAIT_TIMEOUT", "120"))

    for name, url in {
        "embeddings": os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:8000/v1"),
        "qdrant": os.getenv("QDRANT_URL", "http://qdrant:6333"),
        "neo4j": os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
        "redis": os.getenv("REDIS_URI", "redis://redis:6379/0"),
    }.items():
        if not url:
            continue
        print(f"Checking availability of {name} at {url}...")
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
