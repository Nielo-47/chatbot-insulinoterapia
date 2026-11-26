import os
import json
import asyncio
import re
import requests
import time
import sys
import argparse
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

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
    INGESTION_LLM_FALLBACK_MODEL,
    INGESTION_LLM_MODEL,
    INGESTION_LLM_TEMPERATURE,
)
from backend.src.infrastructure.rag.resilient_embeddings import (
    EmbeddingProviderConfig,
    build_embedding_callable,
)
from backend.src.config.env import get_int, require, require_int

WORKING_DIR = require("WORKING_DIR")
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)


async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
    # Use OpenRouter (OpenAI-compatible) for LLM completions.
    # The ingestion pipeline uses its own models (INGESTION_LLM_*), independent
    # from the runtime chatbot models (LLM_MODEL / LLM_FALLBACK_MODEL).
    model = INGESTION_LLM_MODEL
    fallback_model = INGESTION_LLM_FALLBACK_MODEL
    api_key = OPENROUTER_API_KEY
    base_url = OPENROUTER_BASE_URL

    # Rate limit and server error handling: wait-and-retry on 429/500 responses
    max_rate_retries = get_int("LLM_RATE_LIMIT_RETRIES", 3)
    sleep_on_rate = get_int("LLM_RATE_LIMIT_SLEEP", 5)

    max_server_retries = get_int("LLM_SERVER_ERROR_RETRIES", 2)
    sleep_on_server = get_int("LLM_SERVER_ERROR_SLEEP", 3)

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
                temperature=INGESTION_LLM_TEMPERATURE,
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


# Tesseract language packs used for OCR. Tesseract uses 3-letter ISO 639-2
# codes: "por" (Portuguese) and "eng" (English).
OCR_LANGUAGES = ["por", "eng"]

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}


def get_all_documents(root_dir):
    """
    Recursively find all supported documents (PDF, DOCX, PNG, JPG, JPEG) in the directory.
    Excludes Zone.Identifier files.
    """
    documents = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = Path(root) / file
            # Skip Zone.Identifier files
            if ":Zone.Identifier" in file or file.endswith(".Identifier"):
                continue
            # Check if file has supported extension
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                documents.append(file_path)

    return sorted(documents)


def ocr_pdf_page(file_path, page_num, ocr_languages):
    """Render a single PDF page to an image and OCR it. Returns stripped text or ''."""
    from pdf2image import convert_from_path
    from unstructured_pytesseract import pytesseract

    try:
        images = convert_from_path(
            str(file_path), first_page=page_num, last_page=page_num, dpi=200
        )
    except Exception as e:
        print(f"  ⚠️  Could not render page {page_num} of {file_path.name}: {e}")
        return ""
    if not images:
        return ""
    try:
        return pytesseract.image_to_string(
            images[0], lang="+".join(ocr_languages)
        ).strip()
    except Exception as e:
        print(f"  ⚠️  OCR failed for page {page_num} of {file_path.name}: {e}")
        return ""


def ocr_image_file(file_path, ocr_languages):
    """OCR a standalone image file. Returns stripped text or ''."""
    from PIL import Image
    from unstructured_pytesseract import pytesseract

    try:
        with Image.open(str(file_path)) as img:
            return pytesseract.image_to_string(
                img, lang="+".join(ocr_languages)
            ).strip()
    except Exception as e:
        print(f"  ⚠️  OCR failed for {file_path.name}: {e}")
        return ""


def classify_and_extract_pdf(file_path, ocr_languages):
    """
    Classify a PDF and extract text per page. Pages without a selectable text
    layer are rendered to an image and OCR'd.

    Returns (classification, page_contents, page_has_text):
      - classification: "full" | "partial" | "scanned" | "unreadable"
      - page_contents: list of "[PAGE n]\n\n{text}" blocks (text or OCR)
      - page_has_text: {page_num: bool} whether the page had a text layer
    """
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(file_path))
    except Exception as e:
        print(f"  ⚠️  Could not open {file_path.name}: {e}")
        return "unreadable", [], {}

    page_contents = []
    page_has_text = {}
    text_pages = 0

    for page_num, page in enumerate(reader.pages, start=1):
        text = ""
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        page_has_text[page_num] = bool(text)
        if text:
            text_pages += 1
            page_contents.append(f"[PAGE {page_num}]\n\n{text}")
        else:
            ocr_text = ocr_pdf_page(file_path, page_num, ocr_languages)
            if ocr_text:
                page_contents.append(f"[PAGE {page_num}]\n\n{ocr_text}")

    total_pages = len(reader.pages)
    if total_pages == 0 or text_pages == 0:
        classification = "scanned" if page_contents else "unreadable"
    elif text_pages == total_pages:
        classification = "full"
    else:
        classification = "partial"

    return classification, page_contents, page_has_text


async def process_document(file_path, rag):
    """
    Process a single document and insert it into the RAG system.

    Returns a record dict: {"status", "type", "chars", "error"} where status is
    "ingested" | "empty" | "failed".
    """
    try:
        print(f"\n{'='*80}")
        print(f"Reading file: {file_path}")
        print(f"{'='*80}")

        ext = file_path.suffix.lower()
        page_contents = []
        classification = "unknown"

        if ext == ".pdf":
            classification, page_contents, _ = classify_and_extract_pdf(
                file_path, OCR_LANGUAGES
            )
        elif ext in {".png", ".jpg", ".jpeg"}:
            classification = "image"
            ocr_text = ocr_image_file(file_path, OCR_LANGUAGES)
            if ocr_text:
                page_contents.append(ocr_text)
        else:  # .docx
            classification = "docx"
            loader = UnstructuredLoader(str(file_path), languages=["pt", "en"])
            docs = loader.load()
            for doc in docs:
                page_num = doc.metadata.get("page_number", 1)
                content = doc.page_content.strip()
                if content:
                    marked_content = f"[PAGE {page_num}]\n\n{content}"
                    page_contents.append(marked_content)

        if not page_contents:
            print(f"⚠️  No content extracted from {file_path}")
            return {
                "status": "empty",
                "type": classification,
                "chars": 0,
                "error": "No content extracted",
            }

        text = "\n\n".join(page_contents)
        chars = sum(len(block) for block in page_contents)

        # Insert with proper file_paths parameter for citation.
        # LightRAG will automatically check its internal kv_store_doc_status.json
        # to see if this exact content has already been processed.
        await rag.ainsert(input=text, file_paths=str(file_path))

        print(f"✓ File parsed and passed to LightRAG: {file_path.name} ({classification}, {chars} chars)")
        return {
            "status": "ingested",
            "type": classification,
            "chars": chars,
            "error": None,
        }

    except Exception as e:
        print(f"✗ Error processing {file_path}: {str(e)}")
        return {
            "status": "failed",
            "type": "unknown",
            "chars": 0,
            "error": str(e),
        }


def run_dry_run(documents):
    """Classify all documents (PDFs only) without initializing RAG or calling any service."""
    print("\nClassification (dry-run):\n")
    counts = {}
    for doc in documents:
        ext = doc.suffix.lower()
        if ext == ".pdf":
            classification, _, page_has_text = classify_and_extract_pdf(doc, OCR_LANGUAGES)
            pages_with_text = sum(1 for v in page_has_text.values() if v)
            pages_total = len(page_has_text)
            print(f"{classification:>10} | {doc.name} | {pages_with_text}/{pages_total} pages with text")
        elif ext in {".png", ".jpg", ".jpeg"}:
            classification = "image"
            print(f"{classification:>10} | {doc.name}")
        else:
            classification = "docx"
            print(f"{classification:>10} | {doc.name}")
        counts[classification] = counts.get(classification, 0) + 1

    print("\n" + "=" * 40)
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


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
    parser.add_argument("--dry-run", action="store_true", help="Classify documents without initializing RAG or calling external services")
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

    if args.dry_run:
        run_dry_run(documents)
        return

    # Wait for core services to be reachable before initializing RAG
    service_wait_timeout = get_int("SERVICE_WAIT_TIMEOUT", 60)

    embeddings_url = OPENROUTER_BASE_URL + "/v1"
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

    results = []
    for doc_path in documents:
        results.append(await process_document(doc_path, rag))
        await asyncio.sleep(0.5)

    # Summary
    ingested = sum(1 for r in results if r["status"] == "ingested")
    empty = sum(1 for r in results if r["status"] == "empty")
    failed = sum(1 for r in results if r["status"] == "failed")

    print(f"\n{'='*80}")
    print(f"Script Execution Complete!")
    print(f"{'='*80}")
    print(f"✓ Read successfully: {ingested} documents")
    print(f"⚠️  Empty (no content): {empty} documents")
    print(f"✗ Failed to read: {failed} documents")
    print(f"Total attempted: {len(documents)} documents")
    print(
        "\nNote: LightRAG handles actual duplication internally. It will skip graph extraction for files it has already processed in 'kv_store_doc_status.json'."
    )

    # Write per-file report next to the processed index
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "raw_data_dir": RAW_DATA_DIR,
        "results": [
            {"file": str(doc_path), "status": record["status"], "type": record["type"], "chars": record["chars"], "error": record["error"]}
            for doc_path, record in zip(documents, results)
        ],
    }
    report_path = os.path.join(WORKING_DIR, "kb_builder_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport written to {report_path}")

    if empty or failed:
        raise SystemExit(1)

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
