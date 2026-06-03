import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lightrag.base import DocStatus

from backend.scripts.kb_builder import (
    RAW_DATA_DIR,
    get_all_documents,
    initialize_rag,
    process_document,
    wait_for_service,
)
from backend.src.config.env import require, require_int

MANIFEST_PATH = Path(os.getenv("RAW_DATA_MANIFEST_PATH", "data/processed/raw_data_manifest.json"))


def _compute_file_digest(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {}

    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[raw-sync] Warning: could not read manifest {manifest_path}: {exc}")
        return {}


def _save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    temp_path.replace(manifest_path)


async def _get_docs_grouped_by_file_path(rag) -> dict[str, list[str]]:
    docs_by_status = await rag.doc_status.get_docs_by_statuses(list(DocStatus))
    grouped: dict[str, list[str]] = {}

    for doc_id, status_doc in docs_by_status.items():
        file_path = getattr(status_doc, "file_path", None)
        if not file_path:
            continue
        grouped.setdefault(file_path, []).append(doc_id)

    return grouped


async def _delete_docs_for_file_path(rag, file_path: str) -> int:
    docs_by_path = await _get_docs_grouped_by_file_path(rag)
    doc_ids = docs_by_path.get(file_path, [])
    deleted = 0

    for doc_id in doc_ids:
        result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
        if getattr(result, "status", None) == "success":
            deleted += 1
        elif getattr(result, "status", None) == "not_found":
            deleted += 1
        else:
            print(f"[raw-sync] Warning: could not delete {doc_id} for {file_path}: {getattr(result, 'message', 'unknown error')}")

    return deleted


async def _cleanup_duplicate_versions(rag, file_path: str) -> int:
    docs_by_path = await _get_docs_grouped_by_file_path(rag)
    matching_doc_ids = docs_by_path.get(file_path, [])
    if len(matching_doc_ids) <= 1:
        return 0

    statuses = await rag.aget_docs_by_ids(matching_doc_ids)
    latest_doc_id = None
    latest_timestamp = None

    for doc_id, status_doc in statuses.items():
        updated_at = getattr(status_doc, "updated_at", None) or getattr(status_doc, "created_at", None)
        if not updated_at:
            continue
        try:
            timestamp = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if latest_timestamp is None or timestamp > latest_timestamp:
            latest_timestamp = timestamp
            latest_doc_id = doc_id

    removed = 0
    for doc_id in matching_doc_ids:
        if doc_id == latest_doc_id:
            continue
        result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
        if getattr(result, "status", None) in {"success", "not_found"}:
            removed += 1
        else:
            print(f"[raw-sync] Warning: could not remove stale version {doc_id} for {file_path}: {getattr(result, 'message', 'unknown error')}")

    return removed


async def sync_raw_data() -> None:
    rag = None
    try:
        documents = get_all_documents(RAW_DATA_DIR)
        current_manifest: dict[str, dict[str, Any]] = {}
        for document in documents:
            document_path = str(document)
            stat = document.stat()
            current_manifest[document_path] = {
                "sha256": _compute_file_digest(document),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }

        previous_manifest = _load_manifest(MANIFEST_PATH)
        previous_files = {
            path: details
            for path, details in previous_manifest.items()
            if not path.startswith("_")
        }

        files_to_delete = sorted(set(previous_files) - set(current_manifest))
        files_to_process = sorted(
            path
            for path, details in current_manifest.items()
            if previous_files.get(path, {}).get("sha256") != details["sha256"]
        )

        if not previous_manifest:
            print("[raw-sync] No existing manifest found; treating current raw data as the baseline.")

        print(f"[raw-sync] Found {len(documents)} raw document(s).")
        print(f"[raw-sync] {len(files_to_process)} new or changed file(s), {len(files_to_delete)} removed file(s).")

        service_wait_timeout = require_int("SERVICE_WAIT_TIMEOUT")
        embeddings_url = require("EMBEDDING_BINDING_HOST") + "/v1"
        print(f"[raw-sync] Checking embeddings service at {embeddings_url}...")
        ok = await asyncio.get_event_loop().run_in_executor(
            None, wait_for_service, embeddings_url, service_wait_timeout, 1
        )
        if not ok:
            print(
                f"[raw-sync] Warning: embeddings service at {embeddings_url} was not reachable after {service_wait_timeout}s; continuing."
            )

        rag = await initialize_rag()

        success_count = 0
        failure_count = 0
        deleted_count = 0
        duplicate_cleanup_count = 0
        updated_manifest = dict(previous_files)

        for file_path in files_to_delete:
            try:
                deleted = await _delete_docs_for_file_path(rag, file_path)
                if deleted:
                    deleted_count += deleted
                updated_manifest.pop(file_path, None)
            except Exception as exc:
                failure_count += 1
                print(f"[raw-sync] Warning: could not remove stale file {file_path}: {exc}")

        for file_path in files_to_process:
            try:
                document_path = Path(file_path)
                success = await process_document(document_path, rag)
                if success:
                    success_count += 1
                    duplicate_cleanup_count += await _cleanup_duplicate_versions(rag, file_path)
                    updated_manifest[file_path] = current_manifest[file_path]
                else:
                    failure_count += 1
            except Exception as exc:
                failure_count += 1
                print(f"[raw-sync] Warning: unexpected error processing {file_path}: {exc}")

            await asyncio.sleep(0.5)

        _save_manifest(MANIFEST_PATH, updated_manifest)

        print("[raw-sync] Sync complete.")
        print(f"[raw-sync] Successfully processed: {success_count}")
        print(f"[raw-sync] Failed to process: {failure_count}")
        print(f"[raw-sync] Removed stale records: {deleted_count}")
        print(f"[raw-sync] Cleaned up duplicate versions: {duplicate_cleanup_count}")
        print(f"[raw-sync] Manifest saved to: {MANIFEST_PATH}")
    except Exception as exc:
        print(f"[raw-sync] Fatal sync error: {exc}")
    finally:
        if rag is not None:
            try:
                await rag.finalize_storages()
            except Exception as exc:
                print(f"[raw-sync] Warning: could not finalize RAG storages: {exc}")


if __name__ == "__main__":
    asyncio.run(sync_raw_data())
