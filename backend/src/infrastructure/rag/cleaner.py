import logging
import re
from typing import Any

PAGE_PATTERN = re.compile(r'\[PAGE (\d+)\]')


def clean_source_path(file_path: str) -> str:
    if not file_path:
        return file_path

    prefixes_to_remove = ["data/raw/", "data\\raw\\", "./data/raw/"]
    for prefix in prefixes_to_remove:
        if file_path.startswith(prefix):
            return file_path[len(prefix) :]

    return file_path


def extract_page_from_text(text: str) -> int | None:
    """Extract first page number from text containing [PAGE N] marker."""
    match = PAGE_PATTERN.search(text)
    return int(match.group(1)) if match else None


def extract_sources(rag_data: Any) -> tuple[list[dict], int]:
    """Extract structured source info with page and excerpt from RAG response.
    
    Returns list of dicts: {path, page, excerpt}
    """
    sources: list[dict] = []
    seen_chunk_ids = set()

    if not rag_data or not isinstance(rag_data, dict):
        return sources, 0

    try:
        if rag_data.get("status") != "success":
            return sources, 0

        data_section = rag_data.get("data", {})
        if not data_section:
            return sources, 0

        # Build reference_id -> file_path mapping from references
        reference_to_file: dict[str, str] = {}
        references = data_section.get("references", [])
        if isinstance(references, list):
            for ref in references:
                if isinstance(ref, dict):
                    ref_id = ref.get("reference_id")
                    file_path = ref.get("file_path")
                    if ref_id and file_path:
                        reference_to_file[str(ref_id)] = str(file_path)

        # Process chunks to get content + page (deduplicate by chunk_id)
        chunks = data_section.get("chunks", [])
        if isinstance(chunks, list):
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                chunk_id = chunk.get("chunk_id")
                if not chunk_id or chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)

                ref_id = chunk.get("reference_id")
                file_path = reference_to_file.get(str(ref_id)) if ref_id else None
                if not file_path:
                    continue

                clean_path = clean_source_path(str(file_path))
                chunk_content = chunk.get("content", "")
                page_num = extract_page_from_text(chunk_content)

                # Create excerpt: first 200 chars without marker
                excerpt = chunk_content
                if page_num is not None:
                    excerpt = PAGE_PATTERN.sub('', chunk_content, count=1).strip()
                excerpt = excerpt[:200] + ("..." if len(excerpt) > 200 else "")

                sources.append({
                    "path": clean_path,
                    "page": page_num,
                    "excerpt": excerpt,
                })

        return sources, len(sources)

    except Exception as e:
        logging.warning(f"Failed to extract sources from RAG data: {e}")
        return sources, 0
