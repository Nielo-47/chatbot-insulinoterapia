import logging
from typing import Any


def clean_source_path(file_path: str) -> str:
    if not file_path:
        return file_path

    prefixes_to_remove = ["data/raw/", "data\\raw\\", "./data/raw/"]
    for prefix in prefixes_to_remove:
        if file_path.startswith(prefix):
            return file_path[len(prefix) :]

    return file_path


def extract_sources(rag_data: Any) -> tuple[list[str], int]:
    """Extract source file paths from LightRAG query response.
    
    LightRAG response structure:
    {
        "status": "success",
        "data": {
            "chunks": [{"chunk_id": "...", "reference_id": "1"}],
            "references": [{"reference_id": "1", "file_path": "doc.pdf"}],
            "entities": [{"name": "...", "source_id": "chunk-..."}],
            "relationships": [...]
        }
    }
    
    Sources can come from:
    1. references[].file_path - direct source paths
    2. chunks[].reference_id -> references[].file_path - mapped via chunks
    """
    sources: list[str] = []
    seen = set()

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

        # Extract sources from references directly
        for ref in references if isinstance(references, list) else []:
            if isinstance(ref, dict):
                file_path = ref.get("file_path")
                if file_path:
                    clean_path = clean_source_path(str(file_path))
                    if clean_path and clean_path not in seen:
                        sources.append(clean_path)
                        seen.add(clean_path)

        # Extract sources via chunks -> reference_id -> file_path
        chunks = data_section.get("chunks", [])
        if isinstance(chunks, list):
            for chunk in chunks:
                if isinstance(chunk, dict):
                    ref_id = chunk.get("reference_id")
                    if ref_id:
                        ref_key = str(ref_id)
                        if ref_key in reference_to_file:
                            file_path = reference_to_file[ref_key]
                            clean_path = clean_source_path(file_path)
                            if clean_path and clean_path not in seen:
                                sources.append(clean_path)
                                seen.add(clean_path)

        # Extract sources via entities -> source_id (chunk_id) -> chunk -> reference_id -> file_path
        # Build chunk_id -> reference_id mapping
        chunk_to_reference: dict[str, str] = {}
        for chunk in chunks if isinstance(chunks, list) else []:
            if isinstance(chunk, dict):
                chunk_id = chunk.get("chunk_id")
                ref_id = chunk.get("reference_id")
                if chunk_id and ref_id:
                    chunk_to_reference[str(chunk_id)] = str(ref_id)

        entities = data_section.get("entities", [])
        if isinstance(entities, list):
            for entity in entities:
                if isinstance(entity, dict):
                    source_id = entity.get("source_id")
                    if source_id:
                        chunk_key = str(source_id)
                        if chunk_key in chunk_to_reference:
                            ref_key = chunk_to_reference[chunk_key]
                            if ref_key in reference_to_file:
                                file_path = reference_to_file[ref_key]
                                clean_path = clean_source_path(file_path)
                                if clean_path and clean_path not in seen:
                                    sources.append(clean_path)
                                    seen.add(clean_path)

        # Also check relationships for source_id -> file_path
        relationships = data_section.get("relationships", [])
        if isinstance(relationships, list):
            for rel in relationships:
                if isinstance(rel, dict):
                    source_id = rel.get("source_id")
                    if source_id:
                        chunk_key = str(source_id)
                        if chunk_key in chunk_to_reference:
                            ref_key = chunk_to_reference[chunk_key]
                            if ref_key in reference_to_file:
                                file_path = reference_to_file[ref_key]
                                clean_path = clean_source_path(file_path)
                                if clean_path and clean_path not in seen:
                                    sources.append(clean_path)
                                    seen.add(clean_path)

        return sources, len(sources)

    except Exception as e:
        logging.warning(f"Failed to extract sources from RAG data: {e}")
        return sources, 0
