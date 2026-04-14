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

        chunks = data_section.get("chunks", [])
        if isinstance(chunks, list):
            for chunk in chunks:
                if isinstance(chunk, dict):
                    file_path = chunk.get("file_path")
                    if file_path:
                        clean_path = clean_source_path(file_path)
                        if clean_path and clean_path not in seen:
                            sources.append(clean_path)
                            seen.add(clean_path)

        references = data_section.get("references", [])
        if isinstance(references, list):
            for ref in references:
                if isinstance(ref, dict):
                    file_path = ref.get("file_path")
                    if file_path:
                        clean_path = clean_source_path(file_path)
                        if clean_path and clean_path not in seen:
                            sources.append(clean_path)
                            seen.add(clean_path)

        entities = data_section.get("entities", [])
        if isinstance(entities, list):
            for entity in entities:
                if isinstance(entity, dict):
                    file_path = entity.get("file_path")
                    if file_path:
                        clean_path = clean_source_path(file_path)
                        if clean_path and clean_path not in seen:
                            sources.append(clean_path)
                            seen.add(clean_path)

        relationships = data_section.get("relationships", [])
        if isinstance(relationships, list):
            for rel in relationships:
                if isinstance(rel, dict):
                    file_path = rel.get("file_path")
                    if file_path:
                        clean_path = clean_source_path(file_path)
                        if clean_path and clean_path not in seen:
                            sources.append(clean_path)
                            seen.add(clean_path)

        return sources, len(sources)

    except Exception as e:
        logging.warning(f"Failed to extract sources from RAG data: {e}")
        return sources, 0
