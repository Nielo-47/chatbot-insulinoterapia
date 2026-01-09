import os
import asyncio
import nest_asyncio
from functools import partial
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, setup_logger
from langchain_unstructured import UnstructuredLoader

nest_asyncio.apply()


WORKING_DIR = "data/processed/"
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)


async def initialize_rag():
    # Fix 1: Use a slightly smaller context or ensure Ollama can handle it
    # Fix 2: Limit concurrency to avoid the 500 NaN error

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="hf.co/MaziyarPanahi/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M",
        llm_model_max_async=1,
        summary_max_tokens=2048,
        max_entity_tokens=3000,
        max_relation_tokens=4000,
        max_total_tokens=15000,
        chunk_token_size=600,
        chunk_overlap_token_size=150,
        default_llm_timeout=600,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {
                "temperature": 0,
            },
        },
        # Fix 3: Simplify embedding call to use the internal handler
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=partial(
                ollama_embed.func,
                embed_model="paraphrase-multilingual:latest",
                host="http://localhost:11434",
            ),
        ),
    )

    await rag.initialize_storages()
    return rag


async def main():
    rag = await initialize_rag()

    # If the database is corrupted with NaN, we need to clear it
    # Check if files exist and you might want to wipe data/processed/ if errors persist

    loader = UnstructuredLoader("data/raw/DOC_Completo.docx")
    docs = loader.load()

    pages = []
    for doc in docs:
        t = doc.page_content.strip()
        if t:
            pages.append(t)

    text = "\n\n".join(pages)

    # Insert document
    try:
        await rag.ainsert(text)
    except Exception as e:
        print(f"Insertion failed, but continuing to query existing index: {e}")

    # Your queries...
    print(await rag.aquery("What is acne?", param=QueryParam(mode="hybrid")))

    await rag.finalize_storages()


if __name__ == "__main__":
    setup_logger(logger_name="kb_builder", level="DEBUG")
    asyncio.run(main())
