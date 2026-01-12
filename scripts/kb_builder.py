import os
import asyncio
import nest_asyncio
from functools import partial
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc, setup_logger
from langchain_unstructured import UnstructuredLoader

nest_asyncio.apply()

WORKING_DIR = "data/processed/"
RAW_DATA_DIR = "data/raw/"

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="vllm-token-dummy",
        base_url="http://localhost:8000/v1",
        **kwargs,
    )

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        default_llm_timeout=600,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=partial(
                openai_embed.func,
                model="BAAI/bge-m3",
                base_url="http://localhost:8001/v1",
                api_key="vllm-token-dummy",
            ),
        ),
    )
    await rag.initialize_storages()
    return rag

def get_all_documents(root_dir):
    """
    Recursively find all PDF and DOCX files in the directory.
    Excludes Zone.Identifier files.
    """
    supported_extensions = {'.pdf', '.docx'}
    documents = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = Path(root) / file
            # Skip Zone.Identifier files
            if ':Zone.Identifier' in file or file.endswith('.Identifier'):
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
        
        loader = UnstructuredLoader(str(file_path))
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
        await rag.ainsert(
            input=text,
            file_paths=str(file_path)
        )
        
        print(f"✓ Successfully processed: {file_path.name}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {file_path}: {str(e)}")
        return False

async def main():
    rag = await initialize_rag()
    
    # Get all documents
    documents = get_all_documents(RAW_DATA_DIR)
    
    print(f"\nFound {len(documents)} documents to process")
    print(f"{'='*80}\n")
    
    # Track statistics
    successful = 0
    failed = 0
    
    # Process each document
    for doc_path in documents:
        success = await process_document(doc_path, rag)
        if success:
            successful += 1
        else:
            failed += 1
        
        # Optional: Add a small delay to avoid overwhelming the system
        await asyncio.sleep(0.5)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Processing Complete!")
    print(f"{'='*80}")
    print(f"✓ Successfully processed: {successful} documents")
    print(f"✗ Failed: {failed} documents")
    print(f"Total: {len(documents)} documents")
    
    # Example query
    print(f"\n{'='*80}")
    print("Testing query...")
    print(f"{'='*80}\n")
    
    try:
        result = await rag.aquery(
            "Quais são os tipos de insulina disponíveis?", 
            param=QueryParam(mode="hybrid")
        )
        print(f"Query result:\n{result}")
    except Exception as e:
        print(f"Query failed: {e}")
    
    await rag.finalize_storages()

if __name__ == "__main__":
    setup_logger(logger_name="kb_builder", level="DEBUG")
    asyncio.run(main())