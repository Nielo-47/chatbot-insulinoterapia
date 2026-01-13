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
# Custom extraction prompt for medical/diabetes domain
ENTITY_EXTRACTION_PROMPT = """---Task---
Extract entities and relationships from the input text about diabetes and insulin therapy.

---Instructions---
1. **Extract ONLY entities that are explicitly mentioned in the text**
2. **DO NOT invent or hallucinate entities from other domains (sports, finance, technology, etc.)**
3. **Focus on medical and healthcare entities only**
4. Output entities first, then relationships
5. End with <|COMPLETE|>

---Valid Entity Types for Medical Domain---
- Person: Healthcare professionals, patients (only if explicitly named)
- Organization: Medical societies, health ministries, hospitals
- Location: Body parts, anatomical locations (Abdômen, Braços, Coxas)
- Concept: Medical conditions (Diabetes, Hipoglicemia, Lipodistrofia)
- Method: Medical procedures, techniques (Aplicação, Rodízio)
- Artifact: Medical equipment (Seringa, Agulha, Caneta, Glicosímetro)
- Content: Medications (Insulina NPH, Insulina Regular)

---Example (Medical Domain)---
Input Text:
```
A insulina NPH deve ser aplicada com seringa no abdômen. O paciente deve fazer rodízio dos locais de aplicação para evitar lipodistrofia. A Sociedade Brasileira de Diabetes recomenda o uso de agulhas de 4mm.
```

Output:
entity<|#|>Insulina NPH<|#|>Content<|#|>Insulina NPH is a type of insulin medication used in diabetes treatment.
entity<|#|>Seringa<|#|>Artifact<|#|>Seringa (syringe) is a medical device used to inject insulin.
entity<|#|>Abdômen<|#|>Location<|#|>Abdômen (abdomen) is a body region where insulin can be injected.
entity<|#|>Lipodistrofia<|#|>Concept<|#|>Lipodistrofia is a complication from repeated insulin injections in the same location.
entity<|#|>Sociedade Brasileira de Diabetes<|#|>Organization<|#|>Sociedade Brasileira de Diabetes is a Brazilian medical organization that provides diabetes treatment guidelines.
entity<|#|>Rodízio<|#|>Method<|#|>Rodízio is the practice of rotating injection sites to prevent complications.
entity<|#|>Agulha<|#|>Artifact<|#|>Agulha (needle) is a medical device component used for insulin injection.
relation<|#|>Insulina NPH<|#|>Seringa<|#|>medical procedure, administration<|#|>Insulina NPH is administered using a seringa.
relation<|#|>Seringa<|#|>Abdômen<|#|>injection site, application<|#|>Seringa is used to inject insulin in the abdômen.
relation<|#|>Rodízio<|#|>Lipodistrofia<|#|>complication prevention, treatment method<|#|>Rodízio of injection sites helps prevent lipodistrofia.
relation<|#|>Sociedade Brasileira de Diabetes<|#|>Agulha<|#|>medical recommendation, guideline<|#|>Sociedade Brasileira de Diabetes recommends specific needle sizes.
<|COMPLETE|>

---Data to be Processed---
<Entity_types>
[Person,Organization,Location,Concept,Method,Content,Artifact]

<Input Text>
```
{input_text}
```

<Output>
"""

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # Enhanced system prompt to reduce hallucinations
    enhanced_system_prompt = """You are a medical information extraction specialist focused on diabetes and insulin therapy.

CRITICAL RULES:
1. Extract ONLY entities explicitly mentioned in the provided text
2. DO NOT invent, fabricate, or hallucinate ANY information
3. DO NOT add entities from sports, finance, technology, or other unrelated domains
4. Focus ONLY on medical and healthcare-related entities
5. If you're uncertain about an entity, DO NOT include it
6. Use entity names exactly as they appear in the source text
7. Maintain Portuguese terminology for medical terms when present in the source

STRICTLY FORBIDDEN - Never extract these types of entities:
- Athletes, sports events, championships, competitions
- Financial markets, stocks, indexes, companies
- Technology companies, products (unless medical devices)
- Random person names not mentioned in the text
- Geographic locations unrelated to the medical content

ALLOWED - Only extract these types of entities:
- Medical conditions and diseases
- Insulin types and medications
- Medical equipment and devices
- Healthcare procedures and techniques
- Body parts and anatomical locations
- Healthcare organizations and professionals
- Medical guidelines and recommendations
"""
    
    if system_prompt:
        enhanced_system_prompt = f"{enhanced_system_prompt}\n\n{system_prompt}"
    
    return await openai_complete_if_cache(
        "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        prompt,
        system_prompt=enhanced_system_prompt,
        history_messages=history_messages,
        api_key="vllm-token-dummy",
        base_url="http://localhost:8000/v1",
        temperature=0.1,  # Lower temperature for more deterministic output
        **kwargs,
    )

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        summary_max_tokens=2048,
        max_entity_tokens=3000,
        max_relation_tokens=4000,
        max_total_tokens=15000,
        chunk_token_size=600,
        chunk_overlap_token_size=150, 
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