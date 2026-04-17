"""
Script to transfer locally-built knowledge graph data to remote production databases.
Run this from your local machine after building the KG locally.

Workflow:
1. Build KG locally: docker-compose up (with SKIP_KG_BUILD=0)
2. Deploy to Railway with empty databases
3. Transfer data: python populate_remote_kg.py --remote-host your-app.railway.app

Usage:
    python populate_remote_kg.py --remote-host your-railway-app.railway.app
"""

import argparse
import asyncio
import os
import json
from functools import partial
from dotenv import load_dotenv

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

load_dotenv()


async def transfer_knowledge_graph(
    local_working_dir,
    remote_host,
    redis_port=6379,
    qdrant_port=6333,
    neo4j_port=7687
):
    """
    Transfer locally-built KG data to remote databases.
    """
    
    print("\n" + "=" * 60)
    print("📦 Knowledge Graph Transfer Tool")
    print("=" * 60)
    print(f"📂 Source: {local_working_dir}")
    print(f"📍 Target: {remote_host}")
    print("=" * 60 + "\n")
    
    # Validate local data exists
    required_files = [
        "kv_store_full_docs.json",
        "graph_chunk_entity_relation.graphml"
    ]
    
    print("🔍 Checking local data...")
    for file in required_files:
        file_path = os.path.join(local_working_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"❌ Local KG data not found: {file_path}\n"
                f"Please build the KG locally first:\n"
                f"  SKIP_KG_BUILD=0 docker-compose up"
            )
        print(f"   ✓ Found: {file}")
    print()
    
    # Remote database connections
    redis_uri = f"redis://{remote_host}:{redis_port}/0"
    qdrant_url = f"http://{remote_host}:{qdrant_port}"
    neo4j_uri = f"bolt://{remote_host}:{neo4j_port}"
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    # Note: We don't need Ollama for transfer, but LightRAG requires it
    # We'll use a dummy config since we're only transferring data
    local_ollama = "http://localhost:11434"
    
    print("🔗 Connection Configuration:")
    print(f"   Local:     {local_working_dir}")
    print(f"   Redis:     {redis_uri}")
    print(f"   Qdrant:    {qdrant_url}")
    print(f"   Neo4j:     {neo4j_uri}")
    print()
    
    # Set environment for remote databases
    os.environ["NEO4J_URI"] = neo4j_uri
    os.environ["NEO4J_URL"] = neo4j_uri
    os.environ["NEO4J_HOST"] = remote_host
    os.environ["NEO4J_PORT"] = str(neo4j_port)
    os.environ["NEO4J_USERNAME"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = neo4j_password
    
    try:
        # Step 1: Initialize LOCAL RAG to read data
        print("📖 Step 1: Reading local knowledge graph data...")
        local_rag = LightRAG(
            working_dir=local_working_dir,
            kv_storage="JsonKVStorage",  # Read from local JSON files
            vector_storage="NanoVectorDBStorage",  # Read from local vector DB
            graph_storage="NetworkXStorage",  # Read from local GraphML
            llm_model_func=ollama_model_complete,
            llm_model_name="dummy",  # Not used for transfer
            llm_model_kwargs={"host": local_ollama, "options": {}, "timeout": 60},
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=lambda x: None  # Not used for transfer
            ),
        )
        
        await local_rag.initialize_storages()
        print("   ✓ Local data loaded\n")
        
        # Step 2: Initialize REMOTE RAG to write data
        print("📤 Step 2: Connecting to remote databases...")
        remote_rag = LightRAG(
            working_dir=local_working_dir,  # Read structure from local
            kv_storage="RedisKVStorage",
            vector_storage="QdrantVectorDBStorage",
            graph_storage="Neo4JStorage",
            llm_model_func=ollama_model_complete,
            llm_model_name="dummy",  # Not used for transfer
            llm_model_kwargs={"host": local_ollama, "options": {}, "timeout": 60},
            embedding_func=EmbeddingFunc(
                embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
                max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
                func=lambda x: None  # Not used for transfer
            ),
        )
        
        await remote_rag.initialize_storages()
        print("   ✓ Remote databases connected\n")
        
        # Step 3: Transfer KV Store data (Redis)
        print("📋 Step 3: Transferring key-value store data...")
        kv_namespaces = [
            "full_docs",
            "text_chunks", 
            "llm_response_cache",
            "full_entities",
            "full_relations",
            "entity_chunks",
            "relation_chunks"
        ]
        
        total_keys = 0
        for namespace in kv_namespaces:
            # Read from local JSON files
            json_file = os.path.join(local_working_dir, f"kv_store_{namespace}.json")
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Get remote storage for this namespace
                remote_storage = getattr(remote_rag, namespace, None)
                if remote_storage:
                    # Write each key-value pair to remote
                    for key, value in data.items():
                        await remote_storage.upsert({key: value})
                    
                    keys_transferred = len(data)
                    total_keys += keys_transferred
                    print(f"   ✓ {namespace}: {keys_transferred} keys")
        
        print(f"   ✓ Total keys transferred: {total_keys}\n")
        
        # Step 4: Transfer Vector Store data (Qdrant)
        print("🔍 Step 4: Transferring vector embeddings...")
        vector_namespaces = ["chunks", "entities", "relationships"]
        
        total_vectors = 0
        for namespace in vector_namespaces:
            json_file = os.path.join(local_working_dir, f"vdb_{namespace}.json")
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Get remote vector storage
                if hasattr(remote_rag, 'chunks_vdb' if namespace == 'chunks' else f'{namespace}_vdb'):
                    vdb = getattr(remote_rag, 'chunks_vdb' if namespace == 'chunks' else f'{namespace}_vdb')
                    
                    # Batch upsert vectors
                    if data:
                        await vdb.upsert(data)
                        vectors_transferred = len(data)
                        total_vectors += vectors_transferred
                        print(f"   ✓ {namespace}: {vectors_transferred} vectors")
        
        print(f"   ✓ Total vectors transferred: {total_vectors}\n")
        
        # Step 5: Transfer Graph data (Neo4j)
        print("🕸️  Step 5: Transferring knowledge graph structure...")
        graphml_file = os.path.join(local_working_dir, "graph_chunk_entity_relation.graphml")
        
        if os.path.exists(graphml_file):
            # Load GraphML
            import networkx as nx
            local_graph = nx.read_graphml(graphml_file)
            
            # Transfer nodes
            nodes = list(local_graph.nodes(data=True))
            print(f"   Transferring {len(nodes)} nodes...")
            for node_id, node_data in nodes:
                await remote_rag.graph_storage.upsert_node(node_id, node_data)
            print(f"   ✓ Nodes: {len(nodes)}")
            
            # Transfer edges
            edges = list(local_graph.edges(data=True))
            print(f"   Transferring {len(edges)} edges...")
            for source, target, edge_data in edges:
                await remote_rag.graph_storage.upsert_edge(
                    source, target, edge_data
                )
            print(f"   ✓ Edges: {len(edges)}\n")
        
        print("=" * 60)
        print("✅ Knowledge Graph Successfully Transferred!")
        print("=" * 60)
        print(f"\n📊 Transfer Summary:")
        print(f"   • KV Store Keys: {total_keys}")
        print(f"   • Vector Embeddings: {total_vectors}")
        print(f"   • Graph Nodes: {len(nodes) if 'nodes' in locals() else 0}")
        print(f"   • Graph Edges: {len(edges) if 'edges' in locals() else 0}")
        print(f"\n✨ Remote databases are now populated and ready!")
        print(f"   You can now use your Railway app at: https://{remote_host}")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ Transfer Failed")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Cleanup
        if 'local_rag' in locals():
            await local_rag.finalize_storages()
        if 'remote_rag' in locals():
            await remote_rag.finalize_storages()


def main():
    parser = argparse.ArgumentParser(
        description="Transfer locally-built knowledge graph to remote production databases"
    )
    parser.add_argument(
        "--local-dir",
        default="./data/processed",
        help="Path to local knowledge graph data (default: ./data/processed)"
    )
    parser.add_argument(
        "--remote-host",
        required=True,
        help="Remote host address (e.g., your-app.railway.app)"
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (default: 6379)"
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant port (default: 6333)"
    )
    parser.add_argument(
        "--neo4j-port",
        type=int,
        default=7687,
        help="Neo4j Bolt port (default: 7687)"
    )
    
    args = parser.parse_args()
    
    # Validate local directory exists
    if not os.path.exists(args.local_dir):
        print(f"❌ Error: Local data directory not found: {args.local_dir}")
        print(f"\nPlease build the knowledge graph locally first:")
        print(f"  SKIP_KG_BUILD=0 docker-compose up")
        return 1
    
    # Run the transfer
    asyncio.run(
        transfer_knowledge_graph(
            local_working_dir=args.local_dir,
            remote_host=args.remote_host,
            redis_port=args.redis_port,
            qdrant_port=args.qdrant_port,
            neo4j_port=args.neo4j_port,
        )
    )
    
    print("\n✨ Done!\n")
    return 0


if __name__ == "__main__":
    exit(main())