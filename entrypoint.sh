#!/bin/bash
set -e

# --- Helper Functions ---

# TCP Check (Safe for Redis/Neo4j)
wait_for_tcp() {
    echo "[entrypoint] Waiting for TCP $1 at $2:$3..."
    until python -c "import socket; s = socket.socket(); s.settimeout(1); exit(s.connect_ex(('$2', $3)))" 2>/dev/null; do
        sleep 2
    done
    echo "[entrypoint] ✓ $1 is ready"
}

# HTTP Check (For Qdrant/vLLM)
wait_for_http() {
    echo "[entrypoint] Waiting for HTTP $1 at $2:$3..."
    until curl -s "http://$2:$3/health" >/dev/null 2>&1; do
        sleep 5
    done
    echo "[entrypoint] ✓ $1 is ready"
}

# --- Execution ---

echo "[entrypoint] Starting Health Chatbot API"
echo "==========================================="

# 1. Check Databases (TCP)
wait_for_tcp "Redis" "redis" 6379
wait_for_tcp "Neo4j" "neo4j" 7687

# 2. Check Vector DB (HTTP)
wait_for_http "Qdrant" "qdrant" 6333

# 3. Check vLLM Servers (Assuming they are already running)
echo "[entrypoint] Verifying vLLM availability..."
wait_for_http "vLLM-LLM" "host.docker.internal" 8000
wait_for_http "vLLM-Embed" "host.docker.internal" 8001

echo "[entrypoint] ✓ All services found. Launching Streamlit..."
exec streamlit run src/main.py --server.port=8080 --server.address=0.0.0.0