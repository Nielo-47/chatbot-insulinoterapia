#!/bin/bash
set -e

echo "[entrypoint] Starting Health Chatbot API"
echo "==========================================="

# Function to wait for a service to be ready using Python
wait_for_service() {
    local service=$1
    local host=$2
    local port=$3
    local max_attempts=30
    local attempt=1
    
    echo "[entrypoint] Waiting for $service at $host:$port..."
    
    while [ $attempt -le $max_attempts ]; do
        if python -c "import socket; sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); sock.settimeout(1); result = sock.connect_ex(('$host', $port)); sock.close(); exit(0 if result == 0 else 1)" 2>/dev/null; then
            echo "[entrypoint] ✓ $service is ready"
            return 0
        fi
        echo "[entrypoint]   Attempt $attempt/$max_attempts - $service not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "[entrypoint] ✗ $service failed to start after $max_attempts attempts"
    return 1
}

# Wait for database services
echo ""
echo "[entrypoint] Waiting for backend services..."
echo "==========================================="

wait_for_service "Redis" "redis" 6379 || exit 1
wait_for_service "Qdrant" "qdrant" 6333 || exit 1
wait_for_service "Neo4j" "neo4j" 7687 || exit 1

echo ""
echo "[entrypoint] ✓ All backend services are ready"
echo "==========================================="

# Start vLLM servers if enabled
if [ "${USE_VLLM}" = "1" ] || [ "${USE_VLLM}" = "true" ]; then
    echo ""
    echo "[entrypoint] Starting vLLM servers..."
    echo "==========================================="
    
    # Configuration
    LLM_MODEL=${LLM_MODEL_NAME:-"meta-llama/Llama-3.2-3B-Instruct"}
    EMBED_MODEL=${EMBEDDING_MODEL_NAME:-"sentence-transformers/all-MiniLM-L6-v2"}
    LLM_PORT=${VLLM_LLM_PORT:-8000}
    EMBED_PORT=${VLLM_EMBED_PORT:-8001}
    MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
    GPU_MEM_UTIL=${VLLM_GPU_MEMORY_UTILIZATION:-0.45}
    
    echo "[vllm] Starting LLM server..."
    echo "  Model: $LLM_MODEL"
    echo "  Port: $LLM_PORT"
    echo "  Max length: $MAX_MODEL_LEN"
    echo "  GPU memory: $GPU_MEM_UTIL"
    
    # Start LLM server in background
    vllm serve \
        --model "$LLM_MODEL" \
        --host 0.0.0.0 \
        --port "$LLM_PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
    2>&1 | tee /tmp/vllm_llm.log &
    
    LLM_PID=$!
    echo "[vllm] LLM server started with PID: $LLM_PID"
    
    # Wait for LLM server to be ready
    echo "[vllm] Waiting for LLM server to be ready..."
    max_attempts=60
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:$LLM_PORT/health >/dev/null 2>&1; then
            echo "[vllm] ✓ LLM server is ready!"
            break
        fi
        echo "[vllm]   Attempt $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        echo "[vllm] ✗ LLM server failed to start"
        cat /tmp/vllm_llm.log
        exit 1
    fi
    
    echo ""
    echo "[vllm] Starting Embedding server..."
    echo "  Model: $EMBED_MODEL"
    echo "  Port: $EMBED_PORT"
    echo "  GPU memory: $GPU_MEM_UTIL"
    
    # Start Embedding server in background
    vllm serve \
        --model "$EMBED_MODEL" \
        --host 0.0.0.0 \
        --port "$EMBED_PORT" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
    2>&1 | tee /tmp/vllm_embed.log &
    
    EMBED_PID=$!
    echo "[vllm] Embedding server started with PID: $EMBED_PID"
    
    # Wait for Embedding server to be ready
    echo "[vllm] Waiting for Embedding server to be ready..."
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:$EMBED_PORT/health >/dev/null 2>&1; then
            echo "[vllm] ✓ Embedding server is ready!"
            break
        fi
        echo "[vllm]   Attempt $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        echo "[vllm] ✗ Embedding server failed to start"
        cat /tmp/vllm_embed.log
        exit 1
    fi
    
    echo ""
    echo "[entrypoint] ✓ All vLLM servers are running"
    echo "  LLM: http://localhost:$LLM_PORT"
    echo "  Embeddings: http://localhost:$EMBED_PORT"
    
else
    echo ""
    echo "[entrypoint] 📝 vLLM is disabled (USE_VLLM not set)"
    echo "[entrypoint] 📝 Using external services"
fi

echo ""
echo "[entrypoint] 📝 Knowledge graph building is disabled"
echo "[entrypoint] 📝 Run 'python -m src.tools.kg_builder' manually to build KG"
echo ""

# Start the application
echo "[entrypoint] Starting Streamlit application on port ${PORT:-8080}"
echo "==========================================="
echo ""

exec streamlit run src/main.py \
    --server.port=${PORT:-8080} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.serverAddress=0.0.0.0 \
    --browser.gatherUsageStats=false