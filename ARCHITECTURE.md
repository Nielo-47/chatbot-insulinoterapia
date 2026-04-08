# Diabetes Chatbot - Backend/UI Separation Architecture

## Overview

The application has been refactored into a **microservices architecture** with separate backend and UI containers:

### Architecture Components

```
┌─────────────┐
│  Browser    │
└──────┬──────┘
       │ HTTP :8501
       ▼
┌─────────────┐
│  UI (ui)    │  Streamlit Frontend
│  Port: 8501 │
└──────┬──────┘
       │ HTTP :8000 (internal)
       ▼
┌─────────────┐
│Backend (api)│  FastAPI + RAG
│  Port: 8000 │
└──────┬──────┘
       │
   ┌───┴────────────────┬──────────┬─────────┐
   ▼                    ▼          ▼         ▼
┌────────┐  ┌────────┐  ┌───────┐  ┌──────┐
│ Redis  │  │ Qdrant │  │ Neo4j │  │ TEI  │
└────────┘  └────────┘  └───────┘  └──────┘
```

## Services

### 1. **Backend** (`backend` service)
- **Technology**: FastAPI + uvicorn
- **Port**: 8000 (internal: `backend:8000`)
- **Dockerfile**: `dockerfile.backend`
- **Requirements**: `requirements.backend.txt`
- **Responsibilities**:
  - RAG (Retrieval-Augmented Generation) engine
  - LightRAG integration
  - Session management
  - Database connections (Redis, Qdrant, Neo4j)
  - Document processing and embeddings

### 2. **UI** (`ui` service)
- **Technology**: Streamlit
- **Port**: 8501 (exposed to host)
- **Dockerfile**: `dockerfile.ui`
- **Requirements**: `requirements.ui.txt`
- **Responsibilities**:
  - User interface
  - HTTP requests to backend API
  - Message history display
  - Session state management

### 3. **Supporting Services**
- **Redis**: Key-value storage
- **Qdrant**: Vector database
- **Neo4j**: Graph database
- **TEI**: Text embeddings inference (GPU-enabled)

## API Endpoints

### Backend API (`http://backend:8000`)

#### `POST /query`
Query the chatbot with a question.

**Request:**
```json
{
  "query": "Como aplicar insulina?",
  "session_id": "optional-session-uuid"
}
```

**Response:**
```json
{
  "response": "Para aplicar insulina...",
  "sources": ["fonte 1", "fonte 2"],
  "source_count": 2,
  "summarized": false,
  "session_id": "session-uuid"
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "Chatbot API is running"
}
```

#### `DELETE /session/{session_id}`
Clear conversation history for a session.

**Response:**
```json
{
  "message": "Session {session_id} cleared successfully"
}
```

## Running the Application

### Development Mode (Separated Services)

```bash
# Start all services with the new architecture
docker-compose up --build

# Access the UI
# http://localhost:8501

# Access the backend API docs
# http://localhost:8000/docs
```

### Legacy Mode (Monolithic)

If you need to run the old monolithic version:

```bash
docker-compose --profile legacy up app
```

## Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Backend API port (exposed to host)
BACKEND_PORT=8000

# UI port (exposed to host)
UI_PORT=8501

# Request timeout for UI -> Backend communication
REQUEST_TIMEOUT=60

# OpenRouter API credentials (required)
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# LLM Model
LLM_MODEL=openai/gpt-4

# Neo4j password
NEO4J_PASSWORD=password
```

## Benefits of This Architecture

1. **Scalability**: Backend and UI can scale independently
2. **Separation of Concerns**: Clear boundary between API logic and presentation
3. **Development**: Can develop/test backend and UI separately
4. **Deployment**: Can deploy UI and backend on different servers/regions
5. **Resource Efficiency**: UI container is lightweight (no heavy ML dependencies)
6. **API Reusability**: Backend API can be consumed by other clients (mobile, etc.)

## Health Checks

- **Backend**: `curl http://localhost:8000/health`
- **UI**: `curl http://localhost:8501/_stcore/health`

## Logs

```bash
# View backend logs
docker-compose logs -f backend

# View UI logs
docker-compose logs -f ui
```

## Development Tips

### Testing Backend Independently

```bash
# Start only backend and dependencies
docker-compose up redis qdrant neo4j tei backend

# Test API
curl http://localhost:8000/health

# Access API docs
# http://localhost:8000/docs
```

### Testing UI Independently

```bash
# Ensure backend is running first
docker-compose up backend

# In another terminal
docker-compose up ui
```

## Migration Notes

### From Monolithic to Microservices

The previous `app` service has been split into:
- `backend`: Handles all RAG/chatbot logic
- `ui`: Streamlit interface only

**Key Changes:**
1. `src/main.py`: Now uses HTTP requests instead of multiprocessing
2. `src/api.py`: New FastAPI backend service
3. `dockerfile.backend`: Backend-specific Dockerfile
4. `dockerfile.ui`: UI-specific Dockerfile
5. `requirements.backend.txt`: Backend dependencies (heavy ML libraries)
6. `requirements.ui.txt`: UI dependencies (lightweight)

The old monolithic service is still available via `--profile legacy` for backward compatibility.
