# Diabetes Chatbot - React + FastAPI Architecture

## Overview

The application follows a microservices architecture with a React SPA frontend and a FastAPI backend.

```text
Browser
  |
  | HTTP :3000
  v
UI Container (React + Vite build served by Nginx)
  |
  | HTTP :8000 (internal Docker network)
  v
Backend Container (FastAPI + RAG)
  |
  +--> Redis
  +--> Qdrant
  +--> Neo4j
  +--> TEI (embeddings)
```

## Services

### 1. backend
- Technology: FastAPI + uvicorn
- Port: 8000
- Dockerfile: dockerfile.backend
- Responsibilities:
  - Query processing with RAG
  - Session memory (in-process)
  - Integration with Redis/Qdrant/Neo4j/TEI
  - REST API contract for chat UI

### 2. ui
- Technology: React + TypeScript + TailwindCSS (built with Vite)
- Runtime: Nginx static file serving
- Port: 3000
- Dockerfile: dockerfile.ui
- Responsibilities:
  - Chat interface and interaction state
  - Session identifier persistence in browser storage
  - API communication with backend
  - Source list visualization and conversation reset flow

### 3. Supporting services
- Redis: key-value data
- Qdrant: vector storage
- Neo4j: graph storage
- TEI: embedding inference

## Backend API Contract

### POST /query
Request:
```json
{
  "query": "Como aplicar insulina?",
  "session_id": "optional-uuid"
}
```

Response:
```json
{
  "response": "Para aplicar insulina...",
  "sources": ["fonte 1", "fonte 2"],
  "source_count": 2,
  "summarized": false,
  "session_id": "session-uuid"
}
```

### GET /health
Response:
```json
{
  "status": "healthy",
  "message": "Chatbot API is running"
}
```

### DELETE /session/{session_id}
Response:
```json
{
  "message": "Session {session_id} cleared successfully"
}
```

## Frontend Structure

```text
frontend/
  src/
    app/
      App.tsx
    features/
      chat/
        ChatPage.tsx
        components/
          Composer.tsx
          MessageBubble.tsx
          SourceDrawer.tsx
    lib/
      api.ts
      env.ts
      storage.ts
    types/
      chat.ts
    index.css
    main.tsx
```

## Environment Variables

Root .env:
- BACKEND_PORT=8000
- UI_PORT=3000
- VITE_API_URL=/api
- VITE_REQUEST_TIMEOUT_MS=60000
- FRONTEND_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

Frontend .env (optional for local dev):
- VITE_API_URL=/api
- VITE_REQUEST_TIMEOUT_MS=60000

## Running

### Docker Compose
```bash
docker-compose up --build
```

- UI: http://localhost:3000
- Backend docs: http://localhost:8000/docs

The `kb_builder` service is now optional and only runs when requested via profile:

```bash
docker compose --profile kb up --build kb_builder
```

### Frontend local dev (without Docker)
```bash
cd frontend
npm install
npm run dev
```

## Security Notes

- CORS uses explicit origins through FRONTEND_ORIGINS.
- API currently has no authentication layer.
- Session history is in-memory and resets when backend restarts.
