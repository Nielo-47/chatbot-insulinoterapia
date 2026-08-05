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
  +--> PostgreSQL
  +--> Redis (RAG + chat cache)
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
  - Persistent conversation storage (PostgreSQL)
  - Conversation read cache (Redis)
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
- PostgreSQL: durable user/conversation/message storage
- Redis: key-value data and conversation cache
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

### DELETE /user/conversations
Response:
```json
{
  "message": "Conversation cleared successfully"
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
- DATABASE_URL=postgresql+psycopg://chatbot:chatbot@postgres:5432/chatbot
- CHAT_CACHE_REDIS_URL=redis://redis:6379/1
- CHAT_CACHE_TTL_SECONDS=300
- TEMP_USER_PREFIX=session

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
- Authentication is fully delegated to Authentik (forward-auth). The nginx edge
  runs an Authentik auth subrequest for the UI and all `/api/` endpoints (except
  `/api/health`); on success it injects `X-authentik-uid` / `X-authentik-username`
  headers, which the backend consumes only when the direct peer is a configured
  trusted proxy (TRUSTED_PROXY_IPS). Login, sessions, MFA, password policy, rate
  limiting and lockout are all handled by Authentik; the backend stores no
  credentials (the `users` table has no password column).
- The frontend never stores tokens: `/auth/me` probes the session, and login/
  logout are plain redirects to Authentik flows.
- Account deletion revokes the user in Authentik via the Admin API (AUTHENTIK_ADMIN_API_TOKEN)
  before purging local data; it fails closed (502) when Authentik cannot be reached.
- Conversation history is persisted in PostgreSQL and cached in Redis.
