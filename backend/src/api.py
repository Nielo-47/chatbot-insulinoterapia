"""FastAPI Backend for Diabetes Chatbot - Exposes RAG functionality via REST API."""
import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import nest_asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.src.chatbot import Chatbot
from backend.src.config import Config


def _parse_frontend_origins() -> List[str]:
    raw_origins = os.getenv("FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

# Initialize
try:
    nest_asyncio.apply()
except ValueError as e:
    logging.getLogger(__name__).warning(
        "nest_asyncio could not patch the event loop (likely uvloop): %s. Continuing without nest_asyncio.",
        e,
    )
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global chatbot instance
chatbot_instance: Optional[Chatbot] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage chatbot lifecycle."""
    global chatbot_instance
    logger.info("Initializing chatbot...")
    chatbot_instance = Chatbot()
    await chatbot_instance.initialize_rag()
    logger.info("Chatbot initialized successfully")
    yield
    logger.info("Shutting down chatbot...")


app = FastAPI(
    title="Diabetes Chatbot API",
    description="Backend API for diabetes chatbot with RAG functionality",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for communication with UI container
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_frontend_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    sources: List[str]
    source_count: int
    summarized: bool
    session_id: str


class HealthResponse(BaseModel):
    status: str
    message: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    return HealthResponse(status="healthy", message="Chatbot API is running")


@app.post("/query", response_model=QueryResponse)
def query_chatbot(request: QueryRequest):
    """Query the chatbot with a question."""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        # Generate session_id if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"Processing query for session {session_id}: {request.query[:50]}...")
        
        # Query the chatbot
        result = chatbot_instance.query(request.query, session_id=session_id)
        
        # Add session_id to result
        result["session_id"] = session_id
        
        logger.info(
            f"Query completed for session {session_id}: "
            f"response={len(result.get('response', ''))} chars, "
            f"sources={result.get('source_count', 0)}, "
            f"summarized={result.get('summarized', False)}"
        )
        
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f"Error processing query: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a specific session."""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        if session_id in chatbot_instance.conversations:
            del chatbot_instance.conversations[session_id]
            logger.info(f"Cleared session {session_id}")
            return {"message": f"Session {session_id} cleared successfully"}
        else:
            return {"message": f"Session {session_id} not found"}
    except Exception as e:
        logger.error(f"Error clearing session: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Diabetes Chatbot API",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
