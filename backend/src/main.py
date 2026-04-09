import os
import uuid
import logging

import streamlit as st
import requests
from dotenv import load_dotenv

# Initialize
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Assistente de Insulinoterapia",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Backend API configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))


def check_backend_health():
    """Check if backend API is available."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Backend health check failed: {e}")
        return False


def initialize_session_state():
    """Initialize session state for UI."""
    # Initialize persistent conversation state only if missing
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("backend_available", None)


def get_rag_response(query):
    """Query backend API and return complete response with sources."""
    logger.info(f"Querying backend API with session_id: {st.session_state.session_id}")
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={
                "query": query,
                "session_id": st.session_state.session_id
            },
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info(
            f"Received response: {len(result.get('response', ''))} chars, "
            f"{result.get('source_count', 0)} sources"
        )
        
        return result
    
    except requests.exceptions.Timeout:
        logger.error("Request to backend timed out")
        raise Exception("O servidor demorou muito para responder. Tente novamente.")
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to backend")
        raise Exception("Não foi possível conectar ao servidor. Verifique se o serviço está ativo.")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error from backend: {e}")
        raise Exception(f"Erro no servidor: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def handle_user_input(query):
    """Process user query and generate response."""
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("💭 Pensando..."):
            try:
                result = get_rag_response(query)

                # Unpack the result dict
                if isinstance(result, dict):
                    response_text = result.get("response", "")
                    sources = result.get("sources", [])
                    source_count = result.get("source_count", 0)
                    was_summarized = result.get("summarized", False)
                else:
                    # Fallback for backward compatibility if somehow a string is returned
                    response_text = result
                    sources = []
                    source_count = 0
                    was_summarized = False

                # Display summarization warning if it just happened
                if was_summarized:
                    st.info("📋 Conversa resumida - histórico condensado para melhor desempenho")

                # Display the response
                st.markdown(response_text)

                # Display sources in an expander if any exist
                if sources and source_count > 0:
                    with st.expander(f"📚 Referências consultadas ({source_count})"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"{i}. {source}")

                # Store message (only response text, not sources)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                error_msg = f"Erro ao processar: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


def main():
    """Main application entry point."""
    initialize_session_state()

    # Check backend health on first load
    if st.session_state.backend_available is None:
        with st.spinner("🔄 Conectando ao servidor..."):
            st.session_state.backend_available = check_backend_health()
    
    # Show warning if backend is not available
    if not st.session_state.backend_available:
        st.error("⚠️ Não foi possível conectar ao servidor do chatbot. Verifique se o serviço backend está ativo.")
        if st.button("🔄 Tentar reconectar"):
            st.session_state.backend_available = check_backend_health()
            st.rerun()
        return

    st.title("🩺 Assistente de Insulinoterapia")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Digite sua pergunta..."):
        handle_user_input(query)


if __name__ == "__main__":
    main()
