import asyncio
import multiprocessing as mp
import uuid

import streamlit as st
import nest_asyncio
import logging
from dotenv import load_dotenv
from src.chatbot import Chatbot
from src.config import Config

# Initialize
try:
    nest_asyncio.apply()
except ValueError as e:
    logging.getLogger(__name__).warning(
        "nest_asyncio could not patch the event loop (likely uvloop): %s. Continuing without nest_asyncio.",
        e,
    )
load_dotenv()
st.set_page_config(
    page_title="Assistente de Insulinoterapia",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def get_event_loop():
    """Get or create an event loop, avoiding nest_asyncio patched version."""
    # In worker processes or when nest_asyncio failed to patch uvloop,
    # directly create a new event loop to avoid the patched get_event_loop
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
    except Exception as e:
        logging.getLogger(__name__).error("Failed to create event loop: %s", e)
        raise


def rag_worker(input_queue, output_queue):
    """Process RAG queries in a separate process."""
    logging.info("[DEBUG] RAG worker starting...")
    try:
        nest_asyncio.apply()
    except ValueError as e:
        logging.getLogger(__name__).warning(
            "nest_asyncio could not patch the event loop in worker (likely uvloop): %s. Continuing without nest_asyncio.",
            e,
        )

    logging.info("[DEBUG] Creating Chatbot instance...")
    chatbot = Chatbot()

    logging.info("[DEBUG] Getting event loop...")
    loop = get_event_loop()
    logging.info("[DEBUG] Running chatbot.initialize_rag()...")
    loop.run_until_complete(chatbot.initialize_rag())
    logging.info("[DEBUG] RAG worker ready, entering message loop...")

    while True:
        qobj = input_queue.get()
        if qobj is None:  # Stop sentinel
            logging.info("[DEBUG] RAG worker received stop signal")
            break

        # Accept either a plain string or a dict with keys like {"query": ..., "session_id": ...}
        if isinstance(qobj, dict):
            query_text = qobj.get("query")
            session_id = qobj.get("session_id")
        else:
            query_text = qobj
            session_id = None

        try:
            logging.info("[DEBUG] RAG worker processing query: %s...", (query_text or "")[:50])
            result = chatbot.query(query_text, session_id=session_id)
            # result is now a dict with {"response": ..., "sources": ..., "source_count": ..., "summarized": ...}
            was_summarized = result.get("summarized", False)
            logging.info(
                "[DEBUG] RAG worker sending result: response=%d chars, sources=%d%s",
                len(result.get("response", "")),
                result.get("source_count", 0),
                ", CONVERSATION SUMMARIZED" if was_summarized else "",
            )
            output_queue.put(result)
        except Exception as e:
            logging.error("[ERROR] RAG worker exception: %s: %s", type(e).__name__, e)
            import traceback

            traceback.print_exc()
            output_queue.put(e)


def initialize_session_state():
    """Initialize RAG process and session state.

    Preserve existing messages and token history if present. If a previous
    RAG process exists but is not alive, recreate it without clearing
    the conversation state.
    """
    rag_proc = st.session_state.get("rag_process")
    if rag_proc is not None:
        try:
            if getattr(rag_proc, "is_alive", lambda: False)():
                return
        except Exception:
            # If checking fails, proceed to recreate
            pass

    input_queue = mp.Queue()
    output_queue = mp.Queue()
    rag_process = mp.Process(target=rag_worker, args=(input_queue, output_queue))
    rag_process.start()

    st.session_state.input_queue = input_queue
    st.session_state.output_queue = output_queue
    st.session_state.rag_process = rag_process

    # Initialize persistent conversation state only if missing
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", str(uuid.uuid4()))


def get_rag_response(query):
    """Query RAG and return complete response with sources."""
    logging.info("[DEBUG] get_rag_response called, timeout=%s", Config.RAG_TIMEOUT)
    st.session_state.input_queue.put({"query": query, "session_id": st.session_state.session_id})
    logging.info("[DEBUG] Query sent to worker (with session_id), waiting for response...")

    try:
        result = st.session_state.output_queue.get(timeout=Config.RAG_TIMEOUT)
        logging.info("[DEBUG] Received result from worker")
    except Exception as e:
        logging.error("[ERROR] Timeout or error waiting for response: %s: %s", type(e).__name__, e)
        raise

    if isinstance(result, Exception):
        logging.error(
            "[ERROR] Worker returned exception: %s: %s",
            type(result).__name__,
            result,
        )
        raise result

    # Result should now be a dict with response, sources, source_count
    return result


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

    st.title("🩺 Assistente de Insulinoterapia")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Digite sua pergunta..."):
        handle_user_input(query)


if __name__ == "__main__":
    main()
