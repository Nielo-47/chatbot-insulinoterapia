import os
import queue
import asyncio
import multiprocessing as mp
import time

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
    layout="wide",
    initial_sidebar_state="expanded",
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
        query = input_queue.get()
        if query is None:  # Stop sentinel
            logging.info("[DEBUG] RAG worker received stop signal")
            break

        try:
            logging.info("[DEBUG] RAG worker processing query: %s...", query[:50])
            response = chatbot.query(query)
            logging.info("[DEBUG] RAG worker sending response: %s chars", len(response))
            output_queue.put(response)
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
    st.session_state.setdefault("last_token_usage", None)
    st.session_state.setdefault("token_history", [])


def get_rag_response(query):
    """Query RAG and return complete response."""
    logging.info("[DEBUG] get_rag_response called, timeout=%s", Config.RAG_TIMEOUT)
    st.session_state.input_queue.put(query)
    logging.info("[DEBUG] Query sent to worker, waiting for response...")

    try:
        response = st.session_state.output_queue.get(timeout=Config.RAG_TIMEOUT)
        logging.info("[DEBUG] Received response from worker")
    except Exception as e:
        logging.error(
            "[ERROR] Timeout or error waiting for response: %s: %s", type(e).__name__, e
        )
        raise

    if isinstance(response, Exception):
        logging.error(
            "[ERROR] Worker returned exception: %s: %s",
            type(response).__name__,
            response,
        )
        raise response

    return response


def display_chat_history():
    """Display all messages in chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input(query):
    """Process user query and generate response."""
    # Append user message to session history
    st.session_state.messages.append({"role": "user", "content": query})

    # Re-render chat history immediately so user sees their message
    display_chat_history()

    try:
        start_total = time.perf_counter()
        status_placeholder = st.empty()

        # Step 1: Context retrieval (RAG)
        step_1_start = time.perf_counter()
        status_placeholder.markdown(
            "🔍 **Consultando base de conhecimento...**\n\n" f"⏱️ Tempo: 0.00s"
        )

        response = get_rag_response(query)
        step_1_elapsed = time.perf_counter() - step_1_start

        # Step 2: Response generation (if additional processing required)
        step_2_start = time.perf_counter()
        status_placeholder.markdown(
            "✅ **Contexto obtido**\n"
            f"⏱️ Tempo: {step_1_elapsed:.2f}s\n\n"
            "💭 **Gerando resposta...**\n\n"
            "⏱️ Tempo: 0.00s"
        )

        # No extra processing here; capture elapsed
        step_2_elapsed = time.perf_counter() - step_2_start
        total_elapsed = time.perf_counter() - start_total

        # Append assistant response and re-render chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        display_chat_history()

        # Update and clear status promptly to avoid lingering UI
        status_placeholder.markdown(
            "✅ **Contexto obtido**\n"
            f"⏱️ Tempo: {step_1_elapsed:.2f}s\n\n"
            "✅ **Resposta gerada**\n"
            f"⏱️ Tempo: {step_2_elapsed:.2f}s\n\n"
            f"**Tempo total:** {total_elapsed:.2f}s"
        )
        # Small pause so users see timings, then clear
        time.sleep(0.25)
        status_placeholder.empty()

        # Store token usage in history
        if st.session_state.last_token_usage:
            st.session_state.token_history.append(st.session_state.last_token_usage)

            # Display token usage info
            with st.expander("📊 Detalhes de Uso de Tokens"):
                usage = st.session_state.last_token_usage
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tokens de Prompt", usage["prompt_tokens"])
                with col2:
                    st.metric("Tokens de Resposta", usage["completion_tokens"])
                with col3:
                    st.metric("Total de Tokens", usage["total_tokens"])

                # Context window usage
                st.progress(
                    usage["context_percentage"] / 100,
                    text=f"Uso da Janela de Contexto: {usage['context_percentage']:.1f}%",
                )

    except queue.Empty:
        st.error("⏱️ Tempo limite ao esperar resposta do RAG.")
    except asyncio.TimeoutError:
        st.error("⏱️ Tempo limite ao consultar o modelo. Tente novamente.")
    except Exception as e:
        st.error(f"❌ Erro: {str(e)}")


def display_sidebar():
    """Display sidebar with information and controls."""
    with st.sidebar:
        st.header("ℹ️ Sobre o Assistente")
        st.info(
            """
        Este assistente utiliza IA para responder perguntas sobre insulinoterapia, 
        consultando uma base de conhecimento especializada.
        """
        )

        st.divider()

        st.header("📊 Estatísticas da Sessão")
        if st.session_state.messages:
            user_msgs = len(
                [m for m in st.session_state.messages if m["role"] == "user"]
            )
            assistant_msgs = len(
                [m for m in st.session_state.messages if m["role"] == "assistant"]
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Perguntas", user_msgs)
            with col2:
                st.metric("Respostas", assistant_msgs)

            # Token usage statistics
            if st.session_state.token_history:
                st.subheader("📈 Uso de Tokens")
                total_prompt_tokens = sum(
                    t["prompt_tokens"] for t in st.session_state.token_history
                )
                total_completion_tokens = sum(
                    t["completion_tokens"] for t in st.session_state.token_history
                )
                total_tokens = sum(
                    t["total_tokens"] for t in st.session_state.token_history
                )
                avg_context_usage = sum(
                    t["context_percentage"] for t in st.session_state.token_history
                ) / len(st.session_state.token_history)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total de Tokens", f"{total_tokens:,}")
                    st.metric("Tokens de Prompt", f"{total_prompt_tokens:,}")
                with col2:
                    st.metric("Uso Médio do Contexto", f"{avg_context_usage:.1f}%")
                    st.metric("Tokens de Resposta", f"{total_completion_tokens:,}")

                # Last query details
                if st.session_state.last_token_usage:
                    with st.expander("🔍 Última Consulta"):
                        last = st.session_state.last_token_usage
                        st.write(f"**Prompt:** {last['prompt_tokens']:,} tokens")
                        st.write(f"**Resposta:** {last['completion_tokens']:,} tokens")
                        st.write(f"**Total:** {last['total_tokens']:,} tokens")
                        st.progress(
                            last["context_percentage"] / 100,
                            text=f"Contexto: {last['context_percentage']:.1f}%",
                        )
        else:
            st.write("Nenhuma conversa ainda.")

        st.divider()

        if st.button("🗑️ Limpar Conversa", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.token_history = []
            st.session_state.last_token_usage = None
            st.rerun()

        st.divider()

        # LLM/Embedding status (vLLM removed): LLMs use OpenRouter; embeddings served by TEI
        llm_status = "OpenRouter (remote)"
        embed_status = "TEI (embedding service)"

        with st.expander("⚙️ Configurações do Sistema"):
            st.code(
                f"""
Modelo LLM: {Config.LLM_MODEL}
Status LLM: {llm_status}
Host LLM: {Config.OPENROUTER_BASE_URL}

Modelo Embedding: {Config.EMBED_MODEL}
Status Embedding: {embed_status}
Host Embedding:     {Config.EMBED_HOST} (TEI)

Dimensão: {Config.EMBEDDING_DIM}
Max Tokens: {Config.MAX_TOKENS}
Contexto Máximo: {Config.MAX_CONTEXT_LENGTH:,} tokens
            """
            )


def display_welcome_message():
    """Display welcome message when chat is empty."""
    if not st.session_state.messages:
        st.info("👋 **Bem-vindo ao Assistente de Insulinoterapia!**")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📚 Tópicos Disponíveis")
            st.markdown(
                """
            - Tipos de insulina
            - Protocolos de dosagem
            - Ajustes e monitoramento
            - Orientações clínicas
            """
            )

        with col2:
            st.subheader("💡 Dicas de Uso")
            st.markdown(
                """
            - Faça perguntas específicas
            - Use linguagem clara
            - Consulte o histórico na barra lateral
            - Limpe a conversa quando necessário
            """
            )

        st.divider()


def main():
    """Main application entry point."""
    initialize_session_state()

    # Header
    st.title("🩺 Assistente de Insulinoterapia")

    # Sidebar
    display_sidebar()

    # Welcome message
    display_welcome_message()

    # Chat history
    display_chat_history()

    # Chat input
    if query := st.chat_input("💬 Digite sua pergunta sobre insulinoterapia..."):
        handle_user_input(query)


if __name__ == "__main__":
    main()
