import queue
import asyncio
import multiprocessing as mp
from functools import partial

import streamlit as st
import nest_asyncio
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_model_complete, ollama_embed

from utils.llm_helper import chat, stream_parser

# Configuration
KG_DIR = "data/processed/"
RAG_TIMEOUT = 60
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "qwen2.5:4b"
EMBED_MODEL = "paraphrase-multilingual:latest"
MAX_TOKENS = 8192

# Initialize
nest_asyncio.apply()
load_dotenv()
st.set_page_config(
    page_title="Assistente de Insulinoterapia",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_event_loop():
    """Get or create an event loop."""
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            return loop
    except RuntimeError:
        pass
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def initialize_rag():
    """Initialize LightRAG with configuration."""
    return LightRAG(
        working_dir=KG_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        summary_max_tokens=MAX_TOKENS,
        llm_model_kwargs={
            "host": OLLAMA_HOST,
            "options": {"num_ctx": MAX_TOKENS},
            "timeout": 300,
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=MAX_TOKENS,
            func=partial(
                ollama_embed.func,
                embed_model=EMBED_MODEL,
                host=OLLAMA_HOST,
            ),
        ),
    )


def rag_worker(input_queue, output_queue):
    """Process RAG queries in a separate process."""
    nest_asyncio.apply()
    rag = initialize_rag()
    loop = get_event_loop()
    loop.run_until_complete(rag.initialize_storages())

    while True:
        query = input_queue.get()
        if query is None:  # Stop sentinel
            break

        try:
            context = rag.query_data(query, param=QueryParam(mode="hybrid"))
            output_queue.put(context)
        except Exception as e:
            output_queue.put(e)


def initialize_session_state():
    """Initialize RAG process and session state."""
    if "rag_process" in st.session_state:
        return

    input_queue = mp.Queue()
    output_queue = mp.Queue()
    rag_process = mp.Process(target=rag_worker, args=(input_queue, output_queue))
    rag_process.start()

    st.session_state.input_queue = input_queue
    st.session_state.output_queue = output_queue
    st.session_state.rag_process = rag_process
    st.session_state.messages = []


def get_rag_context(query):
    """Query RAG and return context."""
    st.session_state.input_queue.put(query)
    context = st.session_state.output_queue.get(timeout=RAG_TIMEOUT)

    if isinstance(context, Exception):
        raise context

    return context


def display_chat_history():
    """Display all messages in chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input(query):
    """Process user query and generate response."""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(query)

    # Generate assistant response
    with st.chat_message("assistant"):
        try:
            # Get RAG context with status
            with st.status("Processando sua pergunta...", expanded=True) as status:
                st.write("🔍 Consultando base de conhecimento...")
                context = get_rag_context(query)

                st.write("💭 Gerando resposta...")

                # Prepare messages with context
                messages_for_llm = [
                    st.session_state.messages[0],
                    {"role": "system", "content": f"Contexto relevante:\n{context}"},
                    *st.session_state.messages[1:],
                ]

                # Generate and stream response
                llm_stream = chat(query)
                status.update(label="Resposta gerada!", state="complete")

            # Display the response
            response = st.write_stream(stream_parser(llm_stream))

            # Store response
            st.session_state.messages.append({"role": "assistant", "content": response})

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
        else:
            st.write("Nenhuma conversa ainda.")

        st.divider()

        if st.button("🗑️ Limpar Conversa", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        with st.expander("⚙️ Configurações do Sistema"):
            st.code(
                f"""
Modelo LLM: {LLM_MODEL}
Embedding: {EMBED_MODEL}
Max Tokens: {MAX_TOKENS}
Host: {OLLAMA_HOST}
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
