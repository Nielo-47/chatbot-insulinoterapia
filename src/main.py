import os
import queue
import asyncio
import multiprocessing as mp
from typing import List
import tiktoken

import streamlit as st
import nest_asyncio
from dotenv import load_dotenv
from openai import OpenAI
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np

# Configuration
KG_DIR = os.getenv("WORKING_DIR", "data/processed/")
RAG_TIMEOUT = 60
VLLM_LLM_HOST = os.getenv("LLM_BINDING_HOST", "http://localhost:8000")
VLLM_EMBED_HOST = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:8001")
LLM_MODEL = os.getenv("LLM_MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
MAX_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "512"))
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "8192"))  # Default for Llama-3.2-3B

# Initialize OpenAI clients for vLLM (LLM and Embeddings)
vllm_llm_client = OpenAI(
    api_key="EMPTY",
    base_url=f"{VLLM_LLM_HOST}/v1",
)

vllm_embed_client = OpenAI(
    api_key="EMPTY",
    base_url=f"{VLLM_EMBED_HOST}/v1",
)

# Initialize
nest_asyncio.apply()
load_dotenv()
st.set_page_config(
    page_title="Assistente de Insulinoterapia",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize tokenizer for token counting
try:
    # Try to use the exact model tokenizer
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Fallback tokenizer
except:
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Default encoding


def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    try:
        return len(tokenizer.encode(text))
    except:
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4


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


async def vllm_model_complete(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    """Complete text using vLLM OpenAI-compatible API."""
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Calculate token usage
    total_tokens = sum(count_tokens(str(msg.get("content", ""))) for msg in messages)

    print(f"Messages to vLLM (Total tokens: {total_tokens}/{MAX_CONTEXT_LENGTH}):", messages)

    response = vllm_llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=kwargs.get("temperature", 0.1),
        max_tokens=kwargs.get("max_tokens", 2048),
    )

    # Store token usage in session state if available
    try:
        if hasattr(st.session_state, "last_token_usage"):
            completion_tokens = count_tokens(response.choices[0].message.content)
            st.session_state.last_token_usage = {
                "prompt_tokens": total_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens + completion_tokens,
                "context_percentage": (total_tokens / MAX_CONTEXT_LENGTH) * 100,
            }
    except:
        pass

    return response.choices[0].message.content


async def vllm_embed_func(texts: List[str]) -> np.ndarray:
    """Generate embeddings using vLLM OpenAI-compatible API."""
    if isinstance(texts, str):
        texts = [texts]

    print("Generating embeddings for texts:", texts)

    response = vllm_embed_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )

    return np.array([item.embedding for item in response.data])


def initialize_rag():
    """Initialize LightRAG with vLLM configuration."""
    return LightRAG(
        working_dir=KG_DIR,
        llm_model_func=vllm_model_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={
            "timeout": 300,
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=MAX_TOKENS,
            func=vllm_embed_func,
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
            # Use .query() instead of .query_data() to get the full generated response
            response = rag.query(query, param=QueryParam(mode="hybrid"))
            output_queue.put(response)
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
    st.session_state.last_token_usage = None
    st.session_state.token_history = []


def get_rag_response(query):
    """Query RAG and return complete response."""
    st.session_state.input_queue.put(query)
    response = st.session_state.output_queue.get(timeout=RAG_TIMEOUT)

    if isinstance(response, Exception):
        raise response

    return response


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
            # Get complete response from RAG with status
            with st.status("Processando sua pergunta...", expanded=True) as status:
                st.write("🔍 Consultando base de conhecimento...")
                st.write("💭 Gerando resposta...")

                response = get_rag_response(query)

                status.update(label="Resposta gerada!", state="complete")

            # Display the response
            st.markdown(response)

            # Store response
            st.session_state.messages.append({"role": "assistant", "content": response})

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
            user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
            assistant_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Perguntas", user_msgs)
            with col2:
                st.metric("Respostas", assistant_msgs)

            # Token usage statistics
            if st.session_state.token_history:
                st.subheader("📈 Uso de Tokens")
                total_prompt_tokens = sum(t["prompt_tokens"] for t in st.session_state.token_history)
                total_completion_tokens = sum(t["completion_tokens"] for t in st.session_state.token_history)
                total_tokens = sum(t["total_tokens"] for t in st.session_state.token_history)
                avg_context_usage = sum(t["context_percentage"] for t in st.session_state.token_history) / len(
                    st.session_state.token_history
                )

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
                            last["context_percentage"] / 100, text=f"Contexto: {last['context_percentage']:.1f}%"
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

        # Check vLLM server health
        try:
            vllm_llm_client.models.list()
            llm_status = "🟢 Online"
        except:
            llm_status = "🔴 Offline"

        try:
            vllm_embed_client.models.list()
            embed_status = "🟢 Online"
        except:
            embed_status = "🔴 Offline"

        with st.expander("⚙️ Configurações do Sistema"):
            st.code(
                f"""
Modelo LLM: {LLM_MODEL}
Status LLM: {llm_status}
Host LLM: {VLLM_LLM_HOST}

Modelo Embedding: {EMBED_MODEL}
Status Embedding: {embed_status}
Host Embedding: {VLLM_EMBED_HOST}

Dimensão: {EMBEDDING_DIM}
Max Tokens: {MAX_TOKENS}
Contexto Máximo: {MAX_CONTEXT_LENGTH:,} tokens
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
