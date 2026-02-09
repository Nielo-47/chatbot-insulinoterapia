import os
import logging
import numpy as np
from openai import OpenAI
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from typing import List, Dict
from config import Config
from lightrag.prompt import PROMPTS
import asyncio


class Chatbot:
    def __init__(self):
        if not Config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is required. Set the OPENROUTER_API_KEY environment variable.")

        self.llm_config = {
            "api_key": Config.OPENROUTER_API_KEY,
            "base_url": Config.OPENROUTER_BASE_URL,
        }

        self.embed_config = {
            "api_key": os.getenv("EMBEDDING_API_KEY", ""),
            "base_url": f"{Config.EMBED_HOST}/v1",
        }

        print(f"[DEBUG] Chatbot initialized")
        print(f"[DEBUG] LLM config: base_url={self.llm_config['base_url']}")
        print(f"[DEBUG] Embed config: base_url={self.embed_config['base_url']}")
        print(f"[DEBUG] Config.EMBED_HOST={Config.EMBED_HOST}")

        self.rag = None
        # Store per-session conversation histories (session_id -> list of messages)
        self.conversations: Dict[str, List[Dict]] = {}
        # Track which sessions were just summarized
        self.sessions_summarized: set = set()

        PROMPTS["fail_response"] = (
            "Desculpe, não encontrei informações sobre isso nos meus manuais de diabetes e insulinoterapia. "
            "Essa pergunta não está relacionada aos temas que posso ajudar (diabetes, insulina, glicemia). "
            "Se você tiver dúvidas sobre esses temas, ficarei feliz em ajudar!"
        )

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: str = None,
        history_messages: List[Dict] = None,
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> str:
        """Internal method to call LLM directly without RAG."""
        client = OpenAI(**self.llm_config)

        messages = []
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add history messages
        if history_messages:
            for msg in history_messages:
                content = str(msg.get("content", "")).strip()
                if content:
                    messages.append({"role": msg["role"], "content": content})

        # Add current prompt
        user_content = str(prompt).strip() if prompt else "Olá"
        messages.append({"role": "user", "content": user_content})

        # Build extra_headers for OpenRouter
        extra_headers = {}
        if Config.OPENROUTER_HTTP_REFERER:
            extra_headers["HTTP-Referer"] = Config.OPENROUTER_HTTP_REFERER
        if Config.OPENROUTER_SITE_TITLE:
            extra_headers["X-Title"] = Config.OPENROUTER_SITE_TITLE

        try:
            response = client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=extra_headers if extra_headers else None,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )

            res_content = response.choices[0].message.content

            return res_content.strip()

        except Exception as e:
            print(f"Erro na chamada ao OpenRouter: {e}")
            return "Tive um problema técnico. Por favor, tente perguntar de outra forma."

    async def _critique_response(
        self, original_query: str, response: str, history_messages: List[Dict] = None
    ) -> Dict[str, any]:
        """Critique a response for safety, accuracy, and quality using direct LLM call."""
        critique_prompt = Config.CRITIQUE_PROMPT.format(original_query=original_query, response=response)

        critique_system_prompt = (
            "Você é um revisor médico rigoroso. Responda APENAS com JSON válido, "
            "sem texto adicional antes ou depois."
        )

        critique_text = await self._call_llm(
            prompt=critique_prompt,
            system_prompt=critique_system_prompt,
            history_messages=history_messages,
            temperature=0.0,
            max_tokens=600,
        )

        # Parse JSON response
        try:
            import json

            # Remove markdown code blocks if present
            critique_text = critique_text.strip()
            if critique_text.startswith("```"):
                critique_text = critique_text.split("```")[1]
                if critique_text.startswith("json"):
                    critique_text = critique_text[4:]

            critique = json.loads(critique_text.strip())
            return critique
        except json.JSONDecodeError as e:
            print(f"Erro ao parsear crítica: {e}")
            # Default to safe critique if parsing fails
            return {
                "is_safe": True,
                "is_accurate": True,
                "is_clear": True,
                "is_complete": True,
                "is_ethical": True,
                "issues": [],
                "suggestions": [],
                "needs_refinement": False,
            }

    def _build_refinement_query(self, original_query: str, previous_response: str, critique: Dict[str, any]) -> str:
        """Build a refinement query for RAG."""
        issues_text = "\n- ".join(critique.get("issues", ["Nenhum problema identificado"]))
        suggestions_text = "\n- ".join(critique.get("suggestions", ["Mantenha a qualidade atual"]))

        refinement_query = Config.REFINEMENT_PROMPT.format(
            original_query=original_query,
            previous_response=previous_response,
            issues_text=issues_text,
            suggestions_text=suggestions_text,
        )

        return refinement_query

    # --- Conversation/session helpers ---
    def _init_session_if_missing(self, session_id: str):
        if not session_id:
            return
        if session_id not in self.conversations:
            self.conversations[session_id] = []

    def _add_message_to_session(self, session_id: str, role: str, content: str):
        """Add a message to session history and trigger auto-summarization if needed."""
        if not session_id:
            return
        self._init_session_if_missing(session_id)
        self.conversations[session_id].append({"role": role, "content": content})

        # Trigger summarization if too many messages
        try:
            if len(self.conversations[session_id]) >= Config.SUMMARIZE_MAX_MESSAGES:
                self.summarize_session(session_id)
                # Mark that this session was just summarized
                self.sessions_summarized.add(session_id)
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to auto-summarize session %s: %s", session_id, e)

    def add_user_message(self, session_id: str, content: str):
        """Add user message to session."""
        self._add_message_to_session(session_id, "user", content)

    def add_assistant_message(self, session_id: str, content: str):
        """Add assistant message to session."""
        self._add_message_to_session(session_id, "assistant", content)

    def get_conversation(self, session_id: str) -> List[Dict]:
        if not session_id:
            return []
        return list(self.conversations.get(session_id, []))

    def reset_conversation(self, session_id: str):
        if not session_id:
            return
        self.conversations[session_id] = []

    def summarize_session(self, session_id: str, max_messages: int = None) -> str:
        """Summarize the conversation for the given session and replace the history with a single assistant message.

        Returns the summary text.
        """
        if not session_id:
            return ""
        self._init_session_if_missing(session_id)
        msgs = self.conversations.get(session_id, [])
        if not msgs:
            return ""

        max_messages = max_messages or Config.SUMMARIZE_MAX_MESSAGES
        # Use the most recent messages up to max_messages to build the history
        recent = msgs[-max_messages:]
        history_lines = []
        for m in recent:
            role = m.get("role", "")
            content = str(m.get("content", "")).strip()
            if content:
                history_lines.append(f"{role.upper()}: {content}")

        history_text = "\n".join(history_lines)

        summary_prompt = Config.SUMMARY_PROMPT.format(history=history_text)

        try:
            # Use the LLM to create a short summary
            summary = asyncio.run(
                self._call_llm(
                    prompt=summary_prompt,
                    system_prompt=Config.SYSTEM_PROMPT,
                    temperature=0.1,
                    max_tokens=300,
                )
            )
            summary = summary.strip()
            if summary:
                # Replace conversation with a single assistant message containing the summary
                self.conversations[session_id] = [{"role": "assistant", "content": summary}]
                logging.getLogger(__name__).info("Session %s summarized into one message", session_id)
                return summary
        except Exception as e:
            logging.getLogger(__name__).warning("Error summarizing session %s: %s", session_id, e)

        return ""

    def _clean_source_path(self, file_path: str) -> str:
        """Clean up source file paths by removing unnecessary prefixes.

        Examples:
            'data/raw/INSULINOTERAPIA/Tipos de insulinas/file.pdf'
            -> 'INSULINOTERAPIA/Tipos de insulinas/file.pdf'
        """
        if not file_path:
            return file_path

        # Remove common prefixes
        prefixes_to_remove = ["data/raw/", "data\\raw\\", "./data/raw/"]
        for prefix in prefixes_to_remove:
            if file_path.startswith(prefix):
                file_path = file_path[len(prefix) :]
                break

        return file_path

    def _extract_sources(self, rag_data) -> tuple[list[str], int]:
        """Extract unique source references from RAG data.

        rag_data format (from LightRAG.query_data):
        {
            "status": "success" or "failure",
            "message": str,
            "data": {
                "entities": [...],
                "relationships": [...],
                "chunks": [{"content": str, "file_path": str, "chunk_id": str}, ...],
                "references": [{"reference_id": str, "file_path": str}, ...]
            },
            "metadata": {...}
        }

        Returns:
            tuple: (list of source strings, total count of unique sources)
        """
        sources = []
        seen = set()

        if not rag_data or not isinstance(rag_data, dict):
            return sources, 0

        try:
            # Check if this is a success response
            if rag_data.get("status") != "success":
                return sources, 0

            # Access the data section
            data_section = rag_data.get("data", {})
            if not data_section:
                return sources, 0

            # Extract file paths from chunks (most direct source)
            chunks = data_section.get("chunks", [])
            if isinstance(chunks, list):
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        file_path = chunk.get("file_path")
                        if file_path:
                            # Clean the path and deduplicate
                            clean_path = self._clean_source_path(file_path)
                            if clean_path and clean_path not in seen:
                                sources.append(clean_path)
                                seen.add(clean_path)

            # Also check references for additional mapping
            # (in case some sources are only in references)
            references = data_section.get("references", [])
            if isinstance(references, list):
                for ref in references:
                    if isinstance(ref, dict):
                        file_path = ref.get("file_path")
                        if file_path:
                            # Clean the path and deduplicate
                            clean_path = self._clean_source_path(file_path)
                            if clean_path and clean_path not in seen:
                                sources.append(clean_path)
                                seen.add(clean_path)

            # Extract from entities if they have direct file_path
            entities = data_section.get("entities", [])
            if isinstance(entities, list):
                for entity in entities:
                    if isinstance(entity, dict):
                        file_path = entity.get("file_path")
                        if file_path:
                            # Clean the path and deduplicate
                            clean_path = self._clean_source_path(file_path)
                            if clean_path and clean_path not in seen:
                                sources.append(clean_path)
                                seen.add(clean_path)

            # Extract from relationships if they have direct file_path
            relationships = data_section.get("relationships", [])
            if isinstance(relationships, list):
                for rel in relationships:
                    if isinstance(rel, dict):
                        file_path = rel.get("file_path")
                        if file_path:
                            # Clean the path and deduplicate
                            clean_path = self._clean_source_path(file_path)
                            if clean_path and clean_path not in seen:
                                sources.append(clean_path)
                                seen.add(clean_path)

            return sources, len(sources)

        except Exception as e:
            logging.warning(f"Failed to extract sources from RAG data: {e}")
            return sources, 0

    async def initialize_rag(self):
        """Initialize LightRAG with OpenRouter configuration."""
        print("[DEBUG] Initializing RAG...")

        async def openrouter_model_complete(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
            """Simple completion without refinement - used by RAG internally."""
            return await self._call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
            )

        async def tei_embed_func(texts: List[str]) -> np.ndarray:
            """Generate embeddings using TEI OpenAI-compatible API."""
            print(f"[DEBUG] tei_embed_func called with {len(texts) if isinstance(texts, list) else 1} text(s)")
            print(f"[DEBUG] Using embed base_url: {self.embed_config['base_url']}")

            try:
                client = OpenAI(**self.embed_config)

                if isinstance(texts, str):
                    texts = [texts]

                print(f"[DEBUG] Calling embeddings.create with model={Config.EMBED_MODEL}")
                response = client.embeddings.create(
                    model=Config.EMBED_MODEL,
                    input=texts,
                )
                print(f"[DEBUG] Embeddings received: {len(response.data)} embeddings")
                return np.array([item.embedding for item in response.data])
            except Exception as e:
                print(f"[ERROR] Embedding failed: {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()
                raise

        self.rag = LightRAG(
            working_dir=Config.KG_DIR,
            llm_model_func=openrouter_model_complete,
            llm_model_name=Config.LLM_MODEL,
            enable_llm_cache=False,
            embedding_func=EmbeddingFunc(
                embedding_dim=Config.EMBEDDING_DIM,
                max_token_size=Config.MAX_TOKENS,
                func=tei_embed_func,
            ),
        )

        print("[DEBUG] Initializing RAG storages...")
        await self.rag.initialize_storages()
        print("[DEBUG] RAG initialization complete")

    def query(self, query: str, mode: str = "hybrid", session_id: str = None, **query_params) -> Dict:
        """Query RAG with single-round refinement and return response with sources.

        If a session_id is provided, the method will use and update per-session conversation history
        so follow-up questions can be handled correctly.

        Returns:
            dict: {
                "response": str (the assistant's response),
                "sources": list[str] (unique source references),
                "source_count": int (number of sources),
                "summarized": bool (whether conversation was just summarized)
            }
        """
        logging.info("[DEBUG] query() called with mode=%s", mode)
        logging.info("[DEBUG] Original query: %s...", query[:100])

        # Build conversation_history from session storage or provided params
        conversation_history = (
            self.get_conversation(session_id) if session_id else query_params.get("conversation_history", [])
        )

        # Initialize sources tracking and summarization flag
        sources = []
        source_count = 0
        was_summarized = False

        # Step 1: Generate initial response using RAG
        logging.info("[Step 1/3] Generating initial response with RAG...")
        try:
            rag_data = self.rag.query_data(
                query,
                param=QueryParam(
                    mode=mode,
                    user_prompt=query_params.get("system_prompt", Config.SYSTEM_PROMPT),
                    max_total_tokens=query_params.get("max_total_tokens", 12000),
                    top_k=query_params.get("top_k", 10),
                    enable_rerank=False,
                    conversation_history=conversation_history,
                ),
            )

            # Extract sources from rag_data
            sources, source_count = self._extract_sources(rag_data)
            logging.info(f"Extracted {source_count} unique sources")

            data_preview = ""
            if isinstance(rag_data, dict):
                data_value = rag_data.get("data")
                if isinstance(data_value, (list, tuple)):
                    data_preview = str(data_value)[:500]
                elif isinstance(data_value, str):
                    data_preview = data_value[:500]
                else:
                    data_preview = str(data_value)[:500]
            else:
                data_preview = str(rag_data)[:500] if rag_data is not None else ""
            print(f"[DEBUG] RAG returned {len(rag_data) if rag_data else 0} data items \n{data_preview}")

            initial_response = asyncio.run(
                self._call_llm(
                    prompt=query,
                    system_prompt=query_params.get("system_prompt", Config.SYSTEM_PROMPT.format(context=rag_data)),
                    history_messages=conversation_history,
                )
            )
            logging.info("[DEBUG] RAG query completed, response length: %d chars", len(initial_response))
        except Exception as e:
            print(f"[ERROR] RAG query failed: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            raise

        # Append assistant message to session history (user message added by UI)
        if session_id:
            self.add_assistant_message(session_id, initial_response)

        # Check if conversation was just summarized
        was_summarized = session_id in self.sessions_summarized
        if was_summarized:
            self.sessions_summarized.discard(session_id)

        # Se não encontrou contexto, retorna a mensagem padrão sem refinar
        if initial_response == PROMPTS["fail_response"]:
            logging.info("No relevant context found - returning default message")
            return {
                "response": initial_response,
                "sources": sources,
                "source_count": source_count,
                "summarized": was_summarized,
            }

        # Step 2: Critique the response (no RAG, just LLM)
        logging.info("[Step 2/3] Analyzing response quality...")

        history_for_critique = self.get_conversation(session_id) if session_id else conversation_history
        critique = asyncio.run(self._critique_response(query, initial_response, history_messages=history_for_critique))

        # Check if refinement is needed
        if not critique.get("needs_refinement", False):
            logging.info("Response approved - no refinement needed")
            return {
                "response": initial_response,
                "sources": sources,
                "source_count": source_count,
                "summarized": was_summarized,
            }

        # Log issues found
        if critique.get("issues"):
            logging.warning(f"Issues found: {', '.join(critique['issues'])}")

        # Step 3: Refine response using RAG with refinement query
        logging.info("[Step 3/3] Refining response...")
        refinement_query = self._build_refinement_query(query, initial_response, critique)
        refined_response = asyncio.run(self._call_llm(refinement_query, history_messages=history_for_critique))

        # Save refined response into session history
        if session_id:
            self.add_assistant_message(session_id, refined_response)

        logging.info("Response refined and approved")
        return {
            "response": refined_response,
            "sources": sources,
            "source_count": source_count,
            "summarized": was_summarized,
        }
