import os
import logging
import numpy as np
from openai import OpenAI
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from typing import List, Dict
from config import Config
from lightrag.prompt import PROMPTS


class Chatbot:
    def __init__(self):
        if not Config.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required. Set the OPENROUTER_API_KEY environment variable."
            )

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

        messages = [{"role": "system", "content": system_prompt}]

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

            if not res_content or len(res_content.strip()) < 5:
                return "Infelizmente não encontrei informações específicas sobre isso nos meus manuais médicos."

            return res_content.strip()

        except Exception as e:
            print(f"Erro na chamada ao OpenRouter: {e}")
            return (
                "Tive um problema técnico. Por favor, tente perguntar de outra forma."
            )

    async def _critique_response(
        self, original_query: str, response: str
    ) -> Dict[str, any]:
        """Critique a response for safety, accuracy, and quality using direct LLM call."""
        critique_prompt = Config.CRITIQUE_PROMPT.format(
            original_query=original_query, response=response
        )

        critique_system_prompt = (
            "Você é um revisor médico rigoroso. Responda APENAS com JSON válido, "
            "sem texto adicional antes ou depois."
        )

        critique_text = await self._call_llm(
            prompt=critique_prompt,
            system_prompt=critique_system_prompt,
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

    def _build_refinement_query(
        self, original_query: str, previous_response: str, critique: Dict[str, any]
    ) -> str:
        """Build a refinement query for RAG."""
        issues_text = "\n- ".join(
            critique.get("issues", ["Nenhum problema identificado"])
        )
        suggestions_text = "\n- ".join(
            critique.get("suggestions", ["Mantenha a qualidade atual"])
        )

        refinement_query = Config.REFINEMENT_PROMPT.format(
            original_query=original_query,
            previous_response=previous_response,
            issues_text=issues_text,
            suggestions_text=suggestions_text,
        )

        return refinement_query

    async def initialize_rag(self):
        """Initialize LightRAG with OpenRouter configuration."""
        print("[DEBUG] Initializing RAG...")

        async def openrouter_model_complete(
            prompt, system_prompt=None, history_messages=[], **kwargs
        ) -> str:
            """Simple completion without refinement - used by RAG internally."""
            return await self._call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
            )

        async def tei_embed_func(texts: List[str]) -> np.ndarray:
            """Generate embeddings using TEI OpenAI-compatible API."""
            print(
                f"[DEBUG] tei_embed_func called with {len(texts) if isinstance(texts, list) else 1} text(s)"
            )
            print(f"[DEBUG] Using embed base_url: {self.embed_config['base_url']}")

            try:
                client = OpenAI(**self.embed_config)

                if isinstance(texts, str):
                    texts = [texts]

                print(
                    f"[DEBUG] Calling embeddings.create with model={Config.EMBED_MODEL}"
                )
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

    def query(self, query: str, mode: str = "hybrid", **query_params) -> str:
        """Query RAG with single-round refinement and return refined response."""
        logging.info("[DEBUG] query() called with mode=%s", mode)
        logging.info("[DEBUG] Original query: %s...", query[:100])

        # Step 1: Generate initial response using RAG
        print("\n[Step 1/3] Gerando resposta inicial com RAG...")
        try:
            initial_response = self.rag.query(
                query,
                param=QueryParam(
                    mode=mode,
                    user_prompt=query_params.get("system_prompt", Config.SYSTEM_PROMPT),
                    max_total_tokens=query_params.get("max_total_tokens", 12000),
                    top_k=query_params.get("top_k", 10),
                    enable_rerank=False,
                ),
            )
            print(
                f"[DEBUG] RAG query completed, response length: {len(initial_response)} chars"
            )
        except Exception as e:
            print(f"[ERROR] RAG query failed: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            raise

        # Se não encontrou contexto, retorna a mensagem padrão sem refinar
        if initial_response == PROMPTS["fail_response"]:
            print("ℹ Sem contexto relevante - retornando mensagem padrão\n")
            return initial_response

        # Step 2: Critique the response (no RAG, just LLM)
        print("\n[Step 2/3] Analisando qualidade da resposta...")
        import asyncio

        critique = asyncio.run(self._critique_response(query, initial_response))

        # Check if refinement is needed
        if not critique.get("needs_refinement", False):
            print("✓ Resposta aprovada - nenhum refinamento necessário\n")
            return initial_response

        # Log issues found
        if critique.get("issues"):
            print(f"⚠ Problemas encontrados: {', '.join(critique['issues'])}")

        # Step 3: Refine response using RAG with refinement query
        print("\n[Step 3/3] Refinando resposta...")
        refinement_query = self._build_refinement_query(
            query, initial_response, critique
        )
        refined_response = asyncio.run(self._call_llm(refinement_query))

        print("✓ Resposta refinada e aprovada\n")

        return refined_response
