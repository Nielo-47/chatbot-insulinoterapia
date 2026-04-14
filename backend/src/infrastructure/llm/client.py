import logging
from typing import Dict, List, Optional

from openai import AsyncOpenAI

from backend.src.config.infrastructure import OPENROUTER_HTTP_REFERER, OPENROUTER_SITE_TITLE
from backend.src.config.rag import LLM_MODEL

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, api_key: str, base_url: str):
        self._api_key = api_key
        self._base_url = base_url

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> str:
        client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history_messages:
            for msg in history_messages:
                content = str(msg.get("content", "")).strip()
                if content:
                    messages.append({"role": msg["role"], "content": content})

        user_content = str(prompt).strip() if prompt else "Olá"
        messages.append({"role": "user", "content": user_content})

        extra_headers = {}
        if OPENROUTER_HTTP_REFERER:
            extra_headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
        if OPENROUTER_SITE_TITLE:
            extra_headers["X-Title"] = OPENROUTER_SITE_TITLE

        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=extra_headers if extra_headers else None,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )

            res_content = response.choices[0].message.content or ""
            return str(res_content).strip()

        except Exception as e:
            logger.exception("Erro na chamada ao OpenRouter: %s", e)
            return "Tive um problema técnico. Por favor, tente perguntar de outra forma."
