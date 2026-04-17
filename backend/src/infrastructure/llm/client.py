import asyncio
import logging
from typing import Dict, List, Optional

from openai import APITimeoutError, APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from backend.src.config.infrastructure import OPENROUTER_HTTP_REFERER, OPENROUTER_SITE_TITLE
from backend.src.config.rag import LLM_FALLBACK_MODEL, LLM_MODEL, LLM_PRIMARY_RETRIES, LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, api_key: str, base_url: str):
        self._api_key = api_key
        self._base_url = base_url

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        if isinstance(error, (asyncio.TimeoutError, APITimeoutError, APIConnectionError, RateLimitError)):
            return True

        if isinstance(error, APIStatusError):
            status_code = getattr(error, "status_code", None)
            return status_code in {408, 429, 500, 502, 503, 504}

        message = str(error).lower()
        return any(
            token in message
            for token in (
                "timeout",
                "timed out",
                "connection",
                "rate limit",
                "service unavailable",
                "server error",
            )
        )

    async def _call_model(
        self,
        *,
        model: str,
        messages: List[ChatCompletionMessageParam],
        temperature: float,
        max_tokens: int,
    ) -> str:
        client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url, timeout=LLM_TIMEOUT_SECONDS)

        extra_headers = {}
        if OPENROUTER_HTTP_REFERER:
            extra_headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
        if OPENROUTER_SITE_TITLE:
            extra_headers["X-Title"] = OPENROUTER_SITE_TITLE

        response = await client.chat.completions.create(
            model=model,
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

    async def _complete_with_model_retries(
        self,
        *,
        model: str,
        messages: List[ChatCompletionMessageParam],
        temperature: float,
        max_tokens: int,
        retry_attempts: int,
    ) -> Optional[str]:
        for attempt in range(retry_attempts + 1):
            try:
                logger.debug(
                    "Calling OpenRouter model=%s attempt=%d/%d",
                    model,
                    attempt + 1,
                    retry_attempts + 1,
                )
                return await self._call_model(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as error:
                logger.warning(
                    "OpenRouter model %s failed on attempt %d/%d: %s: %s",
                    model,
                    attempt + 1,
                    retry_attempts + 1,
                    type(error).__name__,
                    error,
                )
                if attempt < retry_attempts and self._is_retryable_error(error):
                    await asyncio.sleep(2**attempt)
                    continue
                return None

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> str:
        messages: List[ChatCompletionMessageParam] = []
        if system_prompt:
            system_message: ChatCompletionSystemMessageParam = {"role": "system", "content": system_prompt}
            messages.append(system_message)

        if history_messages:
            for msg in history_messages:
                content = str(msg.get("content", "")).strip()
                if content:
                    if msg.get("role") == "assistant":
                        assistant_message: ChatCompletionAssistantMessageParam = {
                            "role": "assistant",
                            "content": content,
                        }
                        messages.append(assistant_message)
                    elif msg.get("role") == "system":
                        system_message = {"role": "system", "content": content}
                        messages.append(system_message)
                    else:
                        user_message: ChatCompletionUserMessageParam = {"role": "user", "content": content}
                        messages.append(user_message)

        user_content = str(prompt).strip() if prompt else "Olá"
        user_message: ChatCompletionUserMessageParam = {"role": "user", "content": user_content}
        messages.append(user_message)

        try:
            primary_response = await self._complete_with_model_retries(
                model=LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                retry_attempts=LLM_PRIMARY_RETRIES,
            )

            if primary_response:
                return primary_response

            logger.warning("Primary model %s failed; trying fallback model %s", LLM_MODEL, LLM_FALLBACK_MODEL)

            fallback_response = await self._complete_with_model_retries(
                model=LLM_FALLBACK_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                retry_attempts=0,
            )

            if fallback_response:
                return fallback_response

        except Exception as e:
            logger.exception("Erro na chamada ao OpenRouter: %s", e)

        return "Tive um problema técnico. Por favor, tente perguntar de outra forma."
