import logging
from typing import Any, TypeVar, Type, Optional, overload

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

from backend.src.core.config.conversation import (
    RESPONSE_FALLBACK_MODEL,
    RESPONSE_MAX_TOKENS,
    RESPONSE_MODEL,
    RESPONSE_RETRIES_PER_MODEL,
    RESPONSE_TEMPERATURE,
)
from backend.src.core.config.infrastructure import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLM:
    """
    Handles interactions with LLMs, supporting primary and fallback models,
    automatic retries, and both raw text and structured (Pydantic) outputs.
    """

    def __init__(
        self,
        model: str = RESPONSE_MODEL,
        fallback_model: str = RESPONSE_FALLBACK_MODEL,
        temperature: float = RESPONSE_TEMPERATURE,
        max_tokens: int = RESPONSE_MAX_TOKENS,
        retries: int = RESPONSE_RETRIES_PER_MODEL,
    ):
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._retries = retries

        # Initialize the model chain (primary first, then fallback)
        self._llms = [self._create_llm(model), self._create_llm(fallback_model)]

    def _create_llm(self, model_name: str) -> ChatOpenAI:
        """Instantiates a ChatOpenAI client for a specific model."""
        return ChatOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            model=model_name,
            temperature=self._temperature,
            max_completion_tokens=self._max_tokens,
        )

    @overload
    async def _execute_with_fallbacks(self, messages: list[Any], schema: None = None) -> str: ...

    @overload
    async def _execute_with_fallbacks(self, messages: list[Any], schema: Type[T]) -> T: ...

    async def _execute_with_fallbacks(self, messages: list[Any], schema: Optional[Type[T]] = None) -> str | T:
        """Iterates through available models, returning the first successful response."""
        for llm in self._llms:
            result = await self._execute_with_retries(llm, messages, schema)
            if result is not None:
                return result

            logger.warning(f"Exhausted all retries for model '{llm.model_name}'. Moving to fallback.")

        logger.error("All models failed to generate a response.")
        raise RuntimeError("Retry attempts exhausted on all models.")

    async def _execute_with_retries(
        self, llm: ChatOpenAI, messages: list[BaseMessage], schema: Optional[Type[T]] = None
    ) -> str | T | None:
        """Attempts to generate a response with a specific model, retrying on failure."""
        for attempt in range(1, self._retries + 1):
            try:
                if schema:
                    structured_llm = llm.with_structured_output(schema)
                    result = await structured_llm.ainvoke(messages)

                    # Ensure we always return the Pydantic model instance
                    return schema(**result) if isinstance(result, dict) else result

                response = await llm.ainvoke(messages)
                return str(response.content).strip()

            except Exception as e:
                logger.warning(f"Model '{llm.model_name}' failed (Attempt {attempt}/{self._retries}). Error: {e}")

        return None

    async def generate_text(self, messages: list[BaseMessage]) -> str:
        """Generates a plain string response from the LLM."""
        return await self._execute_with_fallbacks(messages, schema=None)

    async def generate_structured(self, messages: list[BaseMessage], schema: Type[T]) -> T:
        """Generates a structured response matching the provided Pydantic schema."""
        return await self._execute_with_fallbacks(messages, schema=schema)
