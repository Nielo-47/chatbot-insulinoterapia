from langchain_openai import ChatOpenAI
from typing import Any, TypeVar, Type, cast
from pydantic import BaseModel

from backend.src.core.config.infrastructure import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

T = TypeVar("T", bound=BaseModel)


class LLM:
    def __init__(self, model: str, temperature: float, max_tokens: int):
        self._llm = ChatOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )

    async def generate_text(self, messages: list) -> str:
        response = await self._llm.ainvoke(messages)
        return response.text.strip()

    async def generate_structured(self, messages: list, schema: Type[T]) -> T:
        structured_llm = self._llm.with_structured_output(schema)
        structured_response = await structured_llm.ainvoke(messages)
        return cast(T, structured_response)
