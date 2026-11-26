import json
import logging
import re
from typing import Any, Callable, Coroutine, List, Optional

from backend.src.config.conversation import SUGGESTIONS_MAX_TOKENS, SUGGESTIONS_TEMPERATURE
from backend.src.config.prompts import SUGGESTIONS_PROMPT

logger = logging.getLogger(__name__)

_MAX_QUESTIONS = 3
_MAX_QUESTION_LENGTH = 120

_LEADING_MARKER = re.compile(r"^[\s\-•*\d.)]+")


def _strip_code_fence(text: str) -> str:
    """Remove optional ```json ... ``` wrapper from LLM output."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _normalize_questions(raw: Any) -> List[str]:
    """Coerce raw LLM output into a clean list of at most 3 short questions."""
    if not isinstance(raw, list):
        return []

    questions: List[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        # Strip leading numbering or bullets the model may have added.
        text = _LEADING_MARKER.sub("", text).strip()
        if not text or text in seen:
            continue
        if len(text) > _MAX_QUESTION_LENGTH:
            text = text[:_MAX_QUESTION_LENGTH - 1].rstrip() + "…"
        seen.add(text)
        questions.append(text)
        if len(questions) >= _MAX_QUESTIONS:
            break
    return questions


class SuggestionService:
    def __init__(self, call_llm: Callable[..., Coroutine[Any, Any, str]]):
        self._call_llm = call_llm

    async def generate(
        self,
        query: str,
        response: str,
        history_messages: Optional[List[dict]] = None,
    ) -> List[str]:
        prompt = SUGGESTIONS_PROMPT.format(query=query, response=response)
        raw = await self._call_llm(
            prompt=prompt,
            history_messages=history_messages,
            temperature=SUGGESTIONS_TEMPERATURE,
            max_tokens=SUGGESTIONS_MAX_TOKENS,
        )

        try:
            payload = json.loads(_strip_code_fence(raw))
        except (json.JSONDecodeError, TypeError):
            logger.warning("Suggestions parse failed — returning empty list", exc_info=True)
            return []

        if isinstance(payload, dict):
            return _normalize_questions(payload.get("questions"))
        return _normalize_questions(payload)
