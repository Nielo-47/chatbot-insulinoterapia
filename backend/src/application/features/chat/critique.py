import json
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

from backend.src.config.conversation import CRITIQUE_MAX_TOKENS, CRITIQUE_TEMPERATURE
from backend.src.config.prompts import CRITIQUE_PROMPT

logger = logging.getLogger(__name__)

_FALLBACK_CRITIQUE: Dict[str, Any] = {
    "is_safe": True,
    "is_accurate": True,
    "is_clear": True,
    "is_complete": True,
    "is_ethical": True,
    "issues": [],
    "suggestions": [],
    "needs_refinement": False,
}


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


class CritiqueService:
    def __init__(self, call_llm: Callable[..., Coroutine[Any, Any, str]]):
        self._call_llm = call_llm

    async def critique_response(
        self,
        original_query: str,
        response: str,
        history_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        raw = await self._call_llm(
            prompt=CRITIQUE_PROMPT.format(original_query=original_query, response=response),
            history_messages=history_messages,
            temperature=CRITIQUE_TEMPERATURE,
            max_tokens=CRITIQUE_MAX_TOKENS,
        )

        try:
            return json.loads(_strip_code_fence(raw))
        except json.JSONDecodeError:
            logger.warning("Critique parse failed — using fallback", exc_info=True)
            return _FALLBACK_CRITIQUE

    @staticmethod
    def build_refinement_query(
        original_query: str,
        previous_response: str,
        critique: Dict[str, Any],
    ) -> str:
        from backend.src.config.prompts import REFINEMENT_PROMPT

        issues = "\n- ".join(critique.get("issues") or ["No issues identified"])
        suggestions = "\n- ".join(critique.get("suggestions") or ["Maintain current quality"])
        return REFINEMENT_PROMPT.format(
            original_query=original_query,
            previous_response=previous_response,
            issues_text=issues,
            suggestions_text=suggestions,
        )