import logging
from typing import Any, Callable, Coroutine, Dict, List

from backend.src.application.contracts.chat import ConversationServiceContract
from backend.src.config.conversation import SUMMARIZER_MAX_TOKENS, SUMMARIZER_TEMPERATURE
from backend.src.config.prompts import SUMMARY_PROMPT, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class SummarizationService:
    def __init__(
        self,
        conversation_service: ConversationServiceContract,
        call_llm: Callable[..., Coroutine[Any, Any, str]],
    ):
        self._conversation_service = conversation_service
        self._call_llm = call_llm

    async def summarize_and_trim(
        self,
        user_id: int,
        history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Summarizes all but the last 2 messages, persists the summary,
        and returns the trimmed history. Returns {} on failure.
        """
        messages_to_summarize = history[:-2]
        if not messages_to_summarize:
            return {}

        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content'].strip()}"
            for m in messages_to_summarize
            if m.get("content", "").strip()
        )

        try:
            summary = await self._call_llm(
                prompt=SUMMARY_PROMPT.format(history=history_text),
                system_prompt=SYSTEM_PROMPT,
                temperature=SUMMARIZER_TEMPERATURE,
                max_tokens=SUMMARIZER_MAX_TOKENS,
            )
            summary = summary.strip()
            if not summary:
                return {}

            self._conversation_service.store_summary(user_id, summary)
            return {
                "summary": summary,
                "conversation_history": history[-2:],
                "was_summarized": True,
            }
        except Exception:
            logger.warning("Summarization failed", exc_info=True)
            return {}
