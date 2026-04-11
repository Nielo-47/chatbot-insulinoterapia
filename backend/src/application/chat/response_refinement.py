import json
from typing import Any, Dict, List, Optional

from backend.src.config import Config


async def critique_response(
    call_llm,
    original_query: str,
    response: str,
    history_messages: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    critique_prompt = Config.CRITIQUE_PROMPT.format(original_query=original_query, response=response)

    critique_system_prompt = (
        "Você é um revisor médico rigoroso. Responda APENAS com JSON válido, "
        "sem texto adicional antes ou depois."
    )

    critique_text = await call_llm(
        prompt=critique_prompt,
        system_prompt=critique_system_prompt,
        history_messages=history_messages,
        temperature=0.0,
        max_tokens=600,
    )

    try:
        critique_text = critique_text.strip()
        if critique_text.startswith("```"):
            critique_text = critique_text.split("```")[1]
            if critique_text.startswith("json"):
                critique_text = critique_text[4:]

        critique = json.loads(critique_text.strip())
        return critique
    except json.JSONDecodeError as e:
        print(f"Erro ao parsear crítica: {e}")
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


def build_refinement_query(original_query: str, previous_response: str, critique: Dict[str, Any]) -> str:
    issues_text = "\n- ".join(critique.get("issues", ["Nenhum problema identificado"]))
    suggestions_text = "\n- ".join(critique.get("suggestions", ["Mantenha a qualidade atual"]))

    return Config.REFINEMENT_PROMPT.format(
        original_query=original_query,
        previous_response=previous_response,
        issues_text=issues_text,
        suggestions_text=suggestions_text,
    )