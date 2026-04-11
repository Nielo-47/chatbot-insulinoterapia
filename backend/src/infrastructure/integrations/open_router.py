from typing import Dict, List, Optional

from openai import OpenAI

from backend.src.config import Config


async def call_openrouter(
	llm_api_key: str,
	llm_base_url: str,
	prompt: str,
	system_prompt: Optional[str] = None,
	history_messages: Optional[List[Dict[str, str]]] = None,
	temperature: float = 0.1,
	max_tokens: int = 800,
) -> str:
	client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)

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

		res_content = response.choices[0].message.content or ""
		return str(res_content).strip()

	except Exception as e:
		print(f"Erro na chamada ao OpenRouter: {e}")
		return "Tive um problema técnico. Por favor, tente perguntar de outra forma."
