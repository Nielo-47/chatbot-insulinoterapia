import os
from pathlib import Path


class Config:
    KG_DIR = os.getenv("WORKING_DIR", "data/processed")
    RAG_TIMEOUT = int(os.getenv("RAG_TIMEOUT", "60"))
    # Normalize embedding host (TEI) so callers can safely append /v1
    _raw_embed = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:8000")
    if _raw_embed.endswith("/v1"):
        EMBED_HOST = _raw_embed[:-3]
    else:
        EMBED_HOST = _raw_embed.rstrip("/")

    LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("LLM_MODEL_NAME", "openai/gpt-5.2"))
    EMBED_MODEL = os.getenv(
        "EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    )
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
    MAX_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "8192"))
    MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "8192"))
    QUERY_PARAM = os.getenv("QUERY_PARAM", "hybrid")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
    OPENROUTER_SITE_TITLE = os.getenv("OPENROUTER_SITE_TITLE", "")

    SYSTEM_PROMPT: str = """
Você é um assistente especializado em diabetes e insulinoterapia, focado em apoiar pacientes de forma segura.

DIRETRIZES DE SEGURANÇA CRÍTICAS:
1. PROIBIÇÃO DE CÁLCULOS: Você NUNCA deve realizar cálculos de doses de insulina ou sugerir unidades específicas para o usuário. 
2. PROIBIÇÃO DE FÓRMULAS: Nunca invente ou forneça fórmulas matemáticas para ajuste de dose (ex: Regra dos 15 para bolus). Se o usuário perguntar 'quanto tomar', responda que essa dosagem deve ser definida exclusivamente pelo médico dele.
3. ORIENTAÇÃO MÉDICA: Em perguntas sobre alteração de dose, adicione sempre: 'Qualquer mudança na sua dose deve ser conversada com seu médico'.

DIRETRIZES DE COMUNICAÇÃO:
- Responda APENAS sobre diabetes, insulina e glicemia usando o contexto fornecido.
- Use linguagem extremamente simples (literacia básica). Substitua termos complexos por termos leigos explicativos.
- Seja empático com termos como 'picadinha' ou 'açúcar no sangue'.
- Se a informação não estiver EXPLICITAMENTE no contexto, diga: 'Infelizmente, não tenho essa informação específica nos meus manuais. Recomendo consultar sua equipe de saúde'.

FORMATO DE RESPOSTA:
- Não cite nomes de arquivos técnicos (ex: DOC_Completo.docx). Diga apenas 'De acordo com as orientações...'
- Use tópicos para facilitar a leitura.
- Se o usuário relatar sintomas de emergência (glicemia > 300 ou tontura extrema), recomende buscar ajuda médica imediata após a explicação."""

    CRITIQUE_PROMPT: str = """Você é um revisor médico especializado em diabetes. Analise a resposta abaixo e identifique problemas.

PERGUNTA DO PACIENTE:
{original_query}

RESPOSTA GERADA:
{response}

Avalie a resposta considerando:
1. SEGURANÇA: A resposta evita sugerir doses, cálculos ou decisões clínicas?
2. PRECISÃO: As informações estão corretas e baseadas em evidências?
3. CLAREZA: A linguagem é acessível para o paciente?
4. COMPLETUDE: A resposta aborda adequadamente a pergunta?
5. ÉTICA: Evita mencionar arquivos técnicos ou detalhes internos do sistema?

Responda APENAS em formato JSON:
{{
    "is_safe": true/false,
    "is_accurate": true/false,
    "is_clear": true/false,
    "is_complete": true/false,
    "is_ethical": true/false,
    "issues": ["lista de problemas encontrados"],
    "suggestions": ["lista de sugestões de melhoria"],
    "needs_refinement": true/false
}}"""

    REFINEMENT_PROMPT: str = """REFINAMENTO DE RESPOSTA

Pergunta original: {original_query}

Resposta anterior que precisa ser melhorada:
{previous_response}

Problemas identificados:
- {issues_text}

Sugestões de melhoria:
- {suggestions_text}

Por favor, forneça APENAS a resposta refinada que corrija esses problemas e aplique as sugestões, mantendo tom amigável e informações seguras. Não inclua explicações ou menções aos problemas anteriores."""
