from backend.src.config.prompts import RAG_FAILURE_RESPONSE
from backend.src.infrastructure.llm.client import LLMClient
from backend.src.infrastructure.rag.client import RAGRuntime
from lightrag.prompt import PROMPTS


class RAGFactory:
    @staticmethod
    def create(llm_client: LLMClient, embed_host: str, embed_api_key: str) -> RAGRuntime:
        runtime = RAGRuntime(embed_api_key, f"{embed_host}/v1")

        # Keep fail response policy in config and apply at composition time.
        PROMPTS["fail_response"] = RAG_FAILURE_RESPONSE

        return runtime
