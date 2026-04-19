from backend.src.config.prompts import RAG_FAILURE_RESPONSE
from backend.src.infrastructure.rag.client import RAGRuntime
from lightrag.prompt import PROMPTS


class RAGFactory:
    @staticmethod
    def create() -> RAGRuntime:
        runtime = RAGRuntime()

        # Keep fail response policy in config and apply at composition time.
        PROMPTS["fail_response"] = RAG_FAILURE_RESPONSE

        return runtime
