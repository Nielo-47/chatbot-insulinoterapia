from langgraph.graph import StateGraph
from backend.src.features.chat.state import QueryGraphState


class QueryGraphBuilder:
    @staticmethod
    def build() -> StateGraph:
        g = StateGraph(QueryGraphState)

        return g
