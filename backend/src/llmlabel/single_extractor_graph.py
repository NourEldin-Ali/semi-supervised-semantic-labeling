from typing import Any, List, Optional

from langgraph.graph import END, START, StateGraph

from config.llm_config import LLMConnector
from enums.llm_type import LLMType
from src.llm.state.state_management import RootState
from src.llmlabel.models.single_question import SingleQuestion
from src.llmlabel.nodes.llm_extractor_single import (
    check_existing_questions,
    extract_labels_from_a_question,
)


def _make_single_label_node(llm):
    def node(state: RootState, config: Optional[dict] = None, **_kwargs: Any) -> dict:
        return extract_labels_from_a_question(state, llm, config=config)
    return node


class SingleOrchestrator:
    """LangGraph-based orchestrator that labels one question at a time."""
    def __init__(
        self,
        model_name: str = "gpt-5.2-2025-12-11",
        model_type=LLMType.OPEN_AI,
        api_key: Optional[str] = None,
    ):
        llm = LLMConnector(model_name=model_name, llm_type=model_type, api_key=api_key)()

        builder = StateGraph(state_schema=RootState)
        builder.add_node("get_label", _make_single_label_node(llm))
        builder.add_edge(START, "get_label")
        builder.add_conditional_edges(
            "get_label",
            lambda input_state: check_existing_questions(input_state),
            {"get_label": "get_label", "end": END},
        )
        self.graph = builder.compile()

    def invoke(
        self,
        questions: List[SingleQuestion],
        config: Optional[dict] = None,
    ) -> List:
        """Invoke the orchestrator on a list of questions. Optional config (e.g. callbacks) is passed to the graph."""
        initial_state = {"questions": questions, "labels": []}
        kwargs = {}
        if config is not None:
            kwargs["config"] = config
        final_state = self.graph.invoke(initial_state, **kwargs)
        return final_state["labels"]