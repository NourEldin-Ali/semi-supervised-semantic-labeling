from typing import Any, List, Optional

from langgraph.graph import END, START, StateGraph

from config.llm_config import LLMConnector
from enums.llm_type import LLMType
from src.llm.models.labeled_question import QuestionLabels
from src.llm.nodes.llm_evaluator import check_existing_questions, llm_as_judge
from src.llm.state.evaluation_state_management import EvalState


def _make_eval_node(llm):
    """Node that forwards state and config to llm_as_judge for callback/token tracking."""

    def node(state: EvalState, config: Optional[dict] = None, **_kwargs: Any) -> dict:
        return llm_as_judge(state, llm, config=config)

    return node


class EvaluationOrchestrator:
    """LangGraph-based orchestrator that evaluates labels using LLM as judge."""

    def __init__(
        self,
        model_name: str = "gpt-5.2-2025-12-11",
        model_type=LLMType.OPEN_AI,
        api_key: Optional[str] = None,
    ):
        llm = LLMConnector(
            model_name=model_name,
            llm_type=model_type,
            api_key=api_key,
        )()

        builder = StateGraph(state_schema=EvalState)
        builder.add_node("evaluate", _make_eval_node(llm))

        builder.add_edge(START, "evaluate")
        builder.add_conditional_edges(
            "evaluate",
            lambda input_state: check_existing_questions(input_state),
            {"evaluate": "evaluate", "end": END},
        )

        self.graph = builder.compile()

    def invoke(
        self,
        questions_method_1: List[QuestionLabels],
        questions_method_2: List[QuestionLabels],
        config: Optional[dict] = None,
    ) -> List:
        """Invoke the orchestrator on two lists of questions. Optional config (e.g. callbacks) is passed to the graph."""
        initial_state = {
            "questions_method_1": questions_method_1.copy(),
            "questions_method_2": questions_method_2.copy(),
            "results": [],
        }
        invoke_kwargs: dict = {}
        if config is not None:
            invoke_kwargs["config"] = config
        final_state = self.graph.invoke(initial_state, **invoke_kwargs)
        return final_state["results"]