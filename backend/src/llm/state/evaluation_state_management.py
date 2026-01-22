
from typing import Annotated, TypedDict

from src.llm.models.evaluation import EvaluationResult
from src.llm.models.labeled_question import QuestionLabels
from src.llm.utils.list_and_map import add_evaluation_results


class EvalState(TypedDict):
    """State contract shared between LangGraph nodes evaluating labels."""
    
    questions_method_1: list[QuestionLabels]
    questions_method_2: list[QuestionLabels]
    results: Annotated[list[EvaluationResult], add_evaluation_results]