



from typing import Any, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

from src.llm.models.evaluation import EvaluationResult
from src.llm.models.labeled_question import QuestionLabels
from src.llm.prompts.evaluation import label_evaluation_prompt
from src.llm.state.evaluation_state_management import EvalState

prompt_template = PromptTemplate(template=label_evaluation_prompt)


def evaluate_one_pair(
    q1: QuestionLabels,
    q2: QuestionLabels,
    llm: Any,
    config: Optional[dict] = None,
) -> EvaluationResult:
    """Run LLM-as-judge for a single (method1, method2) pair. No graph state."""
    structured_llm = llm.with_structured_output(EvaluationResult)
    prompt = prompt_template.format_prompt()
    system_msg = SystemMessage(content=prompt.to_string())
    human_question = HumanMessage(content=f"<question>{q1.question}</question>")
    human_msg_1 = HumanMessage(content=q1.to_question_label())
    human_msg_2 = HumanMessage(content=q2.to_question_label())
    invoke_kwargs = {} if config is None else {"config": config}
    reply: EvaluationResult = structured_llm.invoke(
        [system_msg, human_question, human_msg_1, human_msg_2], **invoke_kwargs
    )
    reply.id = q1.id
    return reply


def llm_as_judge(
    state: EvalState,
    llm: Any,
    config: Optional[dict] = None,
) -> dict:
    """LangGraph node: process one pair and return state update. Prefer evaluate_one_pair + loop."""
    questions_method_1 = state.get("questions_method_1") or []
    questions_method_2 = state.get("questions_method_2") or []
    if not questions_method_1 or not questions_method_2:
        raise ValueError("No question available")
    q1, q2 = questions_method_1[0], questions_method_2[0]
    reply = evaluate_one_pair(q1, q2, llm, config=config)
    return {
        "questions_method_1": questions_method_1[1:],
        "questions_method_2": questions_method_2[1:],
        "results": [reply],
    }


def check_existing_questions(state: EvalState) -> str:
    """Signal to LangGraph whether more questions remain in the queue."""
    questions_method_1 = state.get("questions_method_1") or []
    questions_method_2 = state.get("questions_method_2") or []
    remaining = min(len(questions_method_1), len(questions_method_2))
    return "end" if remaining == 0 else "evaluate"