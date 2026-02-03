



from typing import Any, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

from src.llm.models.evaluation import EvaluationResult, LabelEvaluation, PairwiseComparison
from src.llm.models.labeled_question import QuestionLabels
from src.llm.prompts.evaluation import label_evaluation_prompt
from src.llm.prompts.pairwise_comparison import pairwise_comparison_prompt
from src.llm.state.evaluation_state_management import EvalState

prompt_template = PromptTemplate(template=label_evaluation_prompt)
pairwise_prompt_template = PromptTemplate(template=pairwise_comparison_prompt)


def evaluate_single_question(
    question_labels: QuestionLabels,
    llm: Any,
    config: Optional[dict] = None,
) -> LabelEvaluation:
    """Run LLM-as-judge for a single question/label set. No graph state."""
    structured_llm = llm.with_structured_output(LabelEvaluation)
    prompt = prompt_template.format_prompt()
    system_msg = SystemMessage(content=prompt.to_string())
    human_question = HumanMessage(
        content=f"<question id='{question_labels.id}'>{question_labels.question}</question>"
    )
    labels_text = ", ".join(question_labels.labels)
    human_labels = HumanMessage(
        content=f"<labels method='{question_labels.method}'>{labels_text}</labels>"
    )
    invoke_kwargs = {} if config is None else {"config": config}
    reply: LabelEvaluation = structured_llm.invoke(
        [system_msg, human_question, human_labels], **invoke_kwargs
    )
    reply.id = question_labels.id
    reply.method = question_labels.method
    if not reply.labels_considered:
        reply.labels_considered = question_labels.labels
    return reply


def evaluate_one_pair(
    q1: QuestionLabels,
    q2: QuestionLabels,
    llm: Any,
    config: Optional[dict] = None,
) -> EvaluationResult:
    """Run LLM-as-judge for a single (method1, method2) pair using per-question calls."""
    eval1 = evaluate_single_question(q1, llm, config=config)
    eval2 = evaluate_single_question(q2, llm, config=config)
    return EvaluationResult(id=q1.id, evaluations=[eval1, eval2])


def evaluate_single_question_twice(
    question_labels: QuestionLabels,
    llm: Any,
    config: Optional[dict] = None,
) -> List[LabelEvaluation]:
    """Test helper: run two independent evaluations for the same question/labels."""
    first = evaluate_single_question(question_labels, llm, config=config)
    second = evaluate_single_question(question_labels, llm, config=config)
    return [first, second]


def evaluate_pairwise_winner(
    q1: QuestionLabels,
    q2: QuestionLabels,
    llm: Any,
    config: Optional[dict] = None,
) -> PairwiseComparison:
    """Run LLM-as-judge comparing two label sets at once."""
    prompt = pairwise_prompt_template.format_prompt()
    system_msg = SystemMessage(content=prompt.to_string())
    human_question = HumanMessage(content=f"<question id='{q1.id}'>{q1.question}</question>")
    labels_1 = ", ".join(q1.labels)
    labels_2 = ", ".join(q2.labels)
    human_msg_1 = HumanMessage(
        content=f"<labels group='A' method='{q1.method}'>{labels_1}</labels>"
    )
    human_msg_2 = HumanMessage(
        content=f"<labels group='B' method='{q2.method}'>{labels_2}</labels>"
    )
    invoke_kwargs = {} if config is None else {"config": config}
    response = llm.invoke([system_msg, human_question, human_msg_1, human_msg_2], **invoke_kwargs)
    content = getattr(response, "content", None)
    if content is None:
        content = str(response)

    winner = "tie"
    explanation = ""
    for line in str(content).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("winner:"):
            value = stripped.split(":", 1)[1].strip().lower()
            value = value.replace(".", "").replace("(", " ").replace(")", " ").strip()
            if value in ("group a", "a", "groupa"):
                winner = "method1"
            elif value in ("group b", "b", "groupb"):
                winner = "method2"
            elif value in ("tie", "draw", "equal"):
                winner = "tie"
            else:
                if "group a" in value and "group b" in value:
                    winner = "tie"
                elif "group a" in value:
                    winner = "method1"
                elif "group b" in value:
                    winner = "method2"
                elif "tie" in value:
                    winner = "tie"
        elif lower.startswith("explanation:"):
            explanation = stripped.split(":", 1)[1].strip()

    if not explanation:
        # Fallback: use full content as reasoning if strict format is missing.
        explanation = str(content).strip()

    return PairwiseComparison(
        id=q1.id,
        method1=q1.method,
        method2=q2.method,
        winner=winner,
        reasoning=explanation,
    )


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
