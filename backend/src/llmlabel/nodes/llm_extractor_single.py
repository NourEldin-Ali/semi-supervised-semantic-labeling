



from src.llm.models.label_result import LabelsResult
from src.llmlabel.models.single_question import SingleQuestion
from src.llm.state.state_management import RootState
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from src.llmlabel.prompts.single_labeling import single_labeling_prompt

prompt_template = PromptTemplate(template=single_labeling_prompt)


def extract_labels_from_a_question(state: RootState, llm, config=None):
    questions = state.get("questions") or []
    if not questions:
        raise ValueError("No question available")
    question: SingleQuestion = questions.pop()

    structured_llm = llm.with_structured_output(LabelsResult)
    prompt = prompt_template.format_prompt()
    system_msg = SystemMessage(content=prompt.to_string())
    human_msg = HumanMessage(content=question.get_in_xml())

    invoke_kwargs = {}
    if config is not None:
        invoke_kwargs["config"] = config
    try:
        reply: LabelsResult = structured_llm.invoke([system_msg, human_msg], **invoke_kwargs)
    except Exception as exc:  # pragma: no cover - defensive logging
        raise

    return {
        "questions": questions,
        "labels":  [reply]
    }


def check_existing_questions(state: RootState) -> str:
    """Signal to LangGraph whether more questions remain in the queue."""
    remaining = len(state.get("questions") or [])
    return "end" if remaining == 0 else "get_label"