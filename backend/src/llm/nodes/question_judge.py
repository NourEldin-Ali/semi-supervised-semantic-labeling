from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

from src.llm.models.question_judge import (
    QuestionPairwise,
    QuestionScore,
    QuestionSetPairwise,
    QuestionSetScore,
)
from src.llm.prompts.question_pairwise import question_pairwise_prompt
from src.llm.prompts.question_scoring import question_scoring_prompt
from src.llm.prompts.question_set_pairwise import question_set_pairwise_prompt
from src.llm.prompts.question_set_scoring import question_set_scoring_prompt

score_prompt_template = PromptTemplate(template=question_scoring_prompt)
pairwise_prompt_template = PromptTemplate(template=question_pairwise_prompt)
set_score_prompt_template = PromptTemplate(template=question_set_scoring_prompt)
set_pairwise_prompt_template = PromptTemplate(template=question_set_pairwise_prompt)


def score_question(
    user_need: str,
    question: str,
    llm: Any,
    config: Optional[dict] = None,
) -> QuestionScore:
    """LLM-as-judge: score a single question against a user need."""
    structured_llm = llm.with_structured_output(QuestionScore)
    prompt = score_prompt_template.format_prompt()
    system_msg = SystemMessage(content=prompt.to_string())
    human_need = HumanMessage(content=f"<user_need>{user_need}</user_need>")
    human_question = HumanMessage(content=f"<question>{question}</question>")
    invoke_kwargs = {} if config is None else {"config": config}
    reply: QuestionScore = structured_llm.invoke(
        [system_msg, human_need, human_question], **invoke_kwargs
    )
    return reply


def compare_questions(
    user_need: str,
    question_a: str,
    question_b: str,
    llm: Any,
    config: Optional[dict] = None,
) -> QuestionPairwise:
    """LLM-as-judge: compare two questions for a user need."""
    structured_llm = llm.with_structured_output(QuestionPairwise)
    prompt = pairwise_prompt_template.format_prompt()
    system_msg = SystemMessage(content=prompt.to_string())
    human_need = HumanMessage(content=f"<user_need>{user_need}</user_need>")
    human_question_a = HumanMessage(content=f"<question group='A'>{question_a}</question>")
    human_question_b = HumanMessage(content=f"<question group='B'>{question_b}</question>")
    invoke_kwargs = {} if config is None else {"config": config}
    reply: QuestionPairwise = structured_llm.invoke(
        [system_msg, human_need, human_question_a, human_question_b], **invoke_kwargs
    )
    return reply


def _format_questions(questions: list[str]) -> str:
    if not questions:
        return ""
    return "\n".join(f"- {q}" for q in questions)


def score_question_set(
    user_need: str,
    questions: list[str],
    llm: Any,
    config: Optional[dict] = None,
) -> QuestionSetScore:
    """LLM-as-judge: score a set of questions against a user need."""
    structured_llm = llm.with_structured_output(QuestionSetScore)
    prompt = set_score_prompt_template.format_prompt()
    system_msg = SystemMessage(content=prompt.to_string())
    human_need = HumanMessage(content=f"<user_need>{user_need}</user_need>")
    human_questions = HumanMessage(
        content=f"<questions>\n{_format_questions(questions)}\n</questions>"
    )
    invoke_kwargs = {} if config is None else {"config": config}
    reply: QuestionSetScore = structured_llm.invoke(
        [system_msg, human_need, human_questions], **invoke_kwargs
    )
    return reply


def compare_question_sets(
    user_need: str,
    questions_a: list[str],
    questions_b: list[str],
    llm: Any,
    config: Optional[dict] = None,
) -> QuestionSetPairwise:
    """LLM-as-judge: compare two question sets for a user need."""
    structured_llm = llm.with_structured_output(QuestionSetPairwise)
    prompt = set_pairwise_prompt_template.format_prompt()
    system_msg = SystemMessage(content=prompt.to_string())
    human_need = HumanMessage(content=f"<user_need>{user_need}</user_need>")
    human_questions_a = HumanMessage(
        content=f"<questions group='A'>\n{_format_questions(questions_a)}\n</questions>"
    )
    human_questions_b = HumanMessage(
        content=f"<questions group='B'>\n{_format_questions(questions_b)}\n</questions>"
    )
    invoke_kwargs = {} if config is None else {"config": config}
    reply: QuestionSetPairwise = structured_llm.invoke(
        [system_msg, human_need, human_questions_a, human_questions_b], **invoke_kwargs
    )
    return reply
