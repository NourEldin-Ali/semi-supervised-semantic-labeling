
from typing import Annotated, TypedDict

from src.llm.utils.list_and_map import add_if_not_exists
from src.llmlabel.models.single_question import SingleQuestion
from src.seedlabel.models.group_questions import GroupQuestion
from src.llm.models.label_result import LabelsResult


class RootState(TypedDict):
    """State contract shared between LangGraph nodes handling single/grouped questions."""
    
    questions: list[GroupQuestion | SingleQuestion]
    labels:   Annotated[list[LabelsResult], add_if_not_exists]
