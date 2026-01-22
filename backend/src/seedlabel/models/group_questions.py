from typing import List
from pydantic import BaseModel, Field


class GroupQuestion(BaseModel):
    """Domain model describing a group of semantically related questions."""

    id: str = Field(description="The group id.")
    questions: List[str] = Field(description="Set of questions.")

    def get_in_xml(self) -> str:
        """Render the grouped questions in the XML envelope expected by the prompts."""
        qs = "".join(f"<question>{q}</question>" for q in self.questions)
        return f"<group><id>{self.id}</id><questions>{qs}</questions></group>"
