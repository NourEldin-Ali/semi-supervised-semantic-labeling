from pydantic import BaseModel, Field


class SingleQuestion(BaseModel):
    """Domain model describing a single audit question."""

    id: str = Field(description="The id of the question.")
    question: str = Field(description="The detail of question.")

    def get_in_xml(self) -> str:
        """Render the question in a simple XML envelope expected by the prompts."""
        return f"<question><id>{self.id}</id><detail>{self.question}</detail></question>"
