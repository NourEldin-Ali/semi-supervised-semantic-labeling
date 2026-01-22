from typing import List
from pydantic import BaseModel, Field


class LabelsResult(BaseModel):
    """Structured output containing the labelled id and the extracted labels."""

    id: str = Field(description="The id used as input.")
    labels: List[str] = Field(description="Set of extracted labels.")
    
    def to_dict(self):
        """Convert the response to the mapping expected by downstream aggregators."""
        return {self.id: self.labels}
