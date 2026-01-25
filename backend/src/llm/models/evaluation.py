from typing import List, Optional

from pydantic import BaseModel, Field


class LabelEvaluation(BaseModel):
    """Structured scores for a single question/label pair."""

    id: str = Field(description="The id of the evaluated question.")
    method: str = Field(description="The method used to generate the labels.")
    relevance: float = Field(ge=1, le=5, description="Relevance score (1-5).")
    correctness: float = Field(ge=1, le=5, description="Correctness score (1-5).")
    coverage: float = Field(ge=1, le=5, description="Coverage (key concepts) score (1-5).")
    taxonomy_fit_granularity: float = Field(
        ge=1,
        le=5,
        description="Taxonomy fit & granularity score (1-5).",
    )
    actionability: float = Field(ge=1, le=5, description="Actionability score (1-5).")
    reasoning: str = Field(description="Short justification for the scores.")
    labels_considered: Optional[List[str]] = Field(
        default=None,
        description="Labels that were evaluated (copied back for traceability).",
    )

class EvaluationResult(BaseModel):
    """Evaluation result for a question with multiple labeling methods."""

    id: str = Field(description="The id of the evaluated question.")
    evaluations: List[LabelEvaluation] = Field(
        min_length=2,
        max_length=2,
        description="Exactly two label evaluations, one per method (method1, then method2).",
    )
