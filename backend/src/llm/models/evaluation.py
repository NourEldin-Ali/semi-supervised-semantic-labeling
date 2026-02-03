from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class LabelEvaluation(BaseModel):
    """Structured scores for a single question/label pair."""

    id: str = Field(description="The id of the evaluated question.")
    method: str = Field(description="The method used to generate the labels.")
    correctness: float = Field(ge=1, le=5, description="Correctness score (1-5).")
    completeness: float = Field(ge=1, le=5, description="Completeness score (1-5).")
    generalization: float = Field(ge=1, le=5, description="Generalization score (1-5).")
    consistency: float = Field(ge=1, le=5, description="Consistency score (1-5).")
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


class PairwiseComparison(BaseModel):
    """Pairwise comparison for two label sets on the same question."""

    id: str = Field(description="The id of the evaluated question.")
    method1: str = Field(description="The first labeling method.")
    method2: str = Field(description="The second labeling method.")
    winner: Literal["method1", "method2", "tie"] = Field(
        description="Which method wins overall, or tie if equal."
    )
    reasoning: str = Field(description="Short justification for the winner.")
