from typing import Literal, Optional

from pydantic import BaseModel, Field


class PairwiseJudgment(BaseModel):
    """Structured decision for a pairwise label set comparison."""

    decision: Literal["A is better", "B is better", "Tie"] = Field(
        description='Pairwise decision: "A is better", "B is better", or "Tie".'
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Short justification for the decision.",
    )


class AbsoluteScore(BaseModel):
    """Structured absolute quality scores for a single label set."""

    correctness: float = Field(ge=1, le=5, description="Correctness score (1-5).")
    completeness: float = Field(ge=1, le=5, description="Completeness score (1-5).")
    clarity: float = Field(ge=1, le=5, description="Clarity score (1-5).")
    faithfulness: float = Field(ge=1, le=5, description="Faithfulness score (1-5).")
    reasoning: Optional[str] = Field(
        default=None,
        description="Short justification for the scores.",
    )

