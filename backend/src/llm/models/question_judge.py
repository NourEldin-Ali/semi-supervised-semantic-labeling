from typing import Literal

from pydantic import BaseModel, Field


class QuestionScore(BaseModel):
    """Score a single question against a user need."""

    score: int = Field(ge=0, le=100, description="Overall score from 0 to 100.")
    reasoning: str = Field(description="Brief justification for the score.")


class QuestionPairwise(BaseModel):
    """Pairwise comparison between two questions."""

    score_a: int = Field(ge=0, le=100, description="Score for question A (0-100).")
    score_b: int = Field(ge=0, le=100, description="Score for question B (0-100).")
    winner: Literal["A", "B", "tie"] = Field(
        description="Which question is better overall, or tie if equal."
    )
    reasoning: str = Field(description="Brief justification for the decision.")


class QuestionSetScore(BaseModel):
    """Score a set of questions against a user need."""

    score: int = Field(ge=0, le=100, description="Overall set score from 0 to 100.")
    reasoning: str = Field(description="Brief justification for the score.")


class QuestionSetPairwise(BaseModel):
    """Pairwise comparison between two question sets."""

    score_a: int = Field(ge=0, le=100, description="Score for question set A (0-100).")
    score_b: int = Field(ge=0, le=100, description="Score for question set B (0-100).")
    winner: Literal["A", "B", "tie"] = Field(
        description="Which question set is better overall, or tie if equal."
    )
    reasoning: str = Field(description="Brief justification for the decision.")
