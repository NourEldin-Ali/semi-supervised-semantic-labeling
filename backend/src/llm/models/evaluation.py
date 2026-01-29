from typing import List, Optional

from pydantic import BaseModel, Field


class LabelEvaluation(BaseModel):
    """Structured scores for a single question/label pair."""

    id: str = Field(description="The id of the evaluated question.")
    method: str = Field(description="The method used to generate the labels.")
    intent_alignment_score: float = Field(
        ge=0,
        le=5,
        description="Intent Alignment Score (IAS) (0-5).",
    )
    concept_completeness_score: float = Field(
        ge=0,
        le=5,
        description="Concept Completeness Score (CCS) (0-5).",
    )
    noise_redundancy_penalty: float = Field(
        ge=0,
        le=5,
        description="Noise & Redundancy Penalty (NRP) (0-5).",
    )
    terminology_normalization_score: float = Field(
        ge=0,
        le=5,
        description="Terminology Normalization Score (TNS) (0-5).",
    )
    audit_usefulness_score: float = Field(
        ge=0,
        le=5,
        description="Audit Usefulness Score (AUS) (0-5).",
    )
    control_mapping_clarity_score: float = Field(
        ge=0,
        le=5,
        description="Control-Mapping Clarity Score (CMCS) (0-5).",
    )
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
