from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import numpy as np

from enums.embed_type import EmbedType
from src.embedding.embed import Embedding
from src.select_question.utils import cosine_similarity, load_questions


def _top_indices(scores: np.ndarray, top_k: int) -> List[int]:
    if top_k <= 0:
        return []
    top_k = min(top_k, len(scores))
    return [int(i) for i in np.argsort(-scores)[:top_k]]


def select_questions_by_label_embedding(
    user_need: str,
    csv_input: str | Path | pd.DataFrame,
    embedding_model: str,
    embed_type: EmbedType | str = EmbedType.OPEN_AI,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    text_column: str = "text",
    id_column: str = "id",
    label_column: str = "labels",
    batch_size: int = 32,
    top_k_labels: int = 5,
    top_k_questions: int = 5,
) -> List[Dict[str, Any]]:
    """
    Select top questions by embedding labels and matching them to user_need.

    Steps:
    1) Embed unique labels.
    2) Find top labels similar to user_need.
    3) Return questions whose labels match the top labels.

    Returns a list of dicts: {id, question, score, labels, matched_labels}
    """
    if not user_need or not str(user_need).strip():
        raise ValueError("user_need must be a non-empty string.")
    user_need = str(user_need).strip()
    if top_k_questions <= 0:
        return []

    records = load_questions(
        csv_input=csv_input,
        text_column=text_column,
        id_column=id_column,
        label_column=label_column,
    )

    all_labels: List[str] = []
    for rec in records:
        for label in rec.labels:
            if label and label not in all_labels:
                all_labels.append(label)

    if not all_labels:
        raise ValueError(
            f"No labels found in column '{label_column}'. "
            "Provide a CSV with labels or specify the correct label_column."
        )

    if isinstance(embed_type, str):
        embed_type = EmbedType.from_string(embed_type)

    embedding_instance = Embedding(
        model_name=embedding_model,
        embed_type=embed_type,
        api_key=api_key,
        endpoint=endpoint,
    )

    query_vec, _ = embedding_instance.get_embedding([user_need], batch_size=1)
    label_vecs, _ = embedding_instance.get_embedding(all_labels, batch_size=batch_size)

    label_scores = cosine_similarity(query_vec[0], label_vecs)
    label_score_map = {label: float(label_scores[i]) for i, label in enumerate(all_labels)}

    top_label_indices = _top_indices(label_scores, top_k_labels)
    top_labels = {all_labels[i] for i in top_label_indices}

    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for rec in records:
        if not rec.labels:
            continue
        matched = [lbl for lbl in rec.labels if lbl in top_labels]
        if not matched:
            continue
        score = max(label_score_map.get(lbl, -1.0) for lbl in matched)
        candidates.append(
            (
                score,
                {
                    "id": rec.id,
                    "question": rec.text,
                    "score": score,
                    "labels": rec.labels,
                    "matched_labels": matched,
                },
            )
        )

    # Fallback: if no question matches top labels, rank by best label overall
    if not candidates:
        for rec in records:
            if not rec.labels:
                continue
            scores = [label_score_map.get(lbl, -1.0) for lbl in rec.labels]
            if not scores:
                continue
            score = max(scores)
            candidates.append(
                (
                    score,
                    {
                        "id": rec.id,
                        "question": rec.text,
                        "score": score,
                        "labels": rec.labels,
                        "matched_labels": rec.labels,
                    },
                )
            )

    candidates.sort(key=lambda x: x[0], reverse=True)
    top_k_questions = min(top_k_questions, len(candidates))
    return [candidates[i][1] for i in range(top_k_questions)]


def select_questions_by_label_embedding_aggregate(
    user_need: str,
    csv_input: str | Path | pd.DataFrame,
    embedding_model: str,
    embed_type: EmbedType | str = EmbedType.OPEN_AI,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    text_column: str = "text",
    id_column: str = "id",
    label_column: str = "labels",
    batch_size: int = 32,
    top_k_questions: int = 5,
) -> List[Dict[str, Any]]:
    """
    Select top questions by aggregating label-embedding similarity.

    Steps:
    1) Embed unique labels.
    2) Compute similarity between user_need and each label.
    3) Score each question by aggregating similarities across its labels.

    Returns a list of dicts: {id, question, score, labels, matched_labels}
    """
    if not user_need or not str(user_need).strip():
        raise ValueError("user_need must be a non-empty string.")
    user_need = str(user_need).strip()
    if top_k_questions <= 0:
        return []

    records = load_questions(
        csv_input=csv_input,
        text_column=text_column,
        id_column=id_column,
        label_column=label_column,
    )

    all_labels: List[str] = []
    for rec in records:
        for label in rec.labels:
            if label and label not in all_labels:
                all_labels.append(label)

    if not all_labels:
        raise ValueError(
            f"No labels found in column '{label_column}'. "
            "Provide a CSV with labels or specify the correct label_column."
        )

    if isinstance(embed_type, str):
        embed_type = EmbedType.from_string(embed_type)

    embedding_instance = Embedding(
        model_name=embedding_model,
        embed_type=embed_type,
        api_key=api_key,
        endpoint=endpoint,
    )

    query_vec, _ = embedding_instance.get_embedding([user_need], batch_size=1)
    label_vecs, _ = embedding_instance.get_embedding(all_labels, batch_size=batch_size)

    label_scores = cosine_similarity(query_vec[0], label_vecs)
    label_score_map = {label: float(label_scores[i]) for i, label in enumerate(all_labels)}

    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for rec in records:
        if not rec.labels:
            continue
        scores = [label_score_map.get(lbl, -1.0) for lbl in rec.labels if lbl]
        if not scores:
            continue
        # Aggregate label similarity per question (mean to avoid bias to many labels).
        score = float(sum(scores) / len(scores))
        matched_labels = sorted(
            [lbl for lbl in rec.labels if lbl],
            key=lambda lbl: label_score_map.get(lbl, -1.0),
            reverse=True,
        )
        candidates.append(
            (
                score,
                {
                    "id": rec.id,
                    "question": rec.text,
                    "score": score,
                    "labels": rec.labels,
                    "matched_labels": matched_labels,
                },
            )
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    top_k_questions = min(top_k_questions, len(candidates))
    return [candidates[i][1] for i in range(top_k_questions)]
