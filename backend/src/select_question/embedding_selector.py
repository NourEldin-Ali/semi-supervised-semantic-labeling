from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import numpy as np

from enums.embed_type import EmbedType
from src.embedding.embed import Embedding
from src.select_question.utils import cosine_similarity, load_questions


def select_questions_by_embedding(
    user_need: str,
    csv_input: str | Path | pd.DataFrame,
    embedding_model: str,
    embed_type: EmbedType | str = EmbedType.OPEN_AI,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    text_column: str = "text",
    id_column: str = "id",
    label_column: str | None = None,
    batch_size: int = 32,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Select top questions by embedding similarity to user_need.

    Returns a list of dicts: {id, question, score, labels}
    """
    if not user_need or not str(user_need).strip():
        raise ValueError("user_need must be a non-empty string.")
    user_need = str(user_need).strip()
    if top_k <= 0:
        return []

    records = load_questions(
        csv_input=csv_input,
        text_column=text_column,
        id_column=id_column,
        label_column=label_column,
    )
    texts = [r.text for r in records]

    if isinstance(embed_type, str):
        embed_type = EmbedType.from_string(embed_type)

    embedding_instance = Embedding(
        model_name=embedding_model,
        embed_type=embed_type,
        api_key=api_key,
        endpoint=endpoint,
    )

    query_vec, _ = embedding_instance.get_embedding([user_need], batch_size=1)
    doc_vecs, _ = embedding_instance.get_embedding(texts, batch_size=batch_size)

    scores = cosine_similarity(query_vec[0], doc_vecs)
    top_k = min(top_k, len(records))
    top_indices = np.argsort(-scores)[:top_k]

    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        rec = records[int(idx)]
        results.append(
            {
                "id": rec.id,
                "question": rec.text,
                "score": float(scores[int(idx)]),
                "labels": rec.labels,
            }
        )
    return results
