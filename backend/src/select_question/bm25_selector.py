from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from src.select_question.utils import load_questions, simple_tokenize


def _invoke_retriever(retriever: BM25Retriever, query: str):
    """Call retriever in a version-tolerant way."""
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    return retriever.get_relevant_documents(query)


def select_questions_bm25(
    user_need: str,
    csv_input: str | Path | pd.DataFrame,
    text_column: str = "text",
    id_column: str = "id",
    label_column: str | None = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Select top questions using BM25 (LangChain BM25Retriever).

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

    documents = [
        Document(
            page_content=rec.text,
            metadata={"id": rec.id, "row": rec.row, "labels": rec.labels},
        )
        for rec in records
    ]

    retriever = BM25Retriever.from_documents(
        documents, preprocess_func=simple_tokenize
    )
    retriever.k = min(top_k, len(documents))

    hits = _invoke_retriever(retriever, user_need)

    results: List[Dict[str, Any]] = []
    for doc in hits:
        metadata = doc.metadata or {}
        results.append(
            {
                "id": metadata.get("id"),
                "question": doc.page_content,
                "score": metadata.get("score"),
                "labels": metadata.get("labels", []),
            }
        )
    return results
