from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence
import re

import numpy as np
import pandas as pd


_EMPTY_STRINGS = {"", "nan", "none", "null"}


@dataclass
class QuestionRecord:
    id: str
    text: str
    labels: List[str]
    row: int


def normalize_id(value: Any, fallback: str) -> str:
    """Normalize ID for matching across CSVs (e.g. 1.0 -> 1)."""
    if value is None or pd.isna(value):
        return fallback
    s = str(value).strip()
    if not s:
        return fallback
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except (ValueError, TypeError):
        return s
    return s


def clean_text(value: Any) -> str:
    """Return a clean string or empty string for null-like values."""
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    if not s or s.lower() in _EMPTY_STRINGS:
        return ""
    return s


def parse_labels(value: Any) -> List[str]:
    """Parse a label column into a list of non-empty labels."""
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    if value is None or pd.isna(value):
        return []
    s = str(value).strip()
    if not s or s.lower() in _EMPTY_STRINGS:
        return []
    for sep in (",", ";", "|"):
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s]


def infer_text_column(df: pd.DataFrame) -> str:
    """Infer a text column as the first object dtype column."""
    for col in df.columns:
        if df[col].dtype == "object":
            return col
    raise ValueError(
        "No text column found in CSV. Please specify the text_column parameter."
    )


def load_questions(
    csv_input: str | Path | pd.DataFrame,
    text_column: Optional[str] = "text",
    id_column: Optional[str] = "id",
    label_column: Optional[str] = None,
    drop_empty: bool = True,
) -> List[QuestionRecord]:
    """Load questions from CSV or DataFrame into QuestionRecord list."""
    if isinstance(csv_input, pd.DataFrame):
        df = csv_input
    else:
        df = pd.read_csv(Path(csv_input))

    if text_column is None:
        text_column = infer_text_column(df)
    elif text_column not in df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found. Available columns: {list(df.columns)}"
        )

    if id_column is not None and id_column not in df.columns:
        raise ValueError(
            f"ID column '{id_column}' not found. Available columns: {list(df.columns)}"
        )

    if label_column is not None and label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found. Available columns: {list(df.columns)}"
        )

    records: List[QuestionRecord] = []
    for row_num, (_, row) in enumerate(df.iterrows()):
        text = clean_text(row[text_column])
        if not text and drop_empty:
            continue
        fallback_id = str(row_num)
        raw_id = row[id_column] if id_column else fallback_id
        qid = normalize_id(raw_id, fallback_id)
        labels = parse_labels(row[label_column]) if label_column else []
        records.append(QuestionRecord(id=qid, text=text, labels=labels, row=row_num))

    if not records:
        raise ValueError("No valid questions found in the CSV file.")

    return records


def simple_tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25: lowercase word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one query vector and doc matrix."""
    if query_vec.ndim == 2:
        query_vec = query_vec[0]
    q = query_vec.astype(np.float32, copy=False)
    d = doc_vecs.astype(np.float32, copy=False)
    q_norm = np.linalg.norm(q) + 1e-12
    d_norms = np.linalg.norm(d, axis=1) + 1e-12
    return (d @ q) / (d_norms * q_norm)
