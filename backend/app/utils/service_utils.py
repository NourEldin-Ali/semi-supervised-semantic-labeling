"""
Service utility functions for orchestrating different operations.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

from src.embedding.embed import Embedding
from src.clustering.clustering import possibilistic_clustering, get_possibilistic_clusters_after_elbow
from src.classification.train_knn import train_knn
from src.classification.predict_knn import predict_knn_labels
from src.seedlabel.group_extractor_graph import GroupOrchestrator
from src.seedlabel.models.group_questions import GroupQuestion
from src.llmlabel.single_extractor_graph import SingleOrchestrator
from src.llmlabel.models.single_question import SingleQuestion
from config.llm_config import LLMConnector
from enums.embed_type import EmbedType
from enums.llm_type import LLMType
from src.llm.models.labeled_question import QuestionLabels
from src.llm.nodes.llm_evaluator import evaluate_one_pair
from src.select_question.bm25_selector import select_questions_bm25
from src.select_question.embedding_selector import select_questions_by_embedding
from src.select_question.label_embedding_selector import select_questions_by_label_embedding


def generate_embeddings_from_csv(
    df: pd.DataFrame,
    text_column: str,
    embedding_model: str,
    embed_type: EmbedType,
    batch_size: int = 32,
    api_key: str = None,
) -> tuple[np.ndarray, int]:
    """Generate embeddings for text column in DataFrame. Returns (embeddings, tokens_consumed)."""
    texts = df[text_column].astype(str).tolist()
    embedding_instance = Embedding(
        model_name=embedding_model,
        embed_type=embed_type,
        api_key=api_key,
    )
    embeddings, tokens = embedding_instance.get_embedding(texts, batch_size=batch_size)
    return embeddings, tokens


def generate_clusters(
    embeddings: np.ndarray,
    k: int = 3,
    metric: str = 'cosine'
) -> List[List[int]]:
    """Generate clustering groups from embeddings."""
    memberships = possibilistic_clustering(embeddings, metric=metric, k=k)
    clusters = get_possibilistic_clusters_after_elbow(memberships)
    return clusters


def generate_group_labels(
    clusters: List[List[int]],
    df: pd.DataFrame,
    id_column: str,
    text_column: str,
    llm_model: str,
    llm_type: LLMType,
    api_key: str = None,
    run_config: dict = None,
) -> Dict[str, List[str]]:
    """Generate labels for each cluster group using LLM."""
    group_questions = []
    for cluster_id, indices in enumerate(clusters):
        questions = [str(df.iloc[idx][text_column]) for idx in indices if idx < len(df)]
        group_questions.append(GroupQuestion(id=str(cluster_id), questions=questions))

    orchestrator = GroupOrchestrator(
        model_name=llm_model,
        model_type=llm_type,
        api_key=api_key,
    )
    invoke_kwargs = {}
    if run_config is not None:
        invoke_kwargs["config"] = run_config
    label_results = orchestrator.invoke(group_questions, **invoke_kwargs)
    
    # Map cluster IDs to labels
    cluster_labels = {}
    for result in label_results:
        cluster_labels[result.id] = result.labels
    
    # Map individual item IDs to their cluster labels
    # Items can belong to multiple clusters, so accumulate labels from all clusters
    item_labels = {}
    for cluster_id, indices in enumerate(clusters):
        cluster_label = cluster_labels.get(str(cluster_id), [])
        for idx in indices:
            if idx < len(df):
                item_id = str(df.iloc[idx][id_column])
                if item_id not in item_labels:
                    item_labels[item_id] = []
                # Add labels from this cluster (avoid duplicates)
                for label in cluster_label:
                    if label not in item_labels[item_id]:
                        item_labels[item_id].append(label)
    
    return item_labels


def generate_single_item_labels(
    df: pd.DataFrame,
    id_column: str,
    text_column: str,
    llm_model: str,
    llm_type: LLMType,
    api_key: str = None,
    run_config: dict = None,
) -> Dict[str, List[str]]:
    """Generate labels for each individual item using LLM."""
    questions = [
        SingleQuestion(id=str(row[id_column]), question=str(row[text_column]))
        for _, row in df.iterrows()
    ]
    orchestrator = SingleOrchestrator(
        model_name=llm_model,
        model_type=llm_type,
        api_key=api_key,
    )
    invoke_kwargs = {}
    if run_config is not None:
        invoke_kwargs["config"] = run_config
    label_results = orchestrator.invoke(questions, **invoke_kwargs)
    
    # Map item IDs to labels
    item_labels = {}
    for result in label_results:
        item_labels[result.id] = result.labels
    
    return item_labels


def train_knn_model(
    embeddings: np.ndarray,
    labeled_df: pd.DataFrame,
    id_column: str,
    label_column: str
) -> tuple:
    """Train a KNN model from labeled data and embeddings."""
    # Create a mapping of DataFrame row indices to labels
    label_mapping = {}
    for df_idx, row in labeled_df.iterrows():
        labels_str = str(row[label_column])
        labels = [l.strip() for l in labels_str.split(",") if l.strip()] if labels_str and labels_str.lower() != 'nan' else []
        label_mapping[df_idx] = labels[0] if labels else None  # Use first label for training
    
    # Filter out rows without labels and align embeddings
    # Map DataFrame index to position in embeddings array
    valid_df_indices = [i for i, label in label_mapping.items() if label is not None]
    if not valid_df_indices:
        raise ValueError("No valid labels found in the labeled CSV file")
    
    # Get the position indices (assuming DataFrame index aligns with embeddings array)
    valid_positions = [int(idx) for idx in valid_df_indices if int(idx) < len(embeddings)]
    if not valid_positions:
        raise ValueError("No valid embedding indices found")
    
    valid_embeddings = embeddings[valid_positions]
    labels_for_training = [label_mapping[i] for i in valid_df_indices if int(i) < len(embeddings)]
    
    # Train KNN model
    knn_model = train_knn(valid_embeddings)
    
    # Store label data for prediction (store both DataFrame indices and position indices)
    label_data = {
        'labels': labels_for_training,
        'indices': valid_positions  # Position indices in embeddings array
    }
    
    return knn_model, label_data, valid_embeddings


def predict_with_knn(
    knn_model: Any,
    label_data: Dict,
    training_embeddings: np.ndarray,
    new_embeddings: np.ndarray,
    k: int = 3
) -> List[str]:
    """Predict labels for new embeddings using trained KNN model."""
    from collections import Counter
    
    # Use the trained knn_model which is a NearestNeighbors instance
    # Find k nearest neighbors for all new embeddings
    distances, indices = knn_model.kneighbors(new_embeddings, n_neighbors=min(k, len(training_embeddings)))
    
    # Get labels for each prediction
    predicted_labels = []
    for neighbor_indices in indices:
        # neighbor_indices are indices into the training_embeddings array
        # which correspond to indices in label_data['labels']
        neighbor_labels = [label_data['labels'][i] for i in neighbor_indices]
        
        # Use majority vote
        if neighbor_labels:
            majority_label = Counter(neighbor_labels).most_common(1)[0][0]
            predicted_labels.append(majority_label)
        else:
            predicted_labels.append("")
    
    return predicted_labels


def _normalize_id(value) -> str:
    """Normalize ID for matching across CSVs (e.g. strip, 1.0 -> 1)."""
    if pd.isna(value):
        return ""
    s = str(value).strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return s
    except (ValueError, TypeError):
        return s


def _parse_labels(val) -> list:
    """Parse label column into list of non-empty strings. Handles NaN, 'nan', comma/semicolon."""
    if pd.isna(val) or val is None:
        return []
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return []
    for sep in (",", ";"):
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s]


def evaluate_with_llm_judge(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    id_column: str,
    text_column: str,
    label_column: str,
    method1_name: str,
    method2_name: str,
    llm_model: str,
    llm_type: LLMType,
    api_key: str = None
) -> Dict[str, Any]:
    """Evaluate two CSV files using LLM as judge. Returns metrics per question and averages."""
    # Ensure both DataFrames have the required columns
    if id_column not in df1.columns or id_column not in df2.columns:
        raise ValueError(f"ID column '{id_column}' not found in one or both CSV files")
    
    if label_column not in df1.columns or label_column not in df2.columns:
        raise ValueError(f"Label column '{label_column}' not found in one or both CSV files")
    
    text_col = (text_column or "text").strip() or "text"
    if text_col not in df1.columns:
        raise ValueError(
            f"Text column '{text_col}' not found in first CSV. Available: {list(df1.columns)}"
        )
    if text_col not in df2.columns:
        raise ValueError(
            f"Text column '{text_col}' not found in second CSV. Available: {list(df2.columns)}"
        )
    text_column = text_col
    
    # Build mappings using normalized IDs (same logical ID matches across files)
    df1_dict = {}
    duplicate_ids_df1 = 0
    for _, row in df1.iterrows():
        nid = _normalize_id(row[id_column])
        if nid:
            if nid in df1_dict:
                duplicate_ids_df1 += 1
            df1_dict[nid] = row
    df2_dict = {}
    duplicate_ids_df2 = 0
    for _, row in df2.iterrows():
        nid = _normalize_id(row[id_column])
        if nid:
            if nid in df2_dict:
                duplicate_ids_df2 += 1
            df2_dict[nid] = row

    if duplicate_ids_df1 or duplicate_ids_df2:
        logger = logging.getLogger("uvicorn.error")
        if duplicate_ids_df1:
            logger.warning(
                "Evaluation: %d duplicate IDs in first CSV (last row per ID kept). "
                "Questions not taken into account for overwritten rows.",
                duplicate_ids_df1,
            )
        if duplicate_ids_df2:
            logger.warning(
                "Evaluation: %d duplicate IDs in second CSV (last row per ID kept). "
                "Questions not taken into account for overwritten rows.",
                duplicate_ids_df2,
            )

    common_ids = set(df1_dict.keys()) & set(df2_dict.keys())
    ids_only_in_file1 = set(df1_dict.keys()) - common_ids
    ids_only_in_file2 = set(df2_dict.keys()) - common_ids

    def _text(r, col):
        v = r.get(col, None)
        if pd.isna(v) or v is None:
            return ""
        s = str(v).strip()
        return s if s and s.lower() not in ("nan", "none", "null") else ""

    questions_method_1 = []
    questions_method_2 = []
    skipped_no_labels = 0
    skipped_no_text = 0

    for item_id in sorted(common_ids):
        row1 = df1_dict[item_id]
        row2 = df2_dict[item_id]

        labels1 = _parse_labels(row1[label_column])
        labels2 = _parse_labels(row2[label_column])

        if not labels1 or not labels2:
            skipped_no_labels += 1
            continue

        question_text = _text(row1, text_column) or _text(row2, text_column)
        if not question_text:
            skipped_no_text += 1
            continue

        questions_method_1.append(QuestionLabels(
            id=str(item_id),
            question=question_text,
            method=method1_name,
            labels=labels1,
        ))
        questions_method_2.append(QuestionLabels(
            id=str(item_id),
            question=question_text,
            method=method2_name,
            labels=labels2,
        ))

    if not questions_method_1 or not questions_method_2:
        raise ValueError(
            "No valid questions with labels found in both methods. "
            f"Common IDs: {len(common_ids)}, skipped (no labels in both): {skipped_no_labels}, "
            f"skipped (no question text): {skipped_no_text}, valid pairs: {len(questions_method_1)}. "
            "Check id_column, label_column, and text_column; ensure both CSVs use the same IDs "
            "and have non-empty labels and question text for those rows."
        )

    from app.utils.execution_stats import run_with_stats
    from src.llm.utils.token_usage_handler import TokenUsageCallbackHandler

    llm = LLMConnector(
        model_name=llm_model,
        llm_type=llm_type,
        api_key=api_key,
    )()
    token_handler = TokenUsageCallbackHandler()
    run_config = {"callbacks": [token_handler]}

    def _run_eval():
        out_list: List[Any] = []
        for q1, q2 in zip(questions_method_1, questions_method_2):
            r = evaluate_one_pair(q1, q2, llm, config=run_config)
            out_list.append(r)
        return out_list

    evaluation_results, run_stats = run_with_stats(_run_eval, token_handler=token_handler)

    # Process results: deduplicate by id (keep first) to avoid duplicate rows from any upstream glitch
    seen_ids: Set[str] = set()
    question_metrics = []
    metric_totals = {
        'relevance_method1': 0.0,
        'correctness_method1': 0.0,
        'coverage_method1': 0.0,
        'taxonomy_fit_granularity_method1': 0.0,
        'actionability_method1': 0.0,
        'relevance_method2': 0.0,
        'correctness_method2': 0.0,
        'coverage_method2': 0.0,
        'taxonomy_fit_granularity_method2': 0.0,
        'actionability_method2': 0.0,
    }

    for result in evaluation_results:
        if result.id in seen_ids:
            continue
        seen_ids.add(result.id)
        eval1, eval2 = result.evaluations

        question_metrics.append({
            'id': result.id,
            'method1': {
                'method': eval1.method,
                'relevance': eval1.relevance,
                'correctness': eval1.correctness,
                'coverage': eval1.coverage,
                'taxonomy_fit_granularity': eval1.taxonomy_fit_granularity,
                'actionability': eval1.actionability,
                'reasoning': eval1.reasoning,
                'labels': eval1.labels_considered or []
            },
            'method2': {
                'method': eval2.method,
                'relevance': eval2.relevance,
                'correctness': eval2.correctness,
                'coverage': eval2.coverage,
                'taxonomy_fit_granularity': eval2.taxonomy_fit_granularity,
                'actionability': eval2.actionability,
                'reasoning': eval2.reasoning,
                'labels': eval2.labels_considered or []
            }
        })
        
        # Accumulate totals for averages
        metric_totals['relevance_method1'] += eval1.relevance
        metric_totals['correctness_method1'] += eval1.correctness
        metric_totals['coverage_method1'] += eval1.coverage
        metric_totals['taxonomy_fit_granularity_method1'] += eval1.taxonomy_fit_granularity
        metric_totals['actionability_method1'] += eval1.actionability
        
        metric_totals['relevance_method2'] += eval2.relevance
        metric_totals['correctness_method2'] += eval2.correctness
        metric_totals['coverage_method2'] += eval2.coverage
        metric_totals['taxonomy_fit_granularity_method2'] += eval2.taxonomy_fit_granularity
        metric_totals['actionability_method2'] += eval2.actionability
    
    num_questions = len(question_metrics)
    
    # Calculate averages
    averages = {
        'method1': {
            'relevance': metric_totals['relevance_method1'] / num_questions if num_questions > 0 else 0,
            'correctness': metric_totals['correctness_method1'] / num_questions if num_questions > 0 else 0,
            'coverage': metric_totals['coverage_method1'] / num_questions if num_questions > 0 else 0,
            'taxonomy_fit_granularity': metric_totals['taxonomy_fit_granularity_method1'] / num_questions if num_questions > 0 else 0,
            'actionability': metric_totals['actionability_method1'] / num_questions if num_questions > 0 else 0,
        },
        'method2': {
            'relevance': metric_totals['relevance_method2'] / num_questions if num_questions > 0 else 0,
            'correctness': metric_totals['correctness_method2'] / num_questions if num_questions > 0 else 0,
            'coverage': metric_totals['coverage_method2'] / num_questions if num_questions > 0 else 0,
            'taxonomy_fit_granularity': metric_totals['taxonomy_fit_granularity_method2'] / num_questions if num_questions > 0 else 0,
            'actionability': metric_totals['actionability_method2'] / num_questions if num_questions > 0 else 0,
        }
    }
    
    ignored_data = {
        'ids_only_in_file1': len(ids_only_in_file1),
        'ids_only_in_file2': len(ids_only_in_file2),
        'skipped_no_labels': skipped_no_labels,
        'skipped_no_text': skipped_no_text,
    }

    out = {
        'total_questions_evaluated': num_questions,
        'method1_name': method1_name,
        'method2_name': method2_name,
        'average_metrics': averages,
        'question_metrics': question_metrics,
        'ignored_data': ignored_data,
    }
    out.update(run_stats)
    return out


def compare_csv_files(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    id_column: str,
    label_column: str = "labels"
) -> Dict[str, Any]:
    """Compare two CSV files with labels and generate evaluation metrics."""
    # Ensure both DataFrames have the required columns
    if id_column not in df1.columns or id_column not in df2.columns:
        raise ValueError(f"ID column '{id_column}' not found in one or both CSV files")
    
    if label_column not in df1.columns or label_column not in df2.columns:
        raise ValueError(f"Label column '{label_column}' not found in one or both CSV files")
    
    # Create mappings of ID to labels
    labels1 = {}
    labels2 = {}
    
    for _, row in df1.iterrows():
        labels_str = str(row[label_column])
        labels1[str(row[id_column])] = set(
            [l.strip() for l in labels_str.split(",") if l.strip()]
        ) if labels_str and labels_str.lower() != 'nan' else set()
    
    for _, row in df2.iterrows():
        labels_str = str(row[label_column])
        labels2[str(row[id_column])] = set(
            [l.strip() for l in labels_str.split(",") if l.strip()]
        ) if labels_str and labels_str.lower() != 'nan' else set()
    
    # Find common IDs
    common_ids = set(labels1.keys()) & set(labels2.keys())
    
    if not common_ids:
        return {
            "total_items": 0,
            "common_items": 0,
            "exact_matches": 0,
            "partial_matches": 0,
            "no_matches": 0,
            "exact_match_rate": 0.0,
            "partial_match_rate": 0.0,
            "comparison_details": []
        }
    
    # Compare labels
    exact_matches = 0
    partial_matches = 0
    no_matches = 0
    comparison_details = []
    
    for item_id in common_ids:
        labels_1 = labels1[item_id]
        labels_2 = labels2[item_id]
        
        intersection = labels_1 & labels_2
        union = labels_1 | labels_2
        
        if labels_1 == labels_2:
            exact_matches += 1
            match_type = "exact"
        elif intersection:
            partial_matches += 1
            match_type = "partial"
        else:
            no_matches += 1
            match_type = "no_match"
        
        comparison_details.append({
            "item_id": item_id,
            "method1_labels": list(labels_1),
            "method2_labels": list(labels_2),
            "common_labels": list(intersection),
            "match_type": match_type,
            "jaccard_similarity": len(intersection) / len(union) if union else 0.0
        })
    
    total_common = len(common_ids)
    
    return {
        "total_items_method1": len(labels1),
        "total_items_method2": len(labels2),
        "common_items": total_common,
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "no_matches": no_matches,
        "exact_match_rate": exact_matches / total_common if total_common > 0 else 0.0,
        "partial_match_rate": partial_matches / total_common if total_common > 0 else 0.0,
        "average_jaccard_similarity": sum(
            detail["jaccard_similarity"] for detail in comparison_details
        ) / total_common if total_common > 0 else 0.0,
        "comparison_details": comparison_details
    }


def select_questions_bm25_service(
    df: pd.DataFrame,
    user_need: str,
    text_column: str,
    id_column: str,
    label_column: str | None = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Select top questions with BM25."""
    return select_questions_bm25(
        user_need=user_need,
        csv_input=df,
        text_column=text_column,
        id_column=id_column,
        label_column=label_column,
        top_k=top_k,
    )


def select_questions_embedding_service(
    df: pd.DataFrame,
    user_need: str,
    embedding_model: str,
    embed_type: EmbedType | str,
    api_key: str | None = None,
    endpoint: str | None = None,
    text_column: str = "text",
    id_column: str = "id",
    label_column: str | None = None,
    batch_size: int = 32,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Select top questions using embeddings."""
    return select_questions_by_embedding(
        user_need=user_need,
        csv_input=df,
        embedding_model=embedding_model,
        embed_type=embed_type,
        api_key=api_key,
        endpoint=endpoint,
        text_column=text_column,
        id_column=id_column,
        label_column=label_column,
        batch_size=batch_size,
        top_k=top_k,
    )


def select_questions_label_embedding_service(
    df: pd.DataFrame,
    user_need: str,
    embedding_model: str,
    embed_type: EmbedType | str,
    api_key: str | None = None,
    endpoint: str | None = None,
    text_column: str = "text",
    id_column: str = "id",
    label_column: str = "labels",
    batch_size: int = 32,
    top_k_labels: int = 5,
    top_k_questions: int = 5,
) -> List[Dict[str, Any]]:
    """Select top questions using label embeddings."""
    return select_questions_by_label_embedding(
        user_need=user_need,
        csv_input=df,
        embedding_model=embedding_model,
        embed_type=embed_type,
        api_key=api_key,
        endpoint=endpoint,
        text_column=text_column,
        id_column=id_column,
        label_column=label_column,
        batch_size=batch_size,
        top_k_labels=top_k_labels,
        top_k_questions=top_k_questions,
    )
