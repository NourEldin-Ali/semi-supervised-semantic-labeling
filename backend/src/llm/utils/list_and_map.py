
from src.llm.models.label_result import LabelsResult


def append_merge(a: dict[str, list[str]] | None, b: dict[str, list[str]] | None):
    """Merge per-group label collections while preserving insertion order."""
    a = {**(a or {})}
    for k, v in (b or {}).items():
        a.setdefault(k, [])
        # v could be a single dict (one result) or a list already
        if isinstance(v, list):
            a[k].extend(v)
        else:
            a[k].append(v)
    return a

def add_if_not_exists(a: list[LabelsResult] | None, b: list[LabelsResult] | None):
    """Append the labels from `b` into `a`, skipping duplicates."""
    if a is None:
        a = []
    if b is None:
        return a

    for x in b:
        if x not in a:
            a.append(x)
    return a

def add_evaluation_results(a: list | None, b: list | None):
    """Append evaluation results from `b` into `a`."""
    if a is None:
        a = []
    if b is None:
        return a
    a.extend(b)
    return a
