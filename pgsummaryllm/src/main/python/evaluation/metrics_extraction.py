import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0

def load_grouping_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_metrics(doc: Dict[str, Any], method_name: str = "unknown") -> Dict[str, Any]:
    opt = doc.get("optimizer_details", {}) or {}
    groups = doc.get("groups", []) or []

    full_local_total = opt.get("full_local_total", None)
    cross_rhs_missing = opt.get("cross_rhs_missing", None)
    full_local_entropy = opt.get("full_local_entropy", None)
    sizes = opt.get("sizes", None)

    if full_local_total is None:
        full_local_total = sum(len(g.get("fully_local_ggds", []) or []) for g in groups)

    preserved_total = sum(len(g.get("preserved_ggds", []) or []) for g in groups)

    if sizes is None:
        sizes = [len(g.get("node_labels_for_local_check", []) or []) for g in groups]

    k = doc.get("k", len(groups) if groups else None)

    if sizes:
        mean_size = sum(sizes) / len(sizes)
        size_std = math.sqrt(sum((s - mean_size) ** 2 for s in sizes) / len(sizes))
        size_cv = _safe_div(size_std, mean_size)
        min_size, max_size = min(sizes), max(sizes)
    else:
        mean_size = size_std = size_cv = 0.0
        min_size = max_size = 0

    locality_ratio = _safe_div(full_local_total, preserved_total)

    if full_local_entropy is None and groups:
        counts = [len(g.get("fully_local_ggds", []) or []) for g in groups]
        total = sum(counts)
        if total > 0:
            probs = [c / total for c in counts if c > 0]
            ent = -sum(p * math.log(p) for p in probs)
            if len(counts) > 1:
                full_local_entropy = ent / math.log(len(counts))
            else:
                full_local_entropy = 0.0
        else:
            full_local_entropy = 0.0

    if cross_rhs_missing is None:
        cross_rhs_missing = 0

    score = opt.get("score", None)

    return {
        "method": method_name,
        "k": k,
        "groups": len(groups),
        "preserved_total": int(preserved_total),
        "fully_local_total": int(full_local_total),
        "locality_ratio": float(locality_ratio),
        "cross_rhs_missing": int(cross_rhs_missing),
        "full_local_entropy": float(full_local_entropy) if full_local_entropy is not None else None,
        "avg_group_size": float(mean_size),
        "min_group_size": int(min_size),
        "max_group_size": int(max_size),
        "size_cv": float(size_cv),
        "score": float(score) if score is not None else None,
    }

def extract_metrics_from_files(files: List[str], method_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if method_names is None:
        method_names = [Path(f).stem for f in files]
    out = []
    for f, name in zip(files, method_names):
        doc = load_grouping_json(f)
        out.append(extract_metrics(doc, method_name=name))
    return out


if __name__ == "__main__":
    path = "groups_out.json"   # <-- put your real file name here
    doc = load_grouping_json(path)
    metrics = extract_metrics(doc, method_name="Constraint-aware")

    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k:25s}: {v}")