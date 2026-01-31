#!/usr/bin/env python3
"""
Option B (Sensitivity vs K): plot metrics as a function of K with 3 lines
(Method = LLM-only, LLM+constraints, LLM+constraints+opt).

YOUR ACTUAL RESULTS STRUCTURE (SUPPORTED):
  results/
    llm_only_k2.json
    llm_only_k3.json
    llm_constraints_k2.json
    llm_constraints_k3.json
    llm_constraints_opt_k2.json
    llm_constraints_opt_k2_preserved_constraints.jsonl   (IGNORED)
    ...

Usage:
  python3 plots2.py \
    --schema_labels schema.json   (or schema_labels.txt) \
    --results_dir results \
    --out_dir plots
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")  # WSL-safe backend
import matplotlib.pyplot as plt


# -------------------- helpers --------------------

def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def extract_node_labels_from_schema_json(schema_obj: Dict[str, Any]) -> Set[str]:
    """
    Best-effort extraction from common schema.json shapes:
    - schema_summary.node_types
    - node_types / node_labels / nodeLabels / nodeTypes
    - nodes: { ... {label: "..."} ... }
    """
    labels: Set[str] = set()

    # nested schema_summary
    ss = schema_obj.get("schema_summary")
    if isinstance(ss, dict):
        for key in ("node_types", "nodeTypes", "node_labels", "nodeLabels"):
            v = ss.get(key)
            if isinstance(v, list):
                labels |= {x.strip() for x in v if isinstance(x, str) and x.strip()}

    # flat lists
    for key in ("node_types", "nodeTypes", "node_labels", "nodeLabels"):
        v = schema_obj.get(key)
        if isinstance(v, list):
            labels |= {x.strip() for x in v if isinstance(x, str) and x.strip()}

    # nodes object
    nodes = schema_obj.get("nodes")
    if isinstance(nodes, dict):
        for _, nobj in nodes.items():
            if isinstance(nobj, dict):
                lbl = nobj.get("label") or nobj.get("nodeLabel") or nobj.get("name")
                if isinstance(lbl, str) and lbl.strip():
                    labels.add(lbl.strip())

    return labels

def load_schema_labels(path: Path) -> Set[str]:
    """
    Accepts:
    - schema_labels.txt (one label per line)
    - schema.json (extract labels)
    """
    if path.suffix.lower() == ".json":
        obj = read_json(path)
        labels = extract_node_labels_from_schema_json(obj)
        if not labels:
            raise SystemExit(
                f"[ERROR] Could not extract node labels from JSON schema: {path}\n"
                "Expected keys like node_types/node_labels or schema_summary.node_types."
            )
        return labels

    # default: text file
    labels: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            labels.add(s)
    if not labels:
        raise SystemExit(f"[ERROR] schema_labels file is empty: {path}")
    return labels

def list_json_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*.json") if p.is_file()])

def guess_k_from_filename(p: Path) -> Optional[int]:
    # tries: _k3, k=3, K3
    m = re.search(r"(?:^|[^0-9])k\s*=\s*(\d+)", p.name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|[^0-9])k(\d+)", p.name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

def _collect_labels_from_anything(x: Any) -> Set[str]:
    out: Set[str] = set()
    if isinstance(x, str):
        out.add(x)
    elif isinstance(x, list):
        for e in x:
            out |= _collect_labels_from_anything(e)
    elif isinstance(x, dict):
        for v in x.values():
            out |= _collect_labels_from_anything(v)
    return out

def extract_method_metrics(obj: Dict[str, Any], schema_labels: Set[str]) -> Dict[str, float]:
    """
    Returns:
      ggds_total: float
      locality_pct: float
      cross_rhs_missing: float (count)  (NaN if missing)
      hallucination_pct: float
    """
    groups = obj.get("groups") or []

    # total GGDs = sum of preserved_ggds lengths
    total = 0
    for g in groups:
        pg = g.get("preserved_ggds") or []
        if isinstance(pg, list):
            total += len(pg)

    # locality: prefer optimizer_details.full_local_total; else sum fully_local_ggds
    opt = obj.get("optimizer_details") or {}
    full_local_total = opt.get("full_local_total")
    if isinstance(full_local_total, int):
        local = full_local_total
    else:
        local = 0
        for g in groups:
            fl = g.get("fully_local_ggds") or []
            if isinstance(fl, list):
                local += len(fl)

    locality_pct = (100.0 * local / total) if total > 0 else 0.0

    # cross rhs missing: NaN if missing
    cross_rhs_missing = opt.get("cross_rhs_missing")
    cross_rhs_missing_val = float(cross_rhs_missing) if isinstance(cross_rhs_missing, (int, float)) else float("nan")

    # hallucination: collect used labels from groups; compare with schema
    used: Set[str] = set()
    for g in groups:
        for field in (
            "node_labels_for_local_check",
            "rhs_needed_labels",
            "node_labels",
            "labels",
            "group_labels",
            "cluster_labels",
        ):
            arr = g.get(field)
            if isinstance(arr, list):
                used |= {x for x in arr if isinstance(x, str)}

        lhs_sigs = g.get("lhs_signatures") or []
        if isinstance(lhs_sigs, list):
            for sig in lhs_sigs:
                if isinstance(sig, list):
                    used |= {x for x in sig if isinstance(x, str)}
                elif isinstance(sig, str):
                    used.add(sig)

        used |= {x for x in _collect_labels_from_anything(g) if isinstance(x, str)}

    # keep only label-like tokens
    used = {u for u in used if re.match(r"^[A-Za-z0-9_]+$", u) and len(u) <= 64}

    if not used:
        halluc_pct = 0.0
    else:
        halluc = [x for x in used if x not in schema_labels]
        halluc_pct = 100.0 * (len(halluc) / len(used))

    return {
        "ggds_total": float(total),
        "locality_pct": float(locality_pct),
        "cross_rhs_missing": float(cross_rhs_missing_val),
        "hallucination_pct": float(halluc_pct),
    }

def load_series_from_results_dir(
    results_dir: Path,
    schema_labels: Set[str],
    filename_regex: str
) -> Tuple[List[int], Dict[int, Dict[str, float]]]:
    """
    Reads ALL .json in results_dir but keeps ONLY those whose filename matches filename_regex.
    filename_regex must contain the K in it (or else K must be in JSON).
    """
    pat = re.compile(filename_regex, flags=re.IGNORECASE)

    metrics_by_k: Dict[int, Dict[str, float]] = {}
    files = list_json_files(results_dir)

    for p in files:
        # ignore any JSON that is not a result (if you ever add such files)
        if "preserved_constraints" in p.name.lower():
            continue

        m = pat.match(p.name)
        if not m:
            continue

        obj = read_json(p)

        k = obj.get("k")
        if not isinstance(k, int):
            # try group from filename regex first
            if m.groups():
                try:
                    k = int(m.group(1))
                except Exception:
                    k = None
            if not isinstance(k, int):
                k = guess_k_from_filename(p)

        if not isinstance(k, int):
            continue

        metrics_by_k[k] = extract_method_metrics(obj, schema_labels)

    ks = sorted(metrics_by_k.keys())
    return ks, metrics_by_k

def plot_lines(
    ks_all: List[int],
    series: List[Tuple[str, List[float]]],
    ylabel: str,
    title: str,
    out_path: Path,
    ylim_0_100: bool = False,
) -> None:
    plt.figure()
    for name, ys in series:
        plt.plot(ks_all, ys, marker="o", label=name)
    plt.xlabel("Number of groups (K)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(ks_all)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if ylim_0_100:
        plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def write_csv(out_path: Path, ks_all: List[int], methods: List[Tuple[str, Dict[int, Dict[str, float]]]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "k", "ggds_total", "locality_pct", "cross_rhs_missing", "hallucination_pct"])
        for method_name, metrics_by_k in methods:
            for k in ks_all:
                m = metrics_by_k.get(k, {})
                w.writerow([
                    method_name,
                    k,
                    m.get("ggds_total", float("nan")),
                    m.get("locality_pct", float("nan")),
                    m.get("cross_rhs_missing", float("nan")),
                    m.get("hallucination_pct", float("nan")),
                ])


# -------------------- main --------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema_labels", required=True, help="schema_labels.txt OR schema.json")
    ap.add_argument("--results_dir", required=True, help="Folder containing llm_only_k*.json etc.")
    ap.add_argument("--out_dir", required=True, help="Where to write PNG plots + CSV.")
    args = ap.parse_args()

    schema_labels = load_schema_labels(Path(args.schema_labels))
    results_dir = Path(args.results_dir)

    # Your exact naming scheme:
    #   llm_only_k2.json
    #   llm_constraints_k3.json
    #   llm_constraints_opt_k8.json
    ks1, m1 = load_series_from_results_dir(results_dir, schema_labels, r"llm_only_k(\d+)\.json$")
    ks2, m2 = load_series_from_results_dir(results_dir, schema_labels, r"llm_constraints_k(\d+)\.json$")
    ks3, m3 = load_series_from_results_dir(results_dir, schema_labels, r"llm_constraints_opt_k(\d+)\.json$")

    ks_all = sorted(set(ks1) | set(ks2) | set(ks3))
    if not ks_all:
        raise SystemExit(
            "No K values found.\n"
            "Check that your files are named like llm_only_k2.json, llm_constraints_k2.json, llm_constraints_opt_k2.json\n"
            f"and that results_dir is correct: {results_dir}"
        )

    def y_for(metrics_by_k: Dict[int, Dict[str, float]], key: str) -> List[float]:
        out: List[float] = []
        for k in ks_all:
            out.append(metrics_by_k.get(k, {}).get(key, float("nan")))
        return out

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Locality vs K
    plot_lines(
        ks_all,
        [
            ("LLM-only", y_for(m1, "locality_pct")),
            ("LLM+constraints", y_for(m2, "locality_pct")),
            ("LLM+constraints+opt", y_for(m3, "locality_pct")),
        ],
        ylabel="GGD locality (%)",
        title="GGD Locality vs K (LDBC)",
        out_path=out_dir / "locality_vs_k.png",
        ylim_0_100=True,
    )

    # 2) Cross RHS missing vs K
    plot_lines(
        ks_all,
        [
            ("LLM-only", y_for(m1, "cross_rhs_missing")),
            ("LLM+constraints", y_for(m2, "cross_rhs_missing")),
            ("LLM+constraints+opt", y_for(m3, "cross_rhs_missing")),
        ],
        ylabel="Cross-group RHS missing (count)",
        title="Cross-group RHS Missing vs K (LDBC)",
        out_path=out_dir / "cross_rhs_missing_vs_k.png",
        ylim_0_100=False,
    )

    # 3) Hallucination vs K
    plot_lines(
        ks_all,
        [
            ("LLM-only", y_for(m1, "hallucination_pct")),
            ("LLM+constraints", y_for(m2, "hallucination_pct")),
            ("LLM+constraints+opt", y_for(m3, "hallucination_pct")),
        ],
        ylabel="Hallucinated labels (%)",
        title="Hallucination vs K (LDBC)",
        out_path=out_dir / "hallucination_vs_k.png",
        ylim_0_100=True,
    )

    # 4) CSV
    write_csv(
        out_dir / "k_sensitivity_summary.csv",
        ks_all,
        [
            ("llm_only", m1),
            ("llm_constraints", m2),
            ("llm_constraints_opt", m3),
        ],
    )

    print("[OK] Found K values:", ks_all)
    print("[OK] Wrote:")
    print(f" - {out_dir / 'locality_vs_k.png'}")
    print(f" - {out_dir / 'cross_rhs_missing_vs_k.png'}")
    print(f" - {out_dir / 'hallucination_vs_k.png'}")
    print(f" - {out_dir / 'k_sensitivity_summary.csv'}")
    print("\nOpen in Windows with:")
    print(f"  explorer.exe {out_dir.resolve()}")


if __name__ == "__main__":
    main()