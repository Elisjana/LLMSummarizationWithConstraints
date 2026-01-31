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

ADDED METRICS (new):
  4) GGD Fragmentation (avg #clusters touched by a GGD closure)  (lower better)
  5) RHS Locality (%) = 100*(1 - cross_rhs_missing / total_rhs_refs) (higher better)
  6) Reasoning load balance:
       - full_local_entropy (higher better)
       - full_local_std     (lower better)
  7) Coverage density = ggds_total / K (ggds per group) (interpretation depends)
  8) Max fragmentation (worst-case)

We keep your original 3 plots + add extra plots + extend CSV.

Usage:
  python3 plots2.py \
    --schema_labels schema.json   (or schema_labels.txt) \
    --results_dir results \
    --out_dir plots
"""

import argparse
import csv
import json
import math
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

def _entropy(counts: List[float]) -> float:
    s = float(sum(counts))
    if s <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / s
        ent -= p * math.log(p + 1e-12)
    return ent

def _std(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    return math.sqrt(v)

def _safe_float(x: Any) -> float:
    return float(x) if isinstance(x, (int, float)) else float("nan")

def _labels_from_group(g: Dict[str, Any]) -> Set[str]:
    """
    Extract cluster labels from a group object in your outputs.
    Prefers lhs_signatures[0] (your typical format), then cluster_labels, then labels.
    """
    # your standard output: "lhs_signatures": [ ["Person","Forum",... ] ]
    lhs_sigs = g.get("lhs_signatures")
    if isinstance(lhs_sigs, list) and lhs_sigs:
        first = lhs_sigs[0]
        if isinstance(first, list):
            return {x for x in first if isinstance(x, str)}
        if isinstance(first, str):
            return {first}

    for key in ("cluster_labels", "labels", "group_labels", "node_labels"):
        v = g.get(key)
        if isinstance(v, list):
            return {x for x in v if isinstance(x, str)}

    # fallback: best effort
    cand = _collect_labels_from_anything(g)
    # keep only label-like tokens (avoid sentences)
    return {u for u in cand if re.match(r"^[A-Za-z0-9_]+$", u) and len(u) <= 64}

def _parse_cluster_index(groups: List[Dict[str, Any]]) -> Tuple[List[Set[str]], Dict[str, int]]:
    """
    Returns:
      cluster_sets: list of label sets per cluster
      label_to_cluster: mapping label -> cluster id
    """
    cluster_sets: List[Set[str]] = []
    for g in groups:
        cluster_sets.append(set(_labels_from_group(g)))

    label_to_cluster: Dict[str, int] = {}
    for i, s in enumerate(cluster_sets):
        for lbl in s:
            # first assignment wins
            if lbl not in label_to_cluster:
                label_to_cluster[lbl] = i
    return cluster_sets, label_to_cluster


# -------------------- Metrics extraction --------------------

def extract_method_metrics(obj: Dict[str, Any], schema_labels: Set[str]) -> Dict[str, float]:
    """
    Computes both your original metrics + new ones.
    Returns floats (NaN if not computable).

    ORIGINAL:
      ggds_total
      locality_pct
      cross_rhs_missing
      hallucination_pct

    NEW:
      fragmentation_avg
      fragmentation_max
      rhs_total_refs
      rhs_locality_pct
      full_local_entropy
      full_local_std
      coverage_density
    """
    groups = obj.get("groups") or []
    opt = obj.get("optimizer_details") or {}

    # ---- Basic counts ----
    # total GGDs = sum of preserved_ggds lengths
    total_ggds = 0
    for g in groups:
        pg = g.get("preserved_ggds") or []
        if isinstance(pg, list):
            total_ggds += len(pg)

    # locality: prefer optimizer_details.full_local_total; else sum fully_local_ggds
    full_local_total = opt.get("full_local_total")
    if isinstance(full_local_total, int):
        local_ggds = full_local_total
    else:
        local_ggds = 0
        for g in groups:
            fl = g.get("fully_local_ggds") or []
            if isinstance(fl, list):
                local_ggds += len(fl)

    locality_pct = (100.0 * local_ggds / total_ggds) if total_ggds > 0 else 0.0

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

    used = {u for u in used if re.match(r"^[A-Za-z0-9_]+$", u) and len(u) <= 64}

    if not used:
        halluc_pct = 0.0
    else:
        halluc = [x for x in used if x not in schema_labels]
        halluc_pct = 100.0 * (len(halluc) / len(used))

    # ---- NEW METRICS ----

    # cluster label sets + label->cluster mapping
    cluster_sets, label_to_cluster = _parse_cluster_index(groups)

    # fragmentation: for each GGD assigned to a group, how many clusters does its label-set touch?
    # We approximate "closure labels" for that GGD using group_node_labels_for_local_check if present,
    # else node_labels_for_local_check, else cluster labels.
    frags: List[int] = []
    rhs_total_refs = 0.0  # total RHS label references aggregated over all GGDs (approx)
    # We do not have per-GGD RHS list in output, only per-group rhs_needed_labels (union),
    # so rhs_total_refs is approximated at group level by |rhs_needed_labels| * |preserved_ggds|.
    for g in groups:
        pg = g.get("preserved_ggds") or []
        if not isinstance(pg, list) or len(pg) == 0:
            continue

        rhs_needed = g.get("rhs_needed_labels") or []
        if isinstance(rhs_needed, list):
            rhs_total_refs += float(len(rhs_needed) * len(pg))

        # best-effort closure-like labels
        closure_like = None
        # if you ever add per-ggd mapping, extend here.
        for key in ("group_node_labels_for_local_check", "node_labels_for_local_check"):
            v = g.get(key)
            if isinstance(v, list) and any(isinstance(x, str) for x in v):
                closure_like = {x for x in v if isinstance(x, str)}
                break
        if closure_like is None:
            closure_like = set(_labels_from_group(g))

        touched = set()
        for lbl in closure_like:
            ci = label_to_cluster.get(lbl)
            if ci is not None:
                touched.add(ci)
        frag_val = len(touched) if touched else 1  # if unknown, treat as 1
        # repeat for each GGD assigned to this group (since we only have group-level closure union)
        frags.extend([frag_val] * len(pg))

    fragmentation_avg = float(sum(frags) / len(frags)) if frags else float("nan")
    fragmentation_max = float(max(frags)) if frags else float("nan")

    # RHS locality %: normalize cross_rhs_missing by rhs_total_refs (approx)
    if not math.isnan(cross_rhs_missing_val) and rhs_total_refs > 0:
        rhs_locality_pct = 100.0 * (1.0 - (cross_rhs_missing_val / rhs_total_refs))
        rhs_locality_pct = max(0.0, min(100.0, rhs_locality_pct))
    else:
        rhs_locality_pct = float("nan")

    # reasoning load balance: use full_local_per_cluster if present
    flpc = opt.get("full_local_per_cluster")
    if isinstance(flpc, list) and flpc and all(isinstance(x, (int, float)) for x in flpc):
        flpc_f = [float(x) for x in flpc]
        full_local_entropy = float(_entropy(flpc_f))
        full_local_std = float(_std(flpc_f))
    else:
        full_local_entropy = float("nan")
        full_local_std = float("nan")

    # coverage density: ggds per group
    k_val = obj.get("k")
    if isinstance(k_val, int) and k_val > 0:
        coverage_density = float(total_ggds / k_val)
    else:
        coverage_density = float("nan")

    return {
        # original
        "ggds_total": float(total_ggds),
        "locality_pct": float(locality_pct),
        "cross_rhs_missing": float(cross_rhs_missing_val),
        "hallucination_pct": float(halluc_pct),

        # new
        "fragmentation_avg": fragmentation_avg,
        "fragmentation_max": fragmentation_max,
        "rhs_total_refs": float(rhs_total_refs) if rhs_total_refs > 0 else float("nan"),
        "rhs_locality_pct": float(rhs_locality_pct),
        "full_local_entropy": float(full_local_entropy),
        "full_local_std": float(full_local_std),
        "coverage_density": float(coverage_density),
    }


def load_series_from_results_dir(
    results_dir: Path,
    schema_labels: Set[str],
    filename_regex: str
) -> Tuple[List[int], Dict[int, Dict[str, float]]]:
    """
    Reads ALL .json in results_dir but keeps ONLY those whose filename matches filename_regex.
    """
    pat = re.compile(filename_regex, flags=re.IGNORECASE)

    metrics_by_k: Dict[int, Dict[str, float]] = {}
    files = list_json_files(results_dir)

    for p in files:
        if "preserved_constraints" in p.name.lower():
            continue

        m = pat.match(p.name)
        if not m:
            continue

        obj = read_json(p)

        k = obj.get("k")
        if not isinstance(k, int):
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
        w.writerow([
            "method", "k",
            "ggds_total",
            "locality_pct",
            "cross_rhs_missing",
            "hallucination_pct",
            "fragmentation_avg",
            "fragmentation_max",
            "rhs_total_refs",
            "rhs_locality_pct",
            "full_local_entropy",
            "full_local_std",
            "coverage_density",
        ])
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
                    m.get("fragmentation_avg", float("nan")),
                    m.get("fragmentation_max", float("nan")),
                    m.get("rhs_total_refs", float("nan")),
                    m.get("rhs_locality_pct", float("nan")),
                    m.get("full_local_entropy", float("nan")),
                    m.get("full_local_std", float("nan")),
                    m.get("coverage_density", float("nan")),
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

    # 4) Fragmentation avg vs K
    plot_lines(
        ks_all,
        [
            ("LLM-only", y_for(m1, "fragmentation_avg")),
            ("LLM+constraints", y_for(m2, "fragmentation_avg")),
            ("LLM+constraints+opt", y_for(m3, "fragmentation_avg")),
        ],
        ylabel="Avg GGD fragmentation (#clusters touched)",
        title="GGD Fragmentation (Avg) vs K (LDBC)",
        out_path=out_dir / "fragmentation_avg_vs_k.png",
        ylim_0_100=False,
    )

    # 5) Fragmentation max vs K
    plot_lines(
        ks_all,
        [
            ("LLM-only", y_for(m1, "fragmentation_max")),
            ("LLM+constraints", y_for(m2, "fragmentation_max")),
            ("LLM+constraints+opt", y_for(m3, "fragmentation_max")),
        ],
        ylabel="Max GGD fragmentation (#clusters touched)",
        title="GGD Fragmentation (Max) vs K (LDBC)",
        out_path=out_dir / "fragmentation_max_vs_k.png",
        ylim_0_100=False,
    )

    # 6) RHS Locality % vs K (normalized)
    plot_lines(
        ks_all,
        [
            ("LLM-only", y_for(m1, "rhs_locality_pct")),
            ("LLM+constraints", y_for(m2, "rhs_locality_pct")),
            ("LLM+constraints+opt", y_for(m3, "rhs_locality_pct")),
        ],
        ylabel="RHS locality (%) (approx)",
        title="RHS Locality vs K (LDBC)",
        out_path=out_dir / "rhs_locality_vs_k.png",
        ylim_0_100=True,
    )

    # 7) Reasoning load entropy vs K
    plot_lines(
        ks_all,
        [
            ("LLM-only", y_for(m1, "full_local_entropy")),
            ("LLM+constraints", y_for(m2, "full_local_entropy")),
            ("LLM+constraints+opt", y_for(m3, "full_local_entropy")),
        ],
        ylabel="Entropy(full_local_per_cluster)",
        title="Reasoning Load Entropy vs K (LDBC)",
        out_path=out_dir / "reasoning_entropy_vs_k.png",
        ylim_0_100=False,
    )

    # 8) Reasoning load std vs K
    plot_lines(
        ks_all,
        [
            ("LLM-only", y_for(m1, "full_local_std")),
            ("LLM+constraints", y_for(m2, "full_local_std")),
            ("LLM+constraints+opt", y_for(m3, "full_local_std")),
        ],
        ylabel="Std(full_local_per_cluster)",
        title="Reasoning Load Std vs K (LDBC)",
        out_path=out_dir / "reasoning_std_vs_k.png",
        ylim_0_100=False,
    )

    # 9) Coverage density vs K
    plot_lines(
        ks_all,
        [
            ("LLM-only", y_for(m1, "coverage_density")),
            ("LLM+constraints", y_for(m2, "coverage_density")),
            ("LLM+constraints+opt", y_for(m3, "coverage_density")),
        ],
        ylabel="GGDs per group (ggds_total / K)",
        title="Coverage Density vs K (LDBC)",
        out_path=out_dir / "coverage_density_vs_k.png",
        ylim_0_100=False,
    )

    # CSV
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
    for fn in [
        "locality_vs_k.png",
        "cross_rhs_missing_vs_k.png",
        "hallucination_vs_k.png",
        "fragmentation_avg_vs_k.png",
        "fragmentation_max_vs_k.png",
        "rhs_locality_vs_k.png",
        "reasoning_entropy_vs_k.png",
        "reasoning_std_vs_k.png",
        "coverage_density_vs_k.png",
        "k_sensitivity_summary.csv",
    ]:
        print(f" - {out_dir / fn}")

    print("\nOpen in Windows with:")
    print(f"  explorer.exe {out_dir.resolve()}")


if __name__ == "__main__":
    main()