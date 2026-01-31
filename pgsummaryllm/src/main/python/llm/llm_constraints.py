#!/usr/bin/env python3
"""
LLM-ONLY (Ollama/Gemma HTTP):
Constraint-aware NODE LABEL clustering from GGDs (LHS+RHS) + per-cluster label summarization.

NO optimizer / NO deterministic refinement.
We still do minimal "sanity repair" to guarantee:
- exactly K clusters
- disjoint cover of all schema labels
- no missing / no duplicates

Run:
  python3 llm_constraints.py \
    --schema schema.json \
    --constraints JsonOutput/ldbc/v1/constraints.jsonl \
    --k 4 \
    --out results/llm_constraints/k=4.json \
    --ollama_host http://localhost:11434 \
    --ollama_model gemma2:2b \
    --prefer_ldbc_style
"""

import argparse
import json
import os
import re
import time
import urllib.request
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma2:2b")


# ============================================================
# JSONL
# ============================================================
def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}")


# ============================================================
# Schema parsing (robust)
# ============================================================
def extract_node_labels_from_schema(schema_obj: Dict[str, Any]) -> List[str]:
    labels: Set[str] = set()

    nodes = schema_obj.get("nodes")
    if isinstance(nodes, dict):
        for _, nobj in nodes.items():
            if isinstance(nobj, dict):
                lbl = nobj.get("label") or nobj.get("nodeLabel") or nobj.get("name")
                if isinstance(lbl, str) and lbl.strip():
                    labels.add(lbl.strip())

    for key in ("node_labels", "nodeLabels", "node_types", "nodeTypes"):
        v = schema_obj.get(key)
        if isinstance(v, list):
            for x in v:
                if isinstance(x, str) and x.strip():
                    labels.add(x.strip())

    if not labels:
        raise KeyError(
            "Could not extract node labels from schema.json. Expected schema['nodes'][*]['label'] "
            "or schema['node_labels']/schema['node_types']."
        )
    return sorted(labels)


# ============================================================
# GGD helpers
# ============================================================
def ggd_id(g: dict) -> str:
    gid = g.get("id") or g.get("cid") or g.get("ggd_id") or g.get("name")
    return str(gid) if gid else "UNKNOWN"


def get_node_label(n: dict) -> Optional[str]:
    for k in ("Node label", "nodeLabel", "label", "node_label", "NodeLabel"):
        v = n.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def labels_from_side(side: Any) -> Set[str]:
    out: Set[str] = set()
    if not isinstance(side, dict):
        return out
    for n in (side.get("nodes", []) or []):
        if isinstance(n, dict):
            lbl = get_node_label(n)
            if lbl:
                out.add(lbl)
    return out


def lhs_labels(g: dict) -> Set[str]:
    return labels_from_side(g.get("lhs", {}) or {})


def rhs_labels(g: dict) -> Set[str]:
    return labels_from_side(g.get("rhs", {}) or {})


# ============================================================
# Ollama HTTP
# ============================================================
def ollama_chat_content(
    host: str,
    model: str,
    prompt: str,
    temperature: float = 0.2,
    num_predict: int = 1000,
    timeout_sec: int = 900,
    stop: Optional[List[str]] = None,
    format_json: bool = True,
) -> str:
    host = host.rstrip("/")
    url = host + "/api/chat"

    options = {"temperature": temperature, "num_predict": num_predict}
    if stop:
        options["stop"] = stop

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": options,
    }
    if format_json:
        payload["format"] = "json"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    obj = json.loads(raw)
    return str((obj.get("message") or {}).get("content") or "").strip()


# ============================================================
# Robust JSON parse/repair (sanitization)
# ============================================================
def sanitize_llm_json_text(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).strip()

    first_obj = s.find("{")
    first_arr = s.find("[")
    if first_obj == -1 and first_arr == -1:
        return s.strip()

    if first_obj == -1:
        start = first_arr
    elif first_arr == -1:
        start = first_obj
    else:
        start = min(first_obj, first_arr)

    end = max(s.rfind("}"), s.rfind("]"))
    if end != -1 and end > start:
        s = s[start:end + 1].strip()
    else:
        s = s[start:].strip()

    s = re.sub(r"\bNaN\b", "null", s)
    s = re.sub(r"\bInfinity\b", "null", s)
    s = re.sub(r"\b-Infinity\b", "null", s)

    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s.strip()


def robust_parse_json(txt: str) -> Any:
    s = sanitize_llm_json_text(txt)
    return json.loads(s)


def parse_or_repair_json(
    *,
    raw_text: str,
    host: str,
    model: str,
    schema_hint: str,
    timeout: int,
    debug_dir: Path,
    tag: str,
) -> Any:
    raw = raw_text if isinstance(raw_text, str) else ("" if raw_text is None else str(raw_text))
    cleaned = sanitize_llm_json_text(raw)

    ts = time.strftime("%Y%m%d-%H%M%S")
    (debug_dir / f"{tag}_{ts}_RAW.txt").write_text(raw, encoding="utf-8")
    (debug_dir / f"{tag}_{ts}_CLEANED.txt").write_text(cleaned, encoding="utf-8")

    try:
        return robust_parse_json(cleaned)
    except Exception as e1:
        (debug_dir / f"{tag}_{ts}_parse_fail.txt").write_text(str(e1), encoding="utf-8")

    # last resort: ask LLM to rewrite valid JSON
    rewrite_prompt = f"""
Rewrite the following into VALID JSON ONLY.

HARD RULES:
- Output must be exactly one JSON object.
- No markdown, no extra text.
- Must match this schema exactly:
{schema_hint}

BROKEN OUTPUT:
{cleaned}
""".strip()

    repaired = ollama_chat_content(
        host=host,
        model=model,
        prompt=rewrite_prompt,
        temperature=0.0,
        num_predict=2400,
        timeout_sec=timeout,
        stop=["```", "\nfinal_checklist", "\n\n\n"],
        format_json=True,
    )
    repaired_clean = sanitize_llm_json_text(repaired)
    (debug_dir / f"{tag}_{ts}_REPAIRED.txt").write_text(repaired, encoding="utf-8")
    (debug_dir / f"{tag}_{ts}_REPAIRED_CLEAN.txt").write_text(repaired_clean, encoding="utf-8")

    return json.loads(repaired_clean)


# ============================================================
# GGD model
# ============================================================
@dataclass(frozen=True)
class GGDInfo:
    cid: str
    lhs: Set[str]
    rhs: Set[str]
    closure: Set[str]


def load_ggds(constraints_jsonl: str, schema_labels: List[str]) -> List[GGDInfo]:
    schema_set = set(schema_labels)
    out: List[GGDInfo] = []
    for g in iter_jsonl(constraints_jsonl):
        cid = ggd_id(g)
        lhs = {x for x in lhs_labels(g) if x in schema_set}
        rhs = {x for x in rhs_labels(g) if x in schema_set}
        if not lhs and not rhs:
            continue
        out.append(GGDInfo(cid=cid, lhs=lhs, rhs=rhs, closure=set(lhs) | set(rhs)))
    return out


# ============================================================
# Evidence building for LLM clustering
# ============================================================
def build_constraint_evidence(
    ggds: List[GGDInfo],
    schema_labels: List[str],
    w_lhs: float = 2.0,
    w_rhs: float = 1.0,
    max_neighbors: int = 6,
    max_examples_per_label: int = 6
) -> Dict[str, Any]:
    freq = Counter()
    freq_lhs = Counter()
    freq_rhs = Counter()
    co = defaultdict(lambda: defaultdict(float))
    label_examples = defaultdict(list)

    for g in ggds:
        lhs = sorted(g.lhs)
        rhs = sorted(g.rhs)

        for l in lhs:
            freq[l] += 1
            freq_lhs[l] += 1
            if len(label_examples[l]) < max_examples_per_label:
                label_examples[l].append(g.cid)

        for l in rhs:
            freq[l] += 1
            freq_rhs[l] += 1
            if len(label_examples[l]) < max_examples_per_label:
                label_examples[l].append(g.cid)

        closure = sorted(g.closure)
        weights = {}
        for l in closure:
            weights[l] = 0.0
            if l in g.lhs:
                weights[l] += w_lhs
            if l in g.rhs:
                weights[l] += w_rhs

        for i in range(len(closure)):
            for j in range(i + 1, len(closure)):
                a, b = closure[i], closure[j]
                w = (weights[a] * weights[b]) ** 0.5
                co[a][b] += w
                co[b][a] += w

    neighbors = {}
    for l in schema_labels:
        items = sorted(co[l].items(), key=lambda kv: kv[1], reverse=True)
        neighbors[l] = [{"label": x, "w": round(float(w), 3)} for x, w in items[:max_neighbors]]

    return {
        "schema_labels": schema_labels,
        "label_frequency_total": {l: int(freq.get(l, 0)) for l in schema_labels},
        "label_frequency_lhs": {l: int(freq_lhs.get(l, 0)) for l in schema_labels},
        "label_frequency_rhs": {l: int(freq_rhs.get(l, 0)) for l in schema_labels},
        "top_neighbors_by_constraints": neighbors,
        "example_ggds_per_label": dict(label_examples),
        "notes": [
            "Cluster labels that frequently co-occur in GGDs (closure=LHS∪RHS).",
            "LHS participation is weighted higher than RHS.",
            "Hard constraint: every schema label appears exactly once across k clusters."
        ]
    }


# ============================================================
# Minimal sanity: force disjoint cover (NOT optimization)
# ============================================================
def canonicalize_partition(clusters: List[List[str]]) -> List[List[str]]:
    return [sorted(list(dict.fromkeys(c))) for c in clusters]


def ensure_disjoint_cover(clusters: List[List[str]], schema_labels: List[str], k: int) -> List[List[str]]:
    clusters = canonicalize_partition(clusters)
    schema_set = set(schema_labels)

    seen = set()
    for i in range(len(clusters)):
        newc = []
        for lbl in clusters[i]:
            if lbl in schema_set and lbl not in seen:
                newc.append(lbl)
                seen.add(lbl)
        clusters[i] = newc

    missing = [l for l in schema_labels if l not in seen]
    for lbl in missing:
        j = min(range(len(clusters)), key=lambda x: len(clusters[x]))
        clusters[j].append(lbl)

    for i in range(len(clusters)):
        if not clusters[i]:
            j = max(range(len(clusters)), key=lambda x: len(clusters[x]))
            clusters[i].append(clusters[j].pop())

    if len(clusters) != k:
        if len(clusters) > k:
            extra = clusters[k:]
            clusters = clusters[:k]
            for e in extra:
                for lbl in e:
                    j = min(range(k), key=lambda x: len(clusters[x]))
                    clusters[j].append(lbl)
        else:
            while len(clusters) < k:
                clusters.append([])
            for i in range(k):
                if not clusters[i]:
                    j = max(range(k), key=lambda x: len(clusters[x]))
                    clusters[i].append(clusters[j].pop())

    return canonicalize_partition(clusters)


# ============================================================
# LLM clustering
# ============================================================
def build_llm_clustering_prompt(k: int, evidence: Dict[str, Any], prefer_ldbc_style: bool) -> str:
    style_hint = ""
    if prefer_ldbc_style:
        style_hint = """
STYLE PREFERENCE (soft constraint):
Prefer clusters like:
- Content & interaction: Person, Forum, Post, Comment, Tag
- Taxonomy: TagClass (+ Tag if needed)
- Context: Organisation, Place
But you MUST still respect GGD evidence and hard constraints.
""".strip()

    return f"""
You are a constraint-aware clustering assistant for property-graph NODE LABELS.

TASK:
Cluster ALL schema node labels into exactly {k} DISJOINT clusters based only on the given GGD evidence.

HARD CONSTRAINTS:
- Output EXACTLY {k} clusters.
- Every schema label MUST appear in EXACTLY ONE cluster (no duplicates, no missing).
- Do NOT invent labels.
- Every cluster MUST be non-empty.
- Output JSON ONLY. No markdown. No trailing commas. No extra keys.

{style_hint}

OUTPUT JSON ONLY (exact keys):
{{
  "explanation": "2-4 sentences",
  "clusters": [
    {{"id":"G1","labels":["..."]}},
    ...
    {{"id":"G{k}","labels":["..."]}}
  ]
}}

GGD-derived evidence (authoritative):
{json.dumps(evidence, ensure_ascii=False)}
""".strip()


def llm_cluster_labels(
    host: str,
    model: str,
    k: int,
    evidence: Dict[str, Any],
    schema_labels: List[str],
    tries: int,
    temperature: float,
    num_predict: int,
    timeout: int,
    debug_dir: Path,
    prefer_ldbc_style: bool
) -> Tuple[List[List[str]], str]:
    prompt = build_llm_clustering_prompt(k, evidence, prefer_ldbc_style=prefer_ldbc_style)

    schema_hint = f"""
{{
  "explanation": "string",
  "clusters": [
    {{"id":"G1","labels":[...]}},
    ...
    {{"id":"G{k}","labels":[...]}}
  ]
}}
""".strip()

    last_raw = ""
    for attempt in range(1, tries + 1):
        raw = ollama_chat_content(
            host=host,
            model=model,
            prompt=prompt,
            temperature=temperature,
            num_predict=num_predict,
            timeout_sec=timeout,
            stop=["```", "\nfinal_checklist", "\n\n\n"],
            format_json=True,
        )
        last_raw = raw

        obj = parse_or_repair_json(
            raw_text=raw,
            host=host,
            model=model,
            schema_hint=schema_hint,
            timeout=timeout,
            debug_dir=debug_dir,
            tag=f"cluster_k{k}_attempt_{attempt}",
        )

        if isinstance(obj, dict) and isinstance(obj.get("clusters"), list):
            expl = str(obj.get("explanation") or "").strip()
            parsed: List[List[str]] = []
            for c in obj["clusters"]:
                if isinstance(c, dict) and isinstance(c.get("labels"), list):
                    parsed.append([str(x).strip() for x in c["labels"] if isinstance(x, str) and x.strip()])

            # IMPORTANT: minimal validity repair (not optimization)
            parsed = ensure_disjoint_cover(parsed, schema_labels, k)
            return parsed, expl

        # retry prompt if invalid
        prompt = f"""
Your previous JSON was invalid or missing required keys.

Return ONLY VALID JSON matching:
{schema_hint}

Rules:
- Exactly {k} clusters, all non-empty
- Use ONLY schema_labels
- Cover all schema_labels exactly once (no duplicates, no missing)

schema_labels:
{json.dumps(schema_labels, ensure_ascii=False)}

GGD evidence:
{json.dumps(evidence, ensure_ascii=False)}
""".strip()

    (debug_dir / f"llm_cluster_failed_k{k}.txt").write_text(last_raw, encoding="utf-8")
    raise RuntimeError(f"LLM clustering failed after retries for k={k}.")


# ============================================================
# Assign GGDs to clusters (for output metrics only)
# ============================================================
def compute_home_cluster_lhs_majority(ggds: List[GGDInfo], cluster_sets: List[Set[str]]) -> Dict[str, int]:
    home = {}
    for g in ggds:
        best_i = 0
        best_score = (-1, -1)
        for i, cl in enumerate(cluster_sets):
            lhs_ov = len(g.lhs & cl)
            clo_ov = len(g.closure & cl)
            score = (lhs_ov, clo_ov)
            if score > best_score:
                best_score = score
                best_i = i
        home[g.cid] = best_i
    return home


def groups_with_metrics(ggds: List[GGDInfo], clusters: List[List[str]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cluster_sets = [set(c) for c in clusters]
    home = compute_home_cluster_lhs_majority(ggds, cluster_sets)

    per = []
    for i, cl in enumerate(cluster_sets):
        per.append({
            "id": f"G{i+1}",
            "cluster_labels": sorted(cl),
            "assigned_ggds": set(),
            "rhs_needed": set(),
            "local_needed": set(),
            "fully_local_ggds": set(),
        })

    cross_rhs_missing = 0
    for g in ggds:
        i = home[g.cid]
        cl = cluster_sets[i]
        per[i]["assigned_ggds"].add(g.cid)
        per[i]["rhs_needed"] |= g.rhs
        per[i]["local_needed"] |= g.closure
        if g.closure <= cl:
            per[i]["fully_local_ggds"].add(g.cid)
        if g.rhs:
            cross_rhs_missing += len(g.rhs - cl)

    groups = []
    for p in per:
        groups.append({
            "id": p["id"],
            "cluster_name": "",
            "cluster_explanation": "",
            "unified_label": "",
            "lhs_signatures": [p["cluster_labels"]],
            "rhs_needed_labels": sorted(p["rhs_needed"]),
            "node_labels_for_local_check": sorted(set(p["cluster_labels"]) | set(p["local_needed"])),
            "preserved_ggds": sorted(p["assigned_ggds"]),
            "fully_local_ggds": sorted(p["fully_local_ggds"]),
        })

    opt_details = {
        "full_local_per_cluster": [len(p["fully_local_ggds"]) for p in per],
        "full_local_total": sum(len(p["fully_local_ggds"]) for p in per),
        "cross_rhs_missing": cross_rhs_missing,
        "sizes": [len(c) for c in clusters],
        "note": "LLM-only run: metrics computed post-hoc (no optimization)."
    }
    return groups, opt_details


# ============================================================
# LLM per-cluster summarization
# ============================================================
def clean_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9_ ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_unified_label(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "ClusterLabel"


def build_cluster_summary_prompt(
    schema_labels: List[str],
    group_id: str,
    cluster_labels: List[str],
    rhs_needed: List[str],
    example_ggds: List[str],
    evidence_neighbors: Dict[str, Any],
    used_unified_labels: List[str],
) -> str:
    return f"""
You are a graph constraint-aware summarization assistant.

TASK:
Given a CLUSTER of schema node labels (already decided), produce:
1) unified_label: a NEW unified label name (single token like SocialContent or Social_Content)
2) cluster_name: a short human-friendly title
3) cluster_explanation: EXACTLY 1–2 sentences describing what the cluster represents, grounded in constraints.

STRICT RULES:
- Do NOT propose changing cluster membership.
- Do NOT invent new member labels.
- Output JSON ONLY with exactly these keys:
  unified_label, cluster_name, cluster_explanation
- unified_label MUST be DIFFERENT from already used unified labels.

already_used_unified_labels:
{json.dumps(used_unified_labels, ensure_ascii=False)}

OUTPUT JSON ONLY:
{{
  "unified_label": "UniqueLabelHere",
  "cluster_name": "Short Title",
  "cluster_explanation": "1–2 sentences."
}}

schema_labels:
{json.dumps(schema_labels, ensure_ascii=False)}

group_id: {group_id}

cluster_labels:
{json.dumps(cluster_labels, ensure_ascii=False)}

rhs_needed_labels (context):
{json.dumps(rhs_needed, ensure_ascii=False)}

example_ggds:
{json.dumps(example_ggds[:10], ensure_ascii=False)}

top_neighbors_by_constraints:
{json.dumps({l: evidence_neighbors.get(l, []) for l in cluster_labels}, ensure_ascii=False)}
""".strip()


def fill_cluster_summaries_llm(
    groups: List[Dict[str, Any]],
    schema_labels: List[str],
    evidence_neighbors: Dict[str, Any],
    host: str,
    model: str,
    tries: int,
    temperature: float,
    num_predict: int,
    timeout: int,
    debug_dir: Path
) -> int:
    ok = 0
    used_unified: List[str] = []

    schema_hint = """
{
  "unified_label": "string",
  "cluster_name": "string",
  "cluster_explanation": "string"
}
""".strip()

    for g in groups:
        gid = g["id"]
        cluster_labels = g["lhs_signatures"][0] if g.get("lhs_signatures") else []
        rhs_needed = g.get("rhs_needed_labels", [])
        example_ggds = g.get("preserved_ggds", [])[:12]

        prompt = build_cluster_summary_prompt(
            schema_labels=schema_labels,
            group_id=gid,
            cluster_labels=cluster_labels,
            rhs_needed=rhs_needed,
            example_ggds=example_ggds,
            evidence_neighbors=evidence_neighbors,
            used_unified_labels=used_unified,
        )

        best = None
        for attempt in range(1, tries + 1):
            raw = ollama_chat_content(
                host=host,
                model=model,
                prompt=prompt,
                temperature=temperature,
                num_predict=num_predict,
                timeout_sec=timeout,
                stop=["```", "\nfinal_checklist", "\n\n\n"],
                format_json=True,
            )
            obj = parse_or_repair_json(
                raw_text=raw,
                host=host,
                model=model,
                schema_hint=schema_hint,
                timeout=timeout,
                debug_dir=debug_dir,
                tag=f"summary_{gid}_attempt_{attempt}",
            )
            if not isinstance(obj, dict):
                continue

            unified = clean_unified_label(obj.get("unified_label", ""))
            if unified in used_unified:
                continue

            cname = clean_name(obj.get("cluster_name", "")) or unified.replace("_", " ")
            expl = (obj.get("cluster_explanation") or "").strip()

            sents = [x.strip() for x in re.split(r"[.!?]+", expl) if x.strip()]
            if len(sents) == 0:
                expl = f"This cluster groups labels that frequently co-occur in constraints: {', '.join(cluster_labels)}."
            elif len(sents) > 2:
                expl = ". ".join(sents[:2]) + "."
            else:
                if not expl.endswith((".", "!", "?")):
                    expl = expl + "."

            best = (unified, cname, expl)
            ok += 1
            break

        if best is None:
            base = clean_unified_label(gid)
            unified = base if base not in used_unified else f"{base}_{len(used_unified)+1}"
            cname = f"{gid} Cluster"
            expl = f"This cluster groups labels that co-occur in constraints: {', '.join(cluster_labels)}."
            best = (unified, cname, expl)

        g["unified_label"] = best[0]
        g["cluster_name"] = best[1]
        g["cluster_explanation"] = best[2]
        used_unified.append(best[0])

    return ok


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True)
    ap.add_argument("--constraints", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", required=True)

    ap.add_argument("--w_lhs", type=float, default=2.0)
    ap.add_argument("--w_rhs", type=float, default=1.0)

    ap.add_argument("--ollama_host", default=DEFAULT_HOST)
    ap.add_argument("--ollama_model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=int, default=900)

    ap.add_argument("--cluster_tries", type=int, default=10)
    ap.add_argument("--cluster_temperature", type=float, default=0.05)
    ap.add_argument("--cluster_num_predict", type=int, default=2200)

    ap.add_argument("--summary_tries", type=int, default=5)
    ap.add_argument("--summary_temperature", type=float, default=0.2)
    ap.add_argument("--summary_num_predict", type=int, default=400)

    ap.add_argument("--prefer_ldbc_style", action="store_true", default=False)

    args = ap.parse_args()

    out_path = Path(args.out)
    debug_dir = out_path.resolve().parent / "_debug_llm_only_clustering"
    debug_dir.mkdir(parents=True, exist_ok=True)

    schema_obj = json.load(open(args.schema, "r", encoding="utf-8"))
    schema_labels = extract_node_labels_from_schema(schema_obj)

    if args.k > len(schema_labels):
        print(f"[WARN] k={args.k} > #labels={len(schema_labels)}; reducing k to {len(schema_labels)}")
        args.k = len(schema_labels)

    ggds = load_ggds(args.constraints, schema_labels)
    if not ggds:
        raise RuntimeError("No usable GGDs found (after filtering to schema labels).")

    evidence = build_constraint_evidence(
        ggds=ggds,
        schema_labels=schema_labels,
        w_lhs=args.w_lhs,
        w_rhs=args.w_rhs,
        max_neighbors=6,
        max_examples_per_label=6
    )
    (debug_dir / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    # 1) LLM clustering (LLM-only)
    clusters, clustering_expl = llm_cluster_labels(
        host=args.ollama_host,
        model=args.ollama_model,
        k=args.k,
        evidence=evidence,
        schema_labels=schema_labels,
        tries=args.cluster_tries,
        temperature=args.cluster_temperature,
        num_predict=args.cluster_num_predict,
        timeout=args.timeout,
        debug_dir=debug_dir,
        prefer_ldbc_style=args.prefer_ldbc_style
    )

    # 2) Compute group accounting/metrics post-hoc (no optimization)
    groups, details = groups_with_metrics(ggds, clusters)

    # 3) LLM summaries for each cluster
    neighbors = evidence.get("top_neighbors_by_constraints", {})
    ok_summaries = fill_cluster_summaries_llm(
        groups=groups,
        schema_labels=schema_labels,
        evidence_neighbors=neighbors,
        host=args.ollama_host,
        model=args.ollama_model,
        tries=args.summary_tries,
        temperature=args.summary_temperature,
        num_predict=args.summary_num_predict,
        timeout=args.timeout,
        debug_dir=debug_dir
    )

    out_obj = {
        "explanation": (
            "LLM clustered schema labels using GGD-derived evidence (LHS weighted, RHS included). "
            "No deterministic optimizer was applied. Post-hoc metrics report locality and cross-cluster RHS."
        ),
        "k": args.k,
        "ollama_model": args.ollama_model,
        "llm_used_for_clustering": True,
        "llm_used_for_cluster_summaries": True,
        "llm_summaries_ok": ok_summaries,
        "llm_clustering_explanation": clustering_expl,
        "optimizer_details": details,  # kept for compatibility
        "groups": groups,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK wrote:", str(out_path))


if __name__ == "__main__":
    main()