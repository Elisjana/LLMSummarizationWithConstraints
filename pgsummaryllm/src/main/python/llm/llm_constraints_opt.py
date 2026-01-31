#!/usr/bin/env python3
"""
LLM-ONLY (Ollama/Gemma HTTP) + Deterministic Refinement:
Constraint-aware NODE LABEL clustering from GGDs (LHS+RHS) + per-cluster label summarization.

SOFT VALIDATION UPDATE (what you asked):
✅ If LLM returns duplicate labels across clusters (e.g., Tag appears twice), the program will NOT crash.
✅ You can choose:
   - allow duplicates AND keep them as-is (no repair), OR
   - allow duplicates BUT automatically project to a disjoint cover for the optimizer (recommended).

Run example (soft, recommended):
  python3 llm_constraints_opt.py \
    --schema schema.json \
    --constraints JsonOutput/ldbc/v1/constraints.jsonl \
    --k 6 \
    --out results/llm_constraints_opt/k=6.json \
    --ollama_host http://localhost:11434 \
    --ollama_model gemma2:2b \
    --prefer_ldbc_style \
    --balance_mode soft \
    --allow_duplicates \
    --auto_project_cover

Run example (soft, keep-as-is):
  python3 llm_constraints_opt.py \
    --schema schema.json \
    --constraints JsonOutput/ldbc/v1/constraints.jsonl \
    --k 6 \
    --out results/llm_constraints_opt/k=6.json \
    --allow_duplicates \
    --no_project_cover
"""

import argparse
import json
import os
import re
import math
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
# Robust JSON parse/repair
# ============================================================
def strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


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


def _scan_first_json_span(text: str) -> Tuple[int, int]:
    s = strip_code_fences(text)
    start_candidates = [(s.find("{"), "{"), (s.find("["), "[")]
    start_candidates = [(i, ch) for (i, ch) in start_candidates if i != -1]
    if not start_candidates:
        raise ValueError("No JSON start token found ('{' or '[').")
    start, _ = min(start_candidates, key=lambda x: x[0])

    stack = []
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                raise ValueError("Unexpected closing bracket while scanning JSON.")
            top = stack[-1]
            if (top == "{" and ch != "}") or (top == "[" and ch != "]"):
                raise ValueError("Mismatched closing bracket while scanning JSON.")
            stack.pop()
            if not stack:
                return start, i + 1

    raise ValueError("Unbalanced JSON braces/brackets (likely truncated).")


def _try_autoclose_json(text: str) -> Any:
    s = strip_code_fences(text)
    start_candidates = [(s.find("{"), "{"), (s.find("["), "[")]
    start_candidates = [(i, ch) for (i, ch) in start_candidates if i != -1]
    if not start_candidates:
        raise ValueError("No JSON start token found to auto-close.")
    start, _ = min(start_candidates, key=lambda x: x[0])
    snippet = s[start:]

    stack = []
    in_str = False
    esc = False

    for ch in snippet:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                raise ValueError("Unexpected closing bracket; cannot auto-close safely.")
            top = stack[-1]
            if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                stack.pop()
            else:
                raise ValueError("Mismatched brackets; cannot auto-close safely.")

    if in_str:
        raise ValueError("Truncated inside a string; cannot auto-close safely.")

    fixed = snippet + "".join("}" if t == "{" else "]" for t in reversed(stack))
    fixed = sanitize_llm_json_text(fixed)
    return json.loads(fixed)


def robust_parse_json(txt: str) -> Any:
    s = strip_code_fences(txt)
    s = sanitize_llm_json_text(s)
    try:
        return json.loads(s)
    except Exception:
        a, b = _scan_first_json_span(s)
        return json.loads(s[a:b])


def llm_rewrite_as_valid_json(
    *,
    host: str,
    model: str,
    broken: str,
    schema_hint: str,
    timeout: int,
) -> str:
    prompt = f"""
Rewrite the following into VALID JSON ONLY.

HARD RULES:
- Output must be exactly one JSON object.
- No markdown, no code fences, no extra text.
- Must match this schema exactly:
{schema_hint}

BROKEN OUTPUT:
{broken}
""".strip()

    return ollama_chat_content(
        host=host,
        model=model,
        prompt=prompt,
        temperature=0.0,
        num_predict=2400,
        timeout_sec=timeout,
        stop=["```", "\nfinal_checklist", "\n\n\n"],
        format_json=True,
    )


def _dump_debug_texts(debug_dir: Path, tag: str, raw: str, cleaned: str, fixed: str) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    (debug_dir / f"{tag}_{ts}_RAW.txt").write_text(raw or "", encoding="utf-8")
    (debug_dir / f"{tag}_{ts}_CLEANED.txt").write_text(cleaned or "", encoding="utf-8")
    (debug_dir / f"{tag}_{ts}_FIXED.txt").write_text(fixed or "", encoding="utf-8")


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

    try:
        return robust_parse_json(cleaned)
    except Exception as e1:
        (debug_dir / f"{tag}_parse_fail_1.txt").write_text(str(e1), encoding="utf-8")
        _dump_debug_texts(debug_dir, f"{tag}_fail1", raw, cleaned, "")

    try:
        return _try_autoclose_json(cleaned)
    except Exception as e2:
        (debug_dir / f"{tag}_parse_fail_autoclose.txt").write_text(str(e2), encoding="utf-8")
        _dump_debug_texts(debug_dir, f"{tag}_fail_autoclose", raw, cleaned, "")

    repaired = llm_rewrite_as_valid_json(
        host=host,
        model=model,
        broken=cleaned,
        schema_hint=schema_hint,
        timeout=timeout,
    )
    repaired_clean = sanitize_llm_json_text(repaired)
    (debug_dir / f"{tag}_repaired_raw.txt").write_text(repaired, encoding="utf-8")
    (debug_dir / f"{tag}_repaired_cleaned.txt").write_text(repaired_clean, encoding="utf-8")

    try:
        return robust_parse_json(repaired_clean)
    except Exception as e3:
        (debug_dir / f"{tag}_parse_fail_repaired.txt").write_text(str(e3), encoding="utf-8")
        _dump_debug_texts(debug_dir, f"{tag}_fail_repaired", raw, cleaned, repaired_clean)
        return _try_autoclose_json(repaired_clean)


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

    evidence = {
        "schema_labels": schema_labels,
        "label_frequency_total": {l: int(freq.get(l, 0)) for l in schema_labels},
        "label_frequency_lhs": {l: int(freq_lhs.get(l, 0)) for l in schema_labels},
        "label_frequency_rhs": {l: int(freq_rhs.get(l, 0)) for l in schema_labels},
        "top_neighbors_by_constraints": neighbors,
        "example_ggds_per_label": dict(label_examples),
        "notes": [
            "Cluster labels that frequently co-occur in GGDs (closure=LHS∪RHS).",
            "LHS participation is weighted higher than RHS.",
            "Hard constraint (original): every schema label appears exactly once across k clusters.",
            "Soft mode: duplicates/missing are tolerated and optionally repaired by projection."
        ]
    }
    return evidence


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


def canonicalize_partition(clusters: List[List[str]]) -> List[List[str]]:
    return [sorted(list(dict.fromkeys(c))) for c in clusters]


def ensure_disjoint_cover(
    clusters: List[List[str]],
    schema_labels: List[str],
    k: int
) -> List[List[str]]:
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


def validate_clusters_obj(
    obj: Any,
    schema_labels: List[str],
    k: int,
    *,
    allow_duplicates: bool = False,
    soft_allow_missing: bool = True
) -> Tuple[bool, List[str], Optional[List[List[str]]], str]:
    """
    Strict mode (default): requires exact disjoint cover.
    Soft mode: tolerates duplicates/missing; never fails for those issues.
    """
    errors: List[str] = []
    schema_set = set(schema_labels)

    if not isinstance(obj, dict):
        return False, ["Top-level JSON is not an object"], None, ""

    clusters = obj.get("clusters")
    expl = str(obj.get("explanation") or "").strip()

    if not isinstance(clusters, list) or not clusters:
        return False, ["Missing/invalid 'clusters' list"], None, expl

    if len(clusters) != k:
        msg = f"Expected exactly {k} clusters, got {len(clusters)}"
        if allow_duplicates:
            errors.append("[SOFT] " + msg)
        else:
            errors.append(msg)

    all_labels: List[str] = []
    unknown = set()
    dup = set()
    parsed: List[List[str]] = []

    for idx, c in enumerate(clusters):
        if not isinstance(c, dict):
            msg = f"Cluster at index {idx} is not an object"
            if allow_duplicates:
                errors.append("[SOFT] " + msg)
                parsed.append([])
                continue
            else:
                errors.append(msg)
                parsed.append([])
                continue

        labels = c.get("labels")
        if not isinstance(labels, list) or not labels:
            msg = f"Cluster at index {idx} has empty/invalid labels list"
            if allow_duplicates:
                errors.append("[SOFT] " + msg)
                parsed.append([])
                continue
            else:
                errors.append(msg)
                parsed.append([])
                continue

        clean = []
        for x in labels:
            if isinstance(x, str) and x.strip():
                clean.append(x.strip())

        uniq = []
        seen_in = set()
        for x in clean:
            if x not in seen_in:
                uniq.append(x)
                seen_in.add(x)

        parsed.append(sorted(uniq))

        for x in uniq:
            if x not in schema_set:
                unknown.add(x)
            if x in all_labels:
                dup.add(x)
            all_labels.append(x)

    missing = sorted(schema_set - set(all_labels))

    if unknown:
        msg = f"Unknown labels (not in schema): {sorted(unknown)}"
        if allow_duplicates:
            errors.append("[SOFT] " + msg)
        else:
            errors.append(msg)

    if dup:
        msg = f"Duplicate labels assigned to multiple clusters: {sorted(dup)}"
        if allow_duplicates:
            errors.append("[SOFT] " + msg)
        else:
            errors.append(msg)

    if missing:
        msg = f"Missing labels not assigned to any cluster: {missing}"
        if allow_duplicates and soft_allow_missing:
            errors.append("[SOFT] " + msg)
        else:
            errors.append(msg)

    if any(len(p) == 0 for p in parsed):
        msg = "At least one cluster is empty after cleaning"
        if allow_duplicates:
            errors.append("[SOFT] " + msg)
        else:
            errors.append(msg)

    if allow_duplicates:
        non_empty = sum(1 for p in parsed if p)
        ok = non_empty > 0
        return ok, errors, parsed, expl
    else:
        ok = len(errors) == 0
        return ok, errors, parsed if ok else None, expl


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
    prefer_ldbc_style: bool,
    *,
    allow_duplicates: bool = False,
    auto_project_cover: bool = True,
) -> Tuple[List[List[str]], str]:
    prompt = build_llm_clustering_prompt(k, evidence, prefer_ldbc_style=prefer_ldbc_style)

    schema_hint = f"""
{{
  "explanation": "string",
  "clusters": [
    {{"id":"G1","labels":[...]}} ,
    ...
    {{"id":"G{k}","labels":[...]}}
  ]
}}
""".strip()

    last_errors: List[str] = []
    last_raw = ""
    last_parsed: Optional[List[List[str]]] = None
    last_expl: str = ""

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
        (debug_dir / f"llm_cluster_raw_{attempt}.txt").write_text(raw, encoding="utf-8")

        obj = parse_or_repair_json(
            raw_text=raw,
            host=host,
            model=model,
            schema_hint=schema_hint,
            timeout=timeout,
            debug_dir=debug_dir,
            tag=f"cluster_k{k}_attempt_{attempt}",
        )

        ok, errors, parsed, expl = validate_clusters_obj(
            obj, schema_labels, k,
            allow_duplicates=allow_duplicates,
            soft_allow_missing=True
        )
        last_errors = errors
        last_expl = expl
        last_parsed = parsed

        if ok and parsed is not None:
            final_clusters = parsed
            if allow_duplicates and auto_project_cover:
                final_clusters = ensure_disjoint_cover(parsed, schema_labels, k)

            (debug_dir / "llm_cluster_final.json").write_text(
                json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (debug_dir / "llm_cluster_final_clusters.json").write_text(
                json.dumps(final_clusters, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if errors:
                (debug_dir / "llm_cluster_soft_warnings.json").write_text(
                    json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            return final_clusters, expl

        prompt = f"""
Your previous JSON is INVALID.

Errors to fix:
{json.dumps(errors, ensure_ascii=False, indent=2)}

Return ONLY VALID JSON (no markdown, no extra keys), schema:
{schema_hint}

Rules:
- Exactly {k} clusters, all non-empty
- Use ONLY schema_labels
- Cover all schema_labels exactly once (no duplicates, no missing)
- No trailing commas

schema_labels:
{json.dumps(schema_labels, ensure_ascii=False)}

GGD evidence:
{json.dumps(evidence, ensure_ascii=False)}
""".strip()

    (debug_dir / "llm_cluster_failed_last_raw.txt").write_text(last_raw, encoding="utf-8")

    if last_parsed is not None:
        fallback = last_parsed
        if allow_duplicates and auto_project_cover:
            fallback = ensure_disjoint_cover(last_parsed, schema_labels, k)

        print("[WARN] LLM clustering failed strict validation after retries; continuing with last attempt.")
        if last_errors:
            print("[WARN] Last issues:", "; ".join(last_errors[:5]))

        (debug_dir / "llm_cluster_fallback_clusters.json").write_text(
            json.dumps(fallback, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if last_errors:
            (debug_dir / "llm_cluster_fallback_warnings.json").write_text(
                json.dumps(last_errors, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return fallback, last_expl

    print("[WARN] No parsable clusters from LLM; using deterministic round-robin fallback.")
    rr = [[] for _ in range(k)]
    for i, lbl in enumerate(schema_labels):
        rr[i % k].append(lbl)
    rr = canonicalize_partition(rr)
    return rr, "Fallback: deterministic round-robin clusters due to repeated LLM parsing/validation failures."


# ============================================================
# Deterministic optimizer with HARD size bounds + fairness
# ============================================================
def compute_home_cluster_lhs_majority(ggds: List["GGDInfo"], cluster_sets: List[Set[str]]) -> Dict[str, int]:
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


def full_local_per_cluster(ggds: List["GGDInfo"], cluster_sets: List[Set[str]]) -> List[int]:
    home = compute_home_cluster_lhs_majority(ggds, cluster_sets)
    cnt = [0] * len(cluster_sets)
    for g in ggds:
        i = home[g.cid]
        if g.closure and g.closure <= cluster_sets[i]:
            cnt[i] += 1
    return cnt


def entropy(counts: List[int]) -> float:
    s = sum(counts)
    if s <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / s
        ent -= p * math.log(p + 1e-12)
    return ent


def score_partition(
    ggds: List["GGDInfo"],
    clusters: List[List[str]],
    *,
    w_full_local: float = 10.0,
    w_min_full_local: float = 6.0,
    w_entropy: float = 2.0,
    w_cross_rhs: float = 1.0,
) -> Tuple[float, Dict[str, Any]]:
    cluster_sets = [set(c) for c in clusters]
    home = compute_home_cluster_lhs_majority(ggds, cluster_sets)

    cross_rhs = 0
    for g in ggds:
        i = home[g.cid]
        cl = cluster_sets[i]
        if g.rhs:
            cross_rhs += len(g.rhs - cl)

    fl = full_local_per_cluster(ggds, cluster_sets)
    fl_total = sum(fl)
    fl_min = min(fl) if fl else 0
    fl_ent = entropy(fl)

    score = (
        w_full_local * fl_total +
        w_min_full_local * fl_min +
        w_entropy * fl_ent -
        w_cross_rhs * cross_rhs
    )

    return score, {
        "full_local_per_cluster": fl,
        "full_local_total": fl_total,
        "full_local_min": fl_min,
        "full_local_entropy": fl_ent,
        "cross_rhs_missing": cross_rhs,
        "sizes": [len(c) for c in clusters],
        "score": score
    }


def compute_size_bounds(n_labels: int, k: int, balance_mode: str) -> Tuple[int, int]:
    ideal = n_labels / max(1, k)
    if balance_mode == "strict":
        lo = math.floor(ideal)
        hi = math.ceil(ideal)
        return max(1, lo), max(1, hi)
    lo = max(1, math.floor(ideal) - 1)
    hi = max(1, math.ceil(ideal) + 1)
    return lo, hi


def optimize_partition_locality(
    ggds: List["GGDInfo"],
    schema_labels: List[str],
    init_clusters: List[List[str]],
    k: int,
    *,
    balance_mode: str = "strict",
    max_iters: int = 12000,
    patience: int = 2500,
    debug_dir: Optional[Path] = None
) -> Tuple[List[List[str]], Dict[str, Any]]:
    clusters = ensure_disjoint_cover(init_clusters, schema_labels, k)
    n = len(schema_labels)
    min_size, max_size = compute_size_bounds(n, k, balance_mode)

    def sizes(cls): return [len(c) for c in cls]

    def _assert_bounds(cls, where=""):
        sz = sizes(cls)
        if any(s < min_size or s > max_size for s in sz):
            raise RuntimeError(
                f"BOUND VIOLATION {where}: sizes={sz} bounds=({min_size},{max_size}) clusters={cls}"
            )

    def _enforce_bounds_by_rebalance(cls: List[List[str]]) -> List[List[str]]:
        cls = canonicalize_partition(cls)

        guard = 0
        while True:
            guard += 1
            if guard > 10000:
                raise RuntimeError("Rebalance guard triggered; bounds seem inconsistent.")

            sz = sizes(cls)
            over = [i for i, s in enumerate(sz) if s > max_size]
            under = [i for i, s in enumerate(sz) if s < min_size]

            if not over and not under:
                break

            if not over or not under:
                a = max(range(k), key=lambda i: len(cls[i]))
                b = min(range(k), key=lambda i: len(cls[i]))
            else:
                a = over[0]
                b = under[0]

            if len(cls[a]) <= min_size:
                break

            lbl = cls[a].pop()
            cls[b].append(lbl)
            cls = canonicalize_partition(cls)

        for i in range(k):
            if not cls[i]:
                j = max(range(k), key=lambda x: len(cls[x]))
                cls[i].append(cls[j].pop())
        cls = canonicalize_partition(cls)
        _assert_bounds(cls, "after rebalance")
        return cls

    clusters = _enforce_bounds_by_rebalance(clusters)
    _assert_bounds(clusters, "after init+rebalance")

    best_score, best_details = score_partition(ggds, clusters)
    best = [c[:] for c in clusters]

    def label_to_cluster_map(cls: List[List[str]]) -> Dict[str, int]:
        m = {}
        for i, c in enumerate(cls):
            for lbl in c:
                m[lbl] = i
        return m

    ltoc = label_to_cluster_map(clusters)
    no_improve = 0
    it = 0

    while it < max_iters and no_improve < patience:
        it += 1
        improved = False

        cluster_sets = [set(c) for c in clusters]
        home = compute_home_cluster_lhs_majority(ggds, cluster_sets)

        rhs_out_counts = Counter()
        for g in ggds:
            hi = home[g.cid]
            cl = cluster_sets[hi]
            for lbl in (g.rhs - cl):
                rhs_out_counts[lbl] += 1

        candidates = [x for x, _ in rhs_out_counts.most_common(8)]
        if not candidates:
            largest = max(range(k), key=lambda i: len(clusters[i]))
            candidates = clusters[largest][:]

        for lbl in candidates[:8]:
            a = ltoc.get(lbl)
            if a is None:
                continue
            if len(clusters[a]) <= min_size:
                continue

            targets = [i for i in range(k) if i != a]
            targets.sort(key=lambda i: len(clusters[i]))

            for b in targets:
                if len(clusters[b]) >= max_size:
                    continue

                trial = [c[:] for c in clusters]
                trial[a] = [x for x in trial[a] if x != lbl]
                trial[b] = trial[b] + [lbl]
                trial = canonicalize_partition(trial)

                if any(len(trial[i]) < min_size or len(trial[i]) > max_size for i in range(k)):
                    continue

                sc, det = score_partition(ggds, trial)
                if sc > best_score:
                    clusters = trial
                    _assert_bounds(clusters, f"after accept move {lbl} {a}->{b}")
                    ltoc = label_to_cluster_map(clusters)
                    best_score, best_details = sc, det
                    best = [c[:] for c in clusters]
                    improved = True
                    break
            if improved:
                break

        if improved:
            no_improve = 0
            continue

        for lbl in candidates[:8]:
            a = ltoc.get(lbl)
            if a is None:
                continue

            for b in range(k):
                if b == a:
                    continue

                for other in clusters[b][:6]:
                    if other == lbl:
                        continue

                    trial = [c[:] for c in clusters]
                    trial[a] = [x for x in trial[a] if x != lbl] + [other]
                    trial[b] = [x for x in trial[b] if x != other] + [lbl]
                    trial = canonicalize_partition(trial)

                    if any(len(trial[i]) < min_size or len(trial[i]) > max_size for i in range(k)):
                        continue

                    sc, det = score_partition(ggds, trial)
                    if sc > best_score:
                        clusters = trial
                        _assert_bounds(clusters, f"after accept swap {lbl}<->{other} ({a}<->{b})")
                        ltoc = label_to_cluster_map(clusters)
                        best_score, best_details = sc, det
                        best = [c[:] for c in clusters]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            no_improve = 0
        else:
            no_improve += 1

    _assert_bounds(best, "before return best")

    if debug_dir:
        (debug_dir / "optimizer_bounds.json").write_text(
            json.dumps({"min_size": min_size, "max_size": max_size, "balance_mode": balance_mode}, indent=2),
            encoding="utf-8"
        )
        (debug_dir / "optimizer_best_score.json").write_text(json.dumps(best_details, indent=2), encoding="utf-8")
        (debug_dir / "optimizer_clusters.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    return best, best_details


# ============================================================
# Group accounting after optimization
# ============================================================
def assign_ggds_to_clusters_lhs_majority(
    ggds: List[GGDInfo],
    clusters: List[List[str]]
) -> List[Dict[str, Any]]:
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

    for g in ggds:
        i = home[g.cid]
        per[i]["assigned_ggds"].add(g.cid)
        per[i]["rhs_needed"] |= g.rhs
        per[i]["local_needed"] |= g.closure
        if g.closure <= cluster_sets[i]:
            per[i]["fully_local_ggds"].add(g.cid)

    out = []
    for p in per:
        out.append({
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
    return out


def write_preserved_constraints_jsonl(
    ggds_jsonl: str,
    final_groups: List[Dict[str, Any]],
    out_jsonl: str
) -> None:
    gid_to_group: Dict[str, str] = {}
    gid_to_local_labels: Dict[str, List[str]] = {}

    for g in final_groups:
        for cid in g.get("preserved_ggds", []):
            gid_to_group[cid] = g["id"]
            gid_to_local_labels[cid] = g.get("node_labels_for_local_check", [])

    outp = Path(out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with open(outp, "w", encoding="utf-8") as w:
        for obj in iter_jsonl(ggds_jsonl):
            cid = ggd_id(obj)
            obj2 = dict(obj)
            obj2["group_assignment"] = {
                "group_id": gid_to_group.get(cid),
                "group_node_labels_for_local_check": gid_to_local_labels.get(cid, []),
            }
            w.write(json.dumps(obj2, ensure_ascii=False) + "\n")


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
            (debug_dir / f"llm_cluster_summary_raw_{gid}_{attempt}.txt").write_text(raw, encoding="utf-8")

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
            unified = base
            if unified in used_unified:
                unified = f"{base}_{len(used_unified)+1}"
            cname = f"{gid} Cluster"
            expl = f"This cluster groups labels that co-occur in constraints: {', '.join(cluster_labels)}."
            best = (unified, cname, expl)

        g["unified_label"] = best[0]
        g["cluster_name"] = best[1]
        g["cluster_explanation"] = best[2]
        used_unified.append(best[0])

        if last_raw:
            (debug_dir / f"llm_cluster_summary_last_raw_{gid}.txt").write_text(last_raw, encoding="utf-8")

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
    ap.add_argument("--cluster_num_predict", type=int, default=2000)

    ap.add_argument("--optimizer_max_iters", type=int, default=8000)
    ap.add_argument("--optimizer_patience", type=int, default=1500)
    ap.add_argument("--balance_mode", choices=["soft", "strict"], default="soft")

    ap.add_argument("--summary_tries", type=int, default=5)
    ap.add_argument("--summary_temperature", type=float, default=0.2)
    ap.add_argument("--summary_num_predict", type=int, default=350)

    ap.add_argument("--prefer_ldbc_style", action="store_true", default=False)

    # ✅ NEW FLAGS (soft behavior)
    ap.add_argument("--allow_duplicates", action="store_true", default=False,
                    help="Soft mode: tolerate duplicates/missing labels in LLM clusters (do not crash).")
    ap.add_argument("--auto_project_cover", dest="auto_project_cover", action="store_true", default=True,
                    help="If soft mode is enabled, project clusters to a disjoint cover before optimizer (recommended).")
    ap.add_argument("--no_project_cover", dest="auto_project_cover", action="store_false",
                    help="If soft mode is enabled, keep clusters as-is (no projection).")

    args = ap.parse_args()

    out_path = Path(args.out)
    debug_dir = out_path.resolve().parent / "_debug_llm_constraint_aware_clustering"
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

    # 1) LLM initial clustering
    init_clusters, clustering_expl = llm_cluster_labels(
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
        prefer_ldbc_style=args.prefer_ldbc_style,
        allow_duplicates=args.allow_duplicates,
        auto_project_cover=args.auto_project_cover
    )

    # If user chose keep-as-is, do NOT force disjoint cover here
    if not (args.allow_duplicates and not args.auto_project_cover):
        init_clusters = ensure_disjoint_cover(init_clusters, schema_labels, args.k)

    (debug_dir / "clusters_llm_initial.json").write_text(json.dumps(init_clusters, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) Optimize to preserve GGDs INSIDE groups
    # Optimizer assumes a disjoint cover; so we always enforce it here.
    opt_clusters, opt_details = optimize_partition_locality(
        ggds=ggds,
        schema_labels=schema_labels,
        init_clusters=init_clusters,
        k=args.k,
        balance_mode=args.balance_mode,
        max_iters=args.optimizer_max_iters,
        patience=args.optimizer_patience,
        debug_dir=debug_dir
    )
    (debug_dir / "clusters_optimized.json").write_text(json.dumps(opt_clusters, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) Group accounting
    groups = assign_ggds_to_clusters_lhs_majority(ggds, opt_clusters)

    # 4) LLM summaries
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
            "Then a deterministic locality optimizer reassigned labels (still disjoint) with HARD size bounds "
            "and fairness to maximize GGDs that are fully local in their home cluster (closure ⊆ cluster), "
            "and to reduce cross-cluster RHS dependencies. Finally, the LLM produced per-cluster unified label "
            "and 1–2 sentence characterization."
        ),
        "k": args.k,
        "ollama_model": args.ollama_model,
        "llm_used_for_clustering": True,
        "llm_used_for_cluster_summaries": True,
        "llm_summaries_ok": ok_summaries,
        "llm_clustering_explanation": clustering_expl,
        "optimizer_details": opt_details,
        "groups": groups,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print("OK wrote:", str(out_path))

    preserved_out = str(out_path.with_suffix("")) + "_preserved_constraints.jsonl"
    write_preserved_constraints_jsonl(args.constraints, groups, preserved_out)
    print("OK wrote:", preserved_out)


if __name__ == "__main__":
    main()