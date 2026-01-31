#!/usr/bin/env python3
"""
PURE LLM-ONLY BASELINE:
Node-label grouping + per-group summarization from schema.json ONLY.

NO constraints used.
NO optimizer / NO deterministic refinement (no locality scoring, no swaps, no bounds).
We only apply a minimal validity "repair" so output is always a disjoint cover:
- every schema label appears exactly once across k clusters
- no missing labels
- no duplicates
- exactly k non-empty clusters

Run:
  python3 llm_only.py \
    --schema schema.json \
    --k 4 \
    --out results/llm_only/k=4.json \
    --ollama_host http://localhost:11434 \
    --ollama_model gemma2:2b
"""

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma2:2b")


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
# Ollama HTTP
# ============================================================
def ollama_chat_content(
    host: str,
    model: str,
    prompt: str,
    temperature: float = 0.2,
    num_predict: int = 1200,
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
# JSON parsing helpers
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

    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s.strip()


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
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / f"{tag}_{ts}_RAW.txt").write_text(raw, encoding="utf-8")
    (debug_dir / f"{tag}_{ts}_CLEANED.txt").write_text(cleaned, encoding="utf-8")

    try:
        return json.loads(cleaned)
    except Exception as e1:
        (debug_dir / f"{tag}_{ts}_parse_fail.txt").write_text(str(e1), encoding="utf-8")

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
        num_predict=2200,
        timeout_sec=timeout,
        stop=["```", "\nfinal_checklist", "\n\n\n"],
        format_json=True,
    )
    repaired_clean = sanitize_llm_json_text(repaired)
    (debug_dir / f"{tag}_{ts}_REPAIRED.txt").write_text(repaired, encoding="utf-8")
    (debug_dir / f"{tag}_{ts}_REPAIRED_CLEAN.txt").write_text(repaired_clean, encoding="utf-8")

    return json.loads(repaired_clean)


# ============================================================
# Minimal validity repair: disjoint cover
# ============================================================
def canonicalize_partition(clusters: List[List[str]]) -> List[List[str]]:
    return [sorted(list(dict.fromkeys(c))) for c in clusters]


def ensure_disjoint_cover(clusters: List[List[str]], schema_labels: List[str], k: int) -> List[List[str]]:
    """
    Minimal guard so the baseline always returns something comparable:
    - remove duplicates across clusters (keep first occurrence)
    - add missing labels into smallest clusters
    - ensure exactly k non-empty clusters
    """
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
# LLM prompts
# ============================================================
def build_llm_clustering_prompt(schema_labels: List[str], k: int) -> str:
    return f"""
You are a clustering assistant for property-graph NODE LABELS.

TASK:
Cluster ALL schema node labels into exactly {k} DISJOINT clusters based ONLY on general semantic similarity
(meaning/role), without using any constraints.

HARD CONSTRAINTS:
- Output EXACTLY {k} clusters.
- Every schema label MUST appear in EXACTLY ONE cluster (no duplicates, no missing).
- Do NOT invent labels.
- Every cluster MUST be non-empty.
- Output JSON ONLY. No markdown. No trailing commas. No extra keys.

OUTPUT JSON ONLY (exact keys):
{{
  "explanation": "2-4 sentences explaining the grouping logic",
  "clusters": [
    {{"id":"G1","labels":["..."]}},
    ...
    {{"id":"G{k}","labels":["..."]}}
  ]
}}

schema_labels (authoritative):
{json.dumps(schema_labels, ensure_ascii=False)}
""".strip()


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
    used_unified_labels: List[str],
) -> str:
    return f"""
You are a schema summarization assistant.

TASK:
Given a CLUSTER of schema node labels (already decided), produce:
1) unified_label: a NEW unified label name (single token like SocialContent or Social_Content)
2) cluster_name: a short human-friendly title
3) cluster_explanation: EXACTLY 1–2 sentences describing what the cluster represents.

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
""".strip()


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", required=True)

    ap.add_argument("--ollama_host", default=DEFAULT_HOST)
    ap.add_argument("--ollama_model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=int, default=900)

    ap.add_argument("--cluster_tries", type=int, default=8)
    ap.add_argument("--cluster_temperature", type=float, default=0.3)
    ap.add_argument("--cluster_num_predict", type=int, default=1800)

    ap.add_argument("--summary_tries", type=int, default=5)
    ap.add_argument("--summary_temperature", type=float, default=0.3)
    ap.add_argument("--summary_num_predict", type=int, default=500)

    args = ap.parse_args()

    out_path = Path(args.out)
    debug_dir = out_path.resolve().parent / "_debug_llm_only_no_constraints"
    debug_dir.mkdir(parents=True, exist_ok=True)

    schema_obj = json.load(open(args.schema, "r", encoding="utf-8"))
    schema_labels = extract_node_labels_from_schema(schema_obj)

    if args.k > len(schema_labels):
        print(f"[WARN] k={args.k} > #labels={len(schema_labels)}; reducing k to {len(schema_labels)}")
        args.k = len(schema_labels)

    # ---- LLM clustering (no constraints) ----
    cluster_prompt = build_llm_clustering_prompt(schema_labels, args.k)

    schema_hint = f"""
{{
  "explanation": "string",
  "clusters": [
    {{"id":"G1","labels":[...]}},
    ...
    {{"id":"G{args.k}","labels":[...]}}
  ]
}}
""".strip()

    clusters = None
    explanation = ""
    last_raw = ""

    for attempt in range(1, args.cluster_tries + 1):
        raw = ollama_chat_content(
            host=args.ollama_host,
            model=args.ollama_model,
            prompt=cluster_prompt,
            temperature=args.cluster_temperature,
            num_predict=args.cluster_num_predict,
            timeout_sec=args.timeout,
            stop=["```", "\nfinal_checklist", "\n\n\n"],
            format_json=True,
        )
        last_raw = raw

        obj = parse_or_repair_json(
            raw_text=raw,
            host=args.ollama_host,
            model=args.ollama_model,
            schema_hint=schema_hint,
            timeout=args.timeout,
            debug_dir=debug_dir,
            tag=f"cluster_k{args.k}_attempt_{attempt}",
        )

        if isinstance(obj, dict) and isinstance(obj.get("clusters"), list):
            explanation = str(obj.get("explanation") or "").strip()
            parsed: List[List[str]] = []
            for c in obj["clusters"]:
                if isinstance(c, dict) and isinstance(c.get("labels"), list):
                    parsed.append([x.strip() for x in c["labels"] if isinstance(x, str) and x.strip()])

            clusters = ensure_disjoint_cover(parsed, schema_labels, args.k)
            break

        cluster_prompt = f"""
Your previous JSON was invalid.

Return ONLY VALID JSON matching:
{schema_hint}

Rules:
- Exactly {args.k} clusters, all non-empty
- Use ONLY schema_labels
- Cover all schema_labels exactly once (no duplicates, no missing)

schema_labels:
{json.dumps(schema_labels, ensure_ascii=False)}
""".strip()

    if clusters is None:
        (debug_dir / f"cluster_failed_k{args.k}.txt").write_text(last_raw, encoding="utf-8")
        raise RuntimeError(f"LLM clustering failed after retries for k={args.k}.")

    # ---- Build groups output (similar structure) ----
    groups: List[Dict[str, Any]] = []
    for i, cl in enumerate(clusters):
        groups.append({
            "id": f"G{i+1}",
            "cluster_name": "",
            "cluster_explanation": "",
            "unified_label": "",
            "lhs_signatures": [sorted(cl)],
            "rhs_needed_labels": [],                 # no constraints
            "node_labels_for_local_check": sorted(cl),  # just the cluster itself
            "preserved_ggds": [],                    # no constraints
            "fully_local_ggds": [],                  # no constraints
        })

    # ---- LLM summaries per group ----
    used_unified: List[str] = []
    summary_schema_hint = """
{
  "unified_label": "string",
  "cluster_name": "string",
  "cluster_explanation": "string"
}
""".strip()

    ok_summaries = 0
    for g in groups:
        gid = g["id"]
        cluster_labels = g["lhs_signatures"][0]

        prompt = build_cluster_summary_prompt(
            schema_labels=schema_labels,
            group_id=gid,
            cluster_labels=cluster_labels,
            used_unified_labels=used_unified,
        )

        best = None
        for attempt in range(1, args.summary_tries + 1):
            raw = ollama_chat_content(
                host=args.ollama_host,
                model=args.ollama_model,
                prompt=prompt,
                temperature=args.summary_temperature,
                num_predict=args.summary_num_predict,
                timeout_sec=args.timeout,
                stop=["```", "\nfinal_checklist", "\n\n\n"],
                format_json=True,
            )

            obj = parse_or_repair_json(
                raw_text=raw,
                host=args.ollama_host,
                model=args.ollama_model,
                schema_hint=summary_schema_hint,
                timeout=args.timeout,
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
                expl = f"This cluster groups semantically related labels: {', '.join(cluster_labels)}."
            elif len(sents) > 2:
                expl = ". ".join(sents[:2]) + "."
            else:
                if not expl.endswith((".", "!", "?")):
                    expl = expl + "."

            best = (unified, cname, expl)
            ok_summaries += 1
            break

        if best is None:
            unified = clean_unified_label(gid)
            if unified in used_unified:
                unified = f"{unified}_{len(used_unified)+1}"
            cname = f"{gid} Cluster"
            expl = f"This cluster groups semantically related labels: {', '.join(cluster_labels)}."
            best = (unified, cname, expl)

        g["unified_label"] = best[0]
        g["cluster_name"] = best[1]
        g["cluster_explanation"] = best[2]
        used_unified.append(best[0])

    out_obj = {
        "explanation": (
            "LLM grouped schema node labels using only semantic similarity (no constraints). "
            "No optimizer/deterministic refinement was applied. Output is normalized into a disjoint cover for comparability."
        ),
        "k": args.k,
        "ollama_model": args.ollama_model,
        "llm_used_for_clustering": True,
        "llm_used_for_cluster_summaries": True,
        "llm_summaries_ok": ok_summaries,
        "llm_clustering_explanation": explanation,
        "optimizer_details": {  # kept only for output compatibility
            "note": "No constraints / no optimization. Metrics not applicable.",
            "sizes": [len(g["lhs_signatures"][0]) for g in groups],
        },
        "groups": groups,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK wrote:", str(out_path))


if __name__ == "__main__":
    main()