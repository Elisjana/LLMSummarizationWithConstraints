#!/usr/bin/env python3
"""
LLM-based (Ollama/Gemma) constraint-preserving schema summarization from GGDs.

What it does:
  1) Reads GGDs from JSONL (structured lhs/rhs OR text summaries)
  2) Extracts a minimal schema footprint required by all GGDs:
        - node labels
        - edge labels (+ endpoints when inferable)
  3) Calls Ollama (Gemma) to produce a short schema summary that preserves GGDs
  4) Outputs JSON + Markdown

Requirements:
  - Ollama running locally (default http://localhost:11434)
  - A Gemma model pulled in Ollama (e.g., gemma2:2b, gemma2:2b-instruct, gemma:2b, etc.)
  - Python 3.9+

Usage:
  python llm_schema_summarizer_ollama.py \
    --input summaryOutput/gemma_outputs.jsonl \
    --ollama_model gemma2:2b-instruct \
    --out_json summaryOutput/schema_summary.json \
    --out_md summaryOutput/schema_summary.md

If your input is raw structured GGDs:
  python llm_schema_summarizer_ollama.py \
    --input constraints/LDBC/v1/constraints.jsonl \
    --ollama_model gemma2:2b-instruct \
    --out_json summaryOutput/schema_summary.json \
    --out_md summaryOutput/schema_summary.md
"""

import argparse
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# -------- Best-effort regex for text summaries --------
EDGE_PATTERN = re.compile(
    r"(?P<edge>[A-Za-z0-9_]+)\s*\(\s*(?P<src>[A-Za-z0-9_]+)\s*(?:->|→|-)\s*(?P<dst>[A-Za-z0-9_]+)\s*\)"
)

NODES_LIST_PATTERN = re.compile(r"(?:Nodes|Node types)\s*:\s*([A-Za-z0-9_,\s]+)", re.IGNORECASE)
EDGES_LIST_PATTERN = re.compile(r"(?:Edges|Edge types)\s*:\s*([A-Za-z0-9_,\s]+)", re.IGNORECASE)


@dataclass(frozen=True)
class EdgeType:
    label: str
    src: Optional[str] = None
    dst: Optional[str] = None

    def key(self) -> Tuple[str, Optional[str], Optional[str]]:
        return (self.label, self.src, self.dst)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}")


def pick_ggd_id(obj: Dict[str, Any]) -> str:
    return str(obj.get("id") or obj.get("cid") or obj.get("ggd_id") or obj.get("name") or "UNKNOWN")


def pick_summary_text(obj: Dict[str, Any]) -> Optional[str]:
    # common keys in LLM output pipelines
    for k in ("summary", "output", "text", "completion", "generated_text", "response"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # nested variants
    for k in ("result", "data"):
        v = obj.get(k)
        if isinstance(v, dict):
            for kk in ("summary", "text", "output"):
                vv = v.get(kk)
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()
    return None


def _get_node_label(node_obj: Dict[str, Any]) -> Optional[str]:
    for k in ("Node label", "nodeLabel", "label", "node_label"):
        if k in node_obj and isinstance(node_obj[k], str) and node_obj[k].strip():
            return node_obj[k].strip()
    return None


def _get_edge_label(edge_obj: Dict[str, Any]) -> Optional[str]:
    for k in ("Edge label", "edgeLabel", "label", "edge_label"):
        if k in edge_obj and isinstance(edge_obj[k], str) and edge_obj[k].strip():
            return edge_obj[k].strip()
    return None


def extract_from_structured_ggd(obj: Dict[str, Any]) -> Tuple[Set[str], Set[EdgeType]]:
    nodes: Set[str] = set()
    edges: Set[EdgeType] = set()

    def is_var(x: Optional[str]) -> bool:
        # typical vars: n0, n1, A, B, etc. We'll treat letter+digits as var.
        return bool(x) and bool(re.fullmatch(r"[A-Za-z]\d+", x))

    def harvest(side: Any) -> None:
        nonlocal nodes, edges
        if not isinstance(side, dict):
            return

        var_to_label: Dict[str, str] = {}

        for n in side.get("nodes", []) or []:
            if not isinstance(n, dict):
                continue
            lbl = _get_node_label(n)
            if lbl:
                nodes.add(lbl)
            v = n.get("var")
            if isinstance(v, str) and lbl:
                var_to_label[v] = lbl

        for e in side.get("edges", []) or []:
            if not isinstance(e, dict):
                continue
            elbl = _get_edge_label(e)
            if not elbl:
                continue

            src = e.get("src")
            dst = e.get("dst")

            src_lbl = var_to_label.get(src, src) if isinstance(src, str) else None
            dst_lbl = var_to_label.get(dst, dst) if isinstance(dst, str) else None

            src_final = None if is_var(src_lbl) else (src_lbl.strip() if isinstance(src_lbl, str) else None)
            dst_final = None if is_var(dst_lbl) else (dst_lbl.strip() if isinstance(dst_lbl, str) else None)

            edges.add(EdgeType(elbl, src_final, dst_final))

    harvest(obj.get("lhs"))
    harvest(obj.get("rhs"))

    return nodes, edges


def extract_from_text_summary(text: str) -> Tuple[Set[str], Set[EdgeType]]:
    nodes: Set[str] = set()
    edges: Set[EdgeType] = set()

    for m in EDGE_PATTERN.finditer(text):
        elbl = m.group("edge").strip()
        src = m.group("src").strip()
        dst = m.group("dst").strip()
        nodes.add(src)
        nodes.add(dst)
        edges.add(EdgeType(elbl, src, dst))

    mn = NODES_LIST_PATTERN.search(text)
    if mn:
        for part in mn.group(1).split(","):
            p = part.strip()
            if p:
                nodes.add(p)

    me = EDGES_LIST_PATTERN.search(text)
    if me:
        for part in me.group(1).split(","):
            p = part.strip()
            if p:
                edges.add(EdgeType(p, None, None))

    return nodes, edges


def build_minimal_schema_footprint(input_jsonl: Path, max_rules_for_prompt: int = 40) -> Dict[str, Any]:
    required_nodes: Set[str] = set()
    required_edges: Dict[Tuple[str, Optional[str], Optional[str]], EdgeType] = {}
    per_ggd_support: Dict[str, Dict[str, Any]] = {}
    ggd_rule_texts: List[str] = []

    for obj in read_jsonl(input_jsonl):
        gid = pick_ggd_id(obj)

        nodes_s: Set[str] = set()
        edges_s: Set[EdgeType] = set()

        if isinstance(obj.get("lhs"), dict) or isinstance(obj.get("rhs"), dict):
            n2, e2 = extract_from_structured_ggd(obj)
            nodes_s |= n2
            edges_s |= e2

        summary_text = pick_summary_text(obj)
        if summary_text:
            n3, e3 = extract_from_text_summary(summary_text)
            nodes_s |= n3
            edges_s |= e3

        if not nodes_s and not edges_s:
            continue

        required_nodes |= nodes_s
        for e in edges_s:
            required_edges[e.key()] = e

        # keep a compact rule representation for the LLM prompt
        # prefer summary text if available, otherwise serialize a small structured view
        if summary_text:
            rule_repr = summary_text
        else:
            # minimal structured excerpt
            rule_repr = json.dumps(
                {
                    "id": gid,
                    "lhs": obj.get("lhs", {}),
                    "rhs": obj.get("rhs", {}),
                },
                ensure_ascii=False
            )
        ggd_rule_texts.append(f"[{gid}] {rule_repr}")

        per_ggd_support[gid] = {
            "requires_nodes": sorted(nodes_s),
            "requires_edges": sorted(
                [
                    f"{e.label}({e.src or '?'}→{e.dst or '?'})" if (e.src or e.dst) else e.label
                    for e in edges_s
                ]
            ),
        }

    edge_types = [
        {"label": e.label, "from": e.src, "to": e.dst}
        for e in sorted(required_edges.values(), key=lambda x: (x.label, x.src or "", x.dst or ""))
    ]

    # limit prompt size (important for small Gemma models)
    ggd_rule_texts = ggd_rule_texts[:max_rules_for_prompt]

    return {
        "node_types": sorted(required_nodes),
        "edge_types": edge_types,
        "constraint_support": per_ggd_support,
        "ggd_rule_texts_for_prompt": ggd_rule_texts,
    }


# ---------- Ollama call (no external deps) ----------
def ollama_generate(
    model: str,
    prompt: str,
    host: str = "http://localhost:11434",
    temperature: float = 0.2,
    num_predict: int = 350,
) -> str:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            out = json.loads(resp.read().decode("utf-8", errors="replace"))
            return (out.get("response") or "").strip()
    except Exception as e:
        raise RuntimeError(
            f"Failed to call Ollama at {url}. "
            f"Make sure Ollama is running and the model exists. Error: {e}"
        )


def build_llm_prompt(min_schema: Dict[str, Any]) -> str:
    nodes = min_schema["node_types"]
    edges = min_schema["edge_types"]
    rules = min_schema["ggd_rule_texts_for_prompt"]

    # We force the model to ONLY use these labels, to avoid hallucinating schema terms.
    # We also demand a JSON block + a short narrative.
    return f"""
You are a property-graph schema summarization assistant.

TASK:
Produce a compact schema summary that PRESERVES the meaning of the given GGDs.
No instance data exists. Only schema concepts are allowed.

STRICT RULES:
- Use ONLY the node labels and edge labels that appear in the provided schema footprint.
- Do NOT invent new node labels, edge labels, or property names.
- Preserve direction of edges when endpoints are provided.
- Output MUST contain:
  (A) A JSON object with fields:
      - node_types: [..]
      - edge_types: [{{
          "label": "...",
          "from": "Label" | null,
          "to": "Label" | null,
          "meaning": "short description"
        }}]
      - preserved_rules: [{{"id":"...","meaning":"... (1-2 sentences)"}}]
  (B) A short human-readable paragraph (max 120 words) explaining what this schema represents.

SCHEMA FOOTPRINT (authoritative):
Node labels ({len(nodes)}):
{json.dumps(nodes, ensure_ascii=False, indent=2)}

Edge types ({len(edges)}):
{json.dumps(edges, ensure_ascii=False, indent=2)}

GGDs (each must be preserved):
{chr(10).join(rules)}
""".strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL containing structured GGDs and/or text summaries.")
    ap.add_argument("--ollama_model", required=True, help='Ollama model name, e.g. "gemma2:2b-instruct".')
    ap.add_argument("--ollama_host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                    help='Ollama host (default http://localhost:11434). Can also set env OLLAMA_HOST.')
    ap.add_argument("--out_json", required=True, help="Output JSON file (machine-friendly schema summary).")
    ap.add_argument("--out_md", required=True, help="Output Markdown file (LLM schema summary).")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--num_predict", type=int, default=350)
    ap.add_argument("--max_rules_for_prompt", type=int, default=40,
                    help="Limit number of GGDs included in prompt (important for small models).")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    # 1) Build minimal footprint from GGDs/summaries
    min_schema = build_minimal_schema_footprint(input_path, max_rules_for_prompt=args.max_rules_for_prompt)

    # 2) Ask Gemma (via Ollama) to generate the human schema summary + structured JSON
    prompt = build_llm_prompt(min_schema)
    llm_text = ollama_generate(
        model=args.ollama_model,
        prompt=prompt,
        host=args.ollama_host,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )

    # 3) Save machine JSON summary (footprint + raw LLM output)
    result = {
        "summary_type": "llm_constraint_preserving_schema_summary",
        "input": str(input_path),
        "ollama": {"host": args.ollama_host, "model": args.ollama_model},
        "minimal_schema_footprint": {
            "node_types": min_schema["node_types"],
            "edge_types": min_schema["edge_types"],
        },
        "constraint_support": min_schema["constraint_support"],
        "llm_output": llm_text,
        "notes": [
            "The minimal_schema_footprint is computed from GGDs/summaries and is guaranteed not to hallucinate.",
            "The llm_output is the readable schema summary produced by Gemma via Ollama.",
        ],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # 4) Save Markdown for reading (raw LLM output + footprint)
    md = []
    md.append("# Constraint-Preserving Schema Summary (Gemma via Ollama)\n")
    md.append(f"- Model: `{args.ollama_model}`")
    md.append(f"- Input: `{input_path}`\n")
    md.append("## Minimal schema footprint (computed)\n")
    md.append("### Node types\n")
    for n in min_schema["node_types"]:
        md.append(f"- {n}")
    md.append("\n### Edge types\n")
    for e in min_schema["edge_types"]:
        frm = e.get("from") or "?"
        to = e.get("to") or "?"
        if e.get("from") or e.get("to"):
            md.append(f"- {e['label']} ({frm} → {to})")
        else:
            md.append(f"- {e['label']}")
    md.append("\n## LLM schema summary (must preserve GGDs)\n")
    md.append(llm_text.strip() if llm_text.strip() else "_(empty response from model)_")
    md.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"[OK] Wrote JSON: {out_json}")
    print(f"[OK] Wrote Markdown: {out_md}")


if __name__ == "__main__":
    main()