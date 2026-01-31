#!/usr/bin/env python3
"""
Constraint-preserving schema summarization (Option B1: canonical per-GGD signatures)
using Ollama (Gemma).

GOAL (your request):
- For EVERY relation in edge_types, produce an LLM-style meaning like:
    "Person works at an organisation."
  instead of patterny:
    "Person WORK_AT Organisation"
- Still GUARANTEE no empty meanings:
  - If LLM output is invalid/empty/patterny -> fallback to safe pattern string.

KEY FEATURES:
- B1 footprint extraction from structured GGDs (lhs/rhs nodes+edges)
- Edge meanings generated in batches via Ollama
- Forces JSON output via payload["format"] = "json"
- Retries once if response is invalid / yields no good meanings
- Validates meanings (grammatical short sentence ending with ".")
- Writes:
  1) compact JSON
  2) Markdown summary
  3) optional debug JSON with prompts + raw outputs

NOTE:
- preserved_rules remain canonical signatures by default (as in your original).
  If you also want meanings inside preserved_rules, I can extend it afterwards.
"""

import argparse
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# -------------------- Utilities --------------------

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


def _get_node_label(node_obj: Dict[str, Any]) -> Optional[str]:
    for k in ("Node label", "nodeLabel", "label", "node_label"):
        v = node_obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _get_edge_label(edge_obj: Dict[str, Any]) -> Optional[str]:
    for k in ("Edge label", "edgeLabel", "label", "edge_label"):
        v = edge_obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


@dataclass(frozen=True)
class EdgeType:
    label: str
    src: Optional[str] = None
    dst: Optional[str] = None

    def key(self) -> Tuple[str, Optional[str], Optional[str]]:
        return (self.label, self.src, self.dst)


def _is_var(x: Optional[str]) -> bool:
    # typical vars: n0, n1, A1 etc.
    return bool(x) and bool(re.fullmatch(r"[A-Za-z]\d+", x))


def edge_sig(label: str, src: Optional[str], dst: Optional[str]) -> str:
    s = src if src else "?"
    d = dst if dst else "?"
    return f"{label}({s}->{d})"


def chunk_list(xs: List[Any], n: int) -> List[List[Any]]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


# -------------------- Extract from structured GGDs --------------------

def extract_side_structured(side: Any) -> Tuple[Set[str], List[EdgeType]]:
    """
    Returns:
      node labels set
      typed edges list with src/dst labels when resolvable
    """
    nodes: Set[str] = set()
    edges: List[EdgeType] = []

    if not isinstance(side, dict):
        return nodes, edges

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

        src_final = None if _is_var(src_lbl) else (src_lbl.strip() if isinstance(src_lbl, str) else None)
        dst_final = None if _is_var(dst_lbl) else (dst_lbl.strip() if isinstance(dst_lbl, str) else None)

        edges.append(EdgeType(elbl, src_final, dst_final))

    return nodes, edges


def extract_structured_ggd(obj: Dict[str, Any]) -> Tuple[Set[str], List[EdgeType], List[EdgeType]]:
    """
    Returns:
      nodes used (union lhs/rhs)
      lhs edges typed
      rhs edges typed
    """
    lhs_nodes, lhs_edges = extract_side_structured(obj.get("lhs"))
    rhs_nodes, rhs_edges = extract_side_structured(obj.get("rhs"))
    all_nodes = set(lhs_nodes) | set(rhs_nodes)
    return all_nodes, lhs_edges, rhs_edges


# -------------------- Build footprint + preserved rules (B1) --------------------

def _dedupe_edges(edges: List[EdgeType]) -> List[EdgeType]:
    seen: Set[Tuple[str, Optional[str], Optional[str]]] = set()
    out: List[EdgeType] = []
    for e in edges:
        if e.key() in seen:
            continue
        seen.add(e.key())
        out.append(e)
    return out


def build_footprint_and_preserved_rules(input_jsonl: Path) -> Dict[str, Any]:
    required_nodes: Set[str] = set()
    required_edges: Dict[Tuple[str, Optional[str], Optional[str]], EdgeType] = {}

    preserved_rules: List[Dict[str, Any]] = []
    total = 0

    for obj in read_jsonl(input_jsonl):
        # Only structured GGDs for B1
        if not isinstance(obj.get("lhs"), dict) or not isinstance(obj.get("rhs"), dict):
            continue

        total += 1
        gid = pick_ggd_id(obj)

        nodes, lhs_edges, rhs_edges = extract_structured_ggd(obj)
        required_nodes |= nodes

        for e in (lhs_edges + rhs_edges):
            required_edges[e.key()] = e

        lhs_sigs = sorted({edge_sig(e.label, e.src, e.dst) for e in _dedupe_edges(lhs_edges)})
        rhs_sigs = sorted({edge_sig(e.label, e.src, e.dst) for e in _dedupe_edges(rhs_edges)})

        preserved_rules.append({"id": gid, "lhs": lhs_sigs, "rhs": rhs_sigs})

    edge_types = [
        {"label": e.label, "from": e.src, "to": e.dst}
        for e in sorted(required_edges.values(), key=lambda x: (x.label, x.src or "", x.dst or ""))
    ]

    return {
        "node_types": sorted(required_nodes),
        "edge_types": edge_types,
        "preserved_rules": preserved_rules,
        "ggds_total": total
    }


# -------------------- Ollama calls --------------------

def ollama_generate(
    model: str,
    prompt: str,
    host: str = "http://localhost:11434",
    temperature: float = 0.1,
    num_predict: int = 220,
    http_timeout: int = 900,
) -> str:
    """
    Calls Ollama /api/chat and forces JSON output using payload["format"]="json".
    """
    host = host.rstrip("/")
    chat_url = host + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",  # IMPORTANT: ask Ollama to return JSON string
        "options": {"temperature": temperature, "num_predict": num_predict},
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(chat_url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=http_timeout) as resp:
        out = json.loads(resp.read().decode("utf-8", errors="replace"))
    msg = out.get("message") or {}
    return str(msg.get("content") or "").strip()


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust extraction: sometimes models still wrap output.
    """
    if not text:
        return None

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE | re.MULTILINE
    ).strip()

    # direct JSON
    try:
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return json.loads(cleaned)
    except Exception:
        pass

    # fallback: extract first {...}
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            return None
    return None


# -------------------- Prompts (improved for better English meanings) --------------------

def build_edge_meanings_prompt(node_types: List[str], edge_batch: List[Dict[str, Any]]) -> str:
    template = {
        "edge_types": [
            {
                "label": "WORK_AT",
                "from": "Person",
                "to": "Organisation",
                "meaning": "Person works at an organisation."
            }
        ]
    }

    examples = {
        "examples": [
            {"label": "WORK_AT", "from": "Person", "to": "Organisation",
             "meaning": "Person works at an organisation."},
            {"label": "STUDY_AT", "from": "Person", "to": "Organisation",
             "meaning": "Person studies at an organisation."},
            {"label": "REPLY_OF", "from": "Comment", "to": "Post",
             "meaning": "Comment replies to a post."},
            {"label": "HAS_INTEREST", "from": "Person", "to": "Tag",
             "meaning": "Person is interested in a tag."}
        ]
    }

    return (
        "You write short natural-language meanings for property-graph relations.\n\n"
        "HARD RULES:\n"
        "- Use ONLY the provided node labels and edge labels.\n"
        "- Do NOT invent new labels or properties.\n"
        "- Output MUST be ONLY valid JSON (no prose, no markdown).\n"
        "- Each meaning MUST be a grammatical English sentence ending with a period.\n"
        "- Keep each meaning <= 12 words.\n"
        "- Do NOT output patterns like: \"Person WORK_AT Organisation\".\n"
        "- Prefer: \"A <verb phrase> a/an <to>\".\n\n"
        "Return JSON in this shape:\n"
        f"{json.dumps(template, ensure_ascii=False, indent=2)}\n\n"
        "Style examples (follow this style closely):\n"
        f"{json.dumps(examples, ensure_ascii=False, indent=2)}\n\n"
        "Node labels:\n"
        f"{json.dumps(node_types, ensure_ascii=False)}\n\n"
        "Edge types to describe:\n"
        f"{json.dumps(edge_batch, ensure_ascii=False)}\n"
    )


def build_schema_text_prompt(node_types: List[str], edges_with_meaning: List[Dict[str, Any]]) -> str:
    template = {"schema_summary_text": "one paragraph, max 120 words"}
    return (
        "You are a property-graph schema summarization assistant.\n\n"
        "STRICT RULES:\n"
        "- Use ONLY the given node labels and edge labels.\n"
        "- Do NOT invent new labels or properties.\n"
        "- Output MUST be ONLY valid JSON (no markdown).\n\n"
        "Return JSON in this shape:\n"
        f"{json.dumps(template, ensure_ascii=False, indent=2)}\n\n"
        "Node labels:\n"
        f"{json.dumps(node_types, ensure_ascii=False)}\n\n"
        "Edge types (authoritative, with meanings):\n"
        f"{json.dumps(edges_with_meaning, ensure_ascii=False)}\n"
    )


# -------------------- Meaning validation + fallback --------------------

def is_good_llm_meaning(m: str) -> bool:
    """
    Accept good English meaning like:
      "Person works at an organisation."
    Reject pattern-like meaning like:
      "Person WORK_AT Organisation"
    """
    if not m:
        return False
    s = m.strip()

    # must end with period
    if not s.endswith("."):
        return False

    # should have enough tokens
    if len(s.split()) < 3:
        return False

    # reject ALL_CAPS tokens (usually the edge label copied)
    if re.search(r"\b[A-Z0-9_]{3,}\b", s):
        return False

    # keep short
    if len(s.split()) > 12:
        return False

    return True


def fallback_meaning(label: str, src: Optional[str], dst: Optional[str]) -> str:
    """
    Last resort only (if LLM fails badly).
    """
    s = src or "?"
    d = dst or "?"
    return f"{s} {label} {d}"


def normalize_edge_meanings(returned: List[Dict[str, Any]], batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Guarantees:
      - One output meaning for EACH edge in batch
      - Meaning is LLM-style when valid; otherwise safe fallback
    """
    idx: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
    for e in returned:
        if not isinstance(e, dict):
            continue
        idx[(e.get("label"), e.get("from"), e.get("to"))] = e

    out: List[Dict[str, Any]] = []
    for b in batch:
        k = (b.get("label"), b.get("from"), b.get("to"))
        got = idx.get(k, {})
        meaning = (got.get("meaning") or "").strip()

        if not is_good_llm_meaning(meaning):
            meaning = fallback_meaning(str(b.get("label")), b.get("from"), b.get("to"))

        out.append({
            "label": b.get("label"),
            "from": b.get("from"),
            "to": b.get("to"),
            "meaning": meaning
        })
    return out


# -------------------- Main --------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL (structured GGDs).")
    ap.add_argument("--ollama_model", required=True, help='Ollama model name, e.g. "gemma2:2b".')
    ap.add_argument("--ollama_host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--out_json", required=True, help="Output compact JSON file.")
    ap.add_argument("--out_md", required=True, help="Output Markdown file.")
    ap.add_argument("--out_debug_json", default=None, help="Optional debug JSON output.")
    ap.add_argument("--edge_batch_size", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.1, help="Lower is more consistent for JSON/meaning.")
    ap.add_argument("--num_predict", type=int, default=220)
    ap.add_argument("--http_timeout", type=int, default=300)
    args = ap.parse_args()

    input_path = Path(args.input)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_debug = Path(args.out_debug_json) if args.out_debug_json else None

    # 1) Build footprint + B1 preserved_rules
    built = build_footprint_and_preserved_rules(input_path)
    node_types: List[str] = built["node_types"]
    edge_types: List[Dict[str, Any]] = built["edge_types"]
    preserved_rules: List[Dict[str, Any]] = built["preserved_rules"]
    ggds_total: int = built["ggds_total"]

    # 2) Ask LLM for edge meanings in batches (FOR ALL edge_types)
    edges_with_meaning: List[Dict[str, Any]] = []
    debug_prompts: List[Dict[str, Any]] = []
    debug_raw: List[Dict[str, Any]] = []

    for idx, batch in enumerate(chunk_list(edge_types, args.edge_batch_size), start=1):
        prompt = build_edge_meanings_prompt(node_types, batch)

        def call_once() -> Tuple[str, Optional[List[Dict[str, Any]]]]:
            txt = ollama_generate(
                model=args.ollama_model,
                prompt=prompt,
                host=args.ollama_host,
                temperature=args.temperature,
                num_predict=args.num_predict,
                http_timeout=args.http_timeout,
            )
            parsed = extract_first_json_object(txt) or {}
            returned = parsed.get("edge_types") if isinstance(parsed.get("edge_types"), list) else None
            return txt, returned

        # attempt 1
        txt, returned = call_once()
        debug_prompts.append({"type": "edge_meanings", "chunk": idx, "prompt": prompt})
        debug_raw.append({"type": "edge_meanings", "chunk": idx, "text": txt})

        # decide retry: no returned or no good meanings
        needs_retry = True
        if returned:
            if any(isinstance(e, dict) and is_good_llm_meaning((e.get("meaning") or "").strip()) for e in returned):
                needs_retry = False

        if needs_retry:
            txt_r, returned_r = call_once()
            debug_raw.append({"type": "edge_meanings_retry", "chunk": idx, "text": txt_r})
            if returned_r:
                returned = returned_r

        normalized = normalize_edge_meanings(returned or [], batch)
        edges_with_meaning.extend(normalized)

    # 3) Ask LLM for one compact schema paragraph
    prompt_txt = build_schema_text_prompt(node_types, edges_with_meaning)
    txt2 = ollama_generate(
        model=args.ollama_model,
        prompt=prompt_txt,
        host=args.ollama_host,
        temperature=args.temperature,
        num_predict=args.num_predict,
        http_timeout=args.http_timeout,
    )
    debug_prompts.append({"type": "schema_text", "prompt": prompt_txt})
    debug_raw.append({"type": "schema_text", "text": txt2})

    parsed2 = extract_first_json_object(txt2) or {}
    schema_summary_text = str(parsed2.get("schema_summary_text") or "").strip()

    # 4) Coverage (B1-style)
    ggds_preserved = len(preserved_rules)

    compact = {
        "schema_summary": {
            "node_types": node_types,
            "edge_types": edges_with_meaning,
            "schema_summary_text": schema_summary_text,
        },
        "constraint_preservation": {
            "coverage": {
                "ggds_total": ggds_total,
                "ggds_preserved": ggds_preserved,
            },
            "preservation_style": "canonical_signatures",
            "preserved_rules": preserved_rules,
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown
    md_lines: List[str] = []
    md_lines.append("# Constraint-Preserving Schema Summary (B1)\n")
    md_lines.append(f"- Model: `{args.ollama_model}`")
    md_lines.append(f"- Input: `{input_path}`")
    md_lines.append(f"- GGDs preserved: **{ggds_preserved}/{ggds_total}**\n")

    md_lines.append("## Node types")
    for n in node_types:
        md_lines.append(f"- {n}")

    md_lines.append("\n## Edge types (LLM meanings, fallback only if necessary)")
    for e in edges_with_meaning:
        md_lines.append(
            f"- {e['label']} ({e.get('from') or '?'} → {e.get('to') or '?'}) — {e.get('meaning') or ''}".rstrip()
        )

    md_lines.append("\n## Schema summary text")
    md_lines.append(schema_summary_text if schema_summary_text else "_(empty)_")

    md_lines.append("\n## Constraint preservation (canonical signatures)")
    md_lines.append(f"- ggds_total: {ggds_total}")
    md_lines.append(f"- ggds_preserved: {ggds_preserved}")
    md_lines.append("\n(See JSON for full preserved_rules list.)\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    # Debug
    if out_debug:
        dbg = {
            "minimal_schema_footprint": {"node_types": node_types, "edge_types": edge_types},
            "ollama": {"host": args.ollama_host, "model": args.ollama_model},
            "prompts": debug_prompts,
            "raw_outputs": debug_raw,
            "final_compact_output": compact,
        }
        out_debug.parent.mkdir(parents=True, exist_ok=True)
        out_debug.write_text(json.dumps(dbg, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Wrote JSON: {out_json}")
    print(f"[OK] Wrote MD:   {out_md}")
    if out_debug:
        print(f"[OK] Wrote DEBUG JSON: {out_debug}")


if __name__ == "__main__":
    main()