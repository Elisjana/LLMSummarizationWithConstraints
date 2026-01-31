# pgsToJSON.py
# ------------------------------------------------------------
# Parse your LDBC-style .pgs schema into schema.json
#
# Supports:
# - CREATE NODE TYPE ( TypeName : Label { ... } );
#   with OPTIONAL properties
# - CREATE EDGE TYPE
#   ( : A | B ) - [ REL { prop Type } ] -> ( : C | D );
#   with optional edge properties inside { ... }
# - (Optional) CREATE GRAPH TYPE ... { ... };
# ------------------------------------------------------------

import re
import json
from pathlib import Path
from typing import Dict, List, Any


# ---------- Regex patterns ----------
NODE_BLOCK_RE = re.compile(r"CREATE\s+NODE\s+TYPE\s*\((.*?)\)\s*;", re.S | re.I)

# Header like: PersonType : Person
NODE_HEADER_RE = re.compile(r"^\s*(\w+)\s*:\s*(\w+)\s*", re.S)

# Property line like: OPTIONAL content String,   or: id String,
NODE_PROP_LINE_RE = re.compile(r"^\s*(OPTIONAL\s+)?(\w+)\s+(\w+)\s*(?:,)?\s*$", re.I)

EDGE_BLOCK_RE = re.compile(r"CREATE\s+EDGE\s+TYPE\s*(.*?)\s*;", re.S | re.I)

# Source node labels inside: ( : Post | Comment )
EDGE_SRC_RE = re.compile(r"\(\s*:\s*([^)]+?)\s*\)", re.S)

# Relationship inside: [ KNOWS { creationDate String } ]
EDGE_REL_RE = re.compile(r"\[\s*([A-Z0-9_]+)\s*(\{.*?\})?\s*\]", re.S)

# Target node labels inside: -> ( : Person )
EDGE_DST_RE = re.compile(r"->\s*\(\s*:\s*([^)]+?)\s*\)", re.S)

GRAPH_TYPE_RE = re.compile(r"CREATE\s+GRAPH\s+TYPE\s+(\w+)\s+(\w+)\s*\{(.*?)\}\s*;", re.S | re.I)


# ---------- Helpers ----------
def _split_union_labels(label_text: str) -> List[str]:
    """
    "Organisation | Post | Person" -> ["Organisation","Post","Person"]
    """
    return [p.strip() for p in label_text.split("|") if p.strip()]


def _parse_node_properties(block_inside_parens: str) -> List[Dict[str, Any]]:
    """
    Extract properties from {...} in a node type block.
    """
    props: List[Dict[str, Any]] = []

    lbrace = block_inside_parens.find("{")
    rbrace = block_inside_parens.rfind("}")
    if lbrace == -1 or rbrace == -1 or rbrace <= lbrace:
        return props

    body = block_inside_parens[lbrace + 1 : rbrace]
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = NODE_PROP_LINE_RE.match(line)
        if not m:
            continue
        optional_kw, name, dtype = m.groups()
        props.append(
            {
                "name": name,
                "type": dtype,
                "optional": bool(optional_kw),
            }
        )
    return props


def _parse_edge_properties(curly_block: str) -> List[Dict[str, Any]]:
    """
    Extract properties from "{ ... }" inside edge brackets.
    Example:
      { joinDate String }
      { creationDate String }
    """
    props: List[Dict[str, Any]] = []
    if not curly_block:
        return props

    inner = curly_block.strip()
    if inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1]

    # allow commas or newlines
    for chunk in re.split(r",|\n", inner):
        line = chunk.strip()
        if not line:
            continue
        m = re.match(r"^(\w+)\s+(\w+)\s*$", line)
        if not m:
            continue
        name, dtype = m.groups()
        props.append({"name": name, "type": dtype, "optional": False})
    return props


# ---------- Parsers ----------
def parse_nodes(text: str) -> Dict[str, Any]:
    """
    Returns:
      {
        "PersonType": {"label":"Person","properties":[...]},
        ...
      }
    """
    nodes: Dict[str, Any] = {}

    for block in NODE_BLOCK_RE.findall(text):
        # block is content inside (...) for node type
        header_part = block[: block.find("{")] if "{" in block else block
        m = NODE_HEADER_RE.search(header_part)
        if not m:
            continue
        type_name, label = m.groups()

        nodes[type_name] = {
            "label": label,
            "properties": _parse_node_properties(block),
        }

    return nodes


def parse_edges(text: str) -> Dict[str, Any]:
    """
    Returns:
      {
        "HAS_CREATOR": {"from":["Post","Comment"],"to":["Person"],"properties":[]},
        "HAS_MEMBER":  {"from":["Forum"],"to":["Person"],"properties":[{"name":"joinDate",...}]},
        ...
      }
    """
    edges: Dict[str, Any] = {}

    for block in EDGE_BLOCK_RE.findall(text):
        # Example block:
        # ( : Forum ) - [ HAS_MEMBER { joinDate String } ] -> ( : Person )
        src_m = EDGE_SRC_RE.search(block)
        rel_m = EDGE_REL_RE.search(block)
        dst_m = EDGE_DST_RE.search(block)

        if not (src_m and rel_m and dst_m):
            # Uncomment for debugging:
            # print("Unparsed EDGE block:\n", block)
            continue

        src_labels_raw = src_m.group(1).strip()
        rel_name = rel_m.group(1).strip()
        rel_props_raw = rel_m.group(2)  # may be None
        dst_labels_raw = dst_m.group(1).strip()

        edges[rel_name] = {
            "from": _split_union_labels(src_labels_raw),
            "to": _split_union_labels(dst_labels_raw),
            "properties": _parse_edge_properties(rel_props_raw) if rel_props_raw else [],
        }

    return edges


def parse_graph_type(text: str) -> Dict[str, Any]:
    """
    Optional: parse CREATE GRAPH TYPE ... { ... };
    """
    m = GRAPH_TYPE_RE.search(text)
    if not m:
        return {}

    gname, gmode, inner = m.groups()
    items = [x.strip().strip(",") for x in inner.splitlines() if x.strip()]
    # remove trailing commas and empty
    items = [x for x in items if x and x != "}"]
    return {"name": gname, "mode": gmode, "items": items}


def parse_pgs_file(pgs_path: str) -> Dict[str, Any]:
    text = Path(pgs_path).read_text(encoding="utf-8")

    schema = {
        "nodes": parse_nodes(text),
        "edges": parse_edges(text),
    }

    g = parse_graph_type(text)
    if g:
        schema["graph"] = g

    return schema


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Parse .pgs schema into schema.json")
    ap.add_argument("pgs", help="Input .pgs schema file")
    ap.add_argument("-o", "--out", default="schema.json", help="Output JSON file")
    args = ap.parse_args()

    schema = parse_pgs_file(args.pgs)
    Path(args.out).write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Schema written to {args.out}")