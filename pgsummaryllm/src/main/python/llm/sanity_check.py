#!/usr/bin/env python3
import json
from collections import Counter

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def get_node_label(n):
    for k in ("Node label", "nodeLabel", "label", "node_label", "NodeLabel"):
        v = n.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def labels_from_side(side):
    out=[]
    if isinstance(side, dict):
        for n in (side.get("nodes", []) or []):
            if isinstance(n, dict):
                lbl = get_node_label(n)
                if lbl:
                    out.append(lbl)
    return out

schema_path = "schema.json"
constraints_path = "JsonOutput/ldbc/v1/constraints.jsonl"

schema = json.load(open(schema_path, "r", encoding="utf-8"))
schema_labels = sorted({v["label"] for v in schema["nodes"].values()})
schema_set = set(schema_labels)

c = Counter()
for g in iter_jsonl(constraints_path):
    for lbl in labels_from_side(g.get("lhs", {})):
        c[lbl] += 1
    for lbl in labels_from_side(g.get("rhs", {})):
        c[lbl] += 1

constraint_set = set(c.keys())
unknown = sorted(constraint_set - schema_set)
missing = sorted(schema_set - constraint_set)

print("Schema labels:", schema_labels)
print("Constraint labels (top 30):", c.most_common(30))
print("Unknown labels in constraints (NOT in schema):", unknown)
print("Schema labels never used in constraints:", missing)