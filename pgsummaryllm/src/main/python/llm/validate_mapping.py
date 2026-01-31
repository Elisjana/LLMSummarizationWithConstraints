import json
import os

def validate_mapping_vs_ggds(mapping_file, summaries_file):
    """
    Checks if the proposed Supernodes accidentally collapse the 
    logical direction of the original GGD constraints.
    """
    # 1. Load the Mapping
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    # Reverse the mapping for easy lookup (Original Label -> Supernode)
    lookup = {}
    for supernode, originals in mapping.items():
        for label in originals:
            lookup[label] = supernode

    # 2. Load the GGD Context from your previous summaries
    print(f"--- VALIDATING: {mapping_file} ---")
    violations = []
    
    with open(summaries_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            ggd_id = data['id']
            summary_text = data['output'].lower()

            # Identify which original labels are mentioned in this specific GGD
            # This is a heuristic check based on the text summaries
            labels_found = [label for label in lookup.keys() if label.lower() in summary_text]

            # Check for collisions: Do two distinct labels in one GGD 
            # now map to the same Supernode?
            seen_supernodes = {}
            for label in labels_found:
                s_node = lookup[label]
                if s_node in seen_supernodes and seen_supernodes[s_node] != label:
                    violations.append({
                        "ggd": ggd_id,
                        "supernode": s_node,
                        "labels": [seen_supernodes[s_node], label]
                    })
                seen_supernodes[s_node] = label

    # 3. Output Results and Save Report
    report_path = "summaryOutput/validation_report.json"
    report = {
        "status": "PASS" if not violations else "WARNING",
        "total_violations": len(violations),
        "details": violations
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    if not violations:
        print("✅ VALIDATION PASSED: All GGD semantic boundaries preserved.")
    else:
        print(f"⚠️ VALIDATION WARNING: Found {len(violations)} potential logical collapses.")
        for v in violations:
            print(f"  - {v['ggd']}: '{v['labels'][0]}' and '{v['labels'][1]}' merged into '{v['supernode']}'")
    
    return report

# --- EXECUTION ---
INPUT_SUMMARIES = "summaryOutput/gemma_outputs.jsonl"
INPUT_MAPPING = "summaryOutput/summarized_schema_mapping.json"

if os.path.exists(INPUT_MAPPING):
    validate_mapping_vs_ggds(INPUT_MAPPING, INPUT_SUMMARIES)
else:
    print("Error: Mapping file not found. Run the mapping script first.")