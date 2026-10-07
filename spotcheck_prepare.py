"""
Phase 2 confirmatory spot-check -- Step 1: prepare the blinded labeling sheet.

Samples N pairs from the SHIPPED judge config's output (phase3_dpo_pairs_mitigated.json,
i.e. the CoT-v2 judge), randomizes which side (A/B) shows the judge's "chosen"
response so you label blind, and writes:

  - data/spotcheck_sheet.txt    -- human-readable, for you to read and label
  - data/spotcheck_template.csv -- fill in the "your_answer" column (A or B)
  - data/spotcheck_key.json     -- hidden answer key; don't open until scoring

Run phase2_spotcheck_score.py after filling in the CSV.
"""

import csv
import json
import random

MITIGATED_PAIRS_PATH = "data/phase3_dpo_pairs_mitigated.json"
N_SAMPLES = 25
SEED = 123  # different seed from the labeling pipeline itself, for independence


def main():
    random.seed(SEED)

    with open(MITIGATED_PAIRS_PATH, encoding="utf-8") as f:
        pairs = json.load(f)

    sample = random.sample(pairs, min(N_SAMPLES, len(pairs)))

    sheet_lines = []
    csv_rows = []
    key = {}

    for i, pair in enumerate(sample, start=1):
        swap = random.random() < 0.5
        # judge's "chosen" is always pair["chosen"]; swap decides which
        # letter (A/B) displays it, so the sheet doesn't reveal the verdict
        shown_a = pair["rejected"] if swap else pair["chosen"]
        shown_b = pair["chosen"] if swap else pair["rejected"]
        judge_verdict = "B" if swap else "A"  # which letter the judge's chosen maps to

        key[str(i)] = judge_verdict

        sheet_lines.append(f"=== Item {i} ===")
        sheet_lines.append(f"Instruction:\n{pair['instruction']}")
        if pair.get("input"):
            sheet_lines.append(f"\nInput:\n{pair['input']}")
        sheet_lines.append(f"\nResponse A:\n{shown_a}")
        sheet_lines.append(f"\nResponse B:\n{shown_b}")
        sheet_lines.append("\nYour answer (A or B, which is better): ______\n")
        sheet_lines.append("-" * 60)
        sheet_lines.append("")

        csv_rows.append({"item_id": i, "your_answer": ""})

    with open("data/spotcheck_sheet.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sheet_lines))

    with open("data/spotcheck_template.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "your_answer"])
        writer.writeheader()
        writer.writerows(csv_rows)

    with open("data/spotcheck_key.json", "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2)

    print(f"Prepared {len(sample)} items.")
    print("Read data/spotcheck_sheet.txt and fill in data/spotcheck_template.csv")
    print("(your_answer column: A or B). Do NOT open data/spotcheck_key.json until")
    print("you're done labeling -- it contains the judge's verdicts.")


if __name__ == "__main__":
    main()
