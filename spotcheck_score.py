"""
Phase 2 confirmatory spot-check -- Step 2: score.

Run this after filling in data/spotcheck_template.csv (your_answer column: A or B).
Compares your blind labels against the judge's actual verdicts.
"""

import csv
import json

FILLED_CSV_PATH = "data/spotcheck_template.csv"
KEY_PATH = "data/spotcheck_key.json"


def main():
    with open(KEY_PATH, encoding="utf-8") as f:
        key = json.load(f)

    with open(FILLED_CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total, agree, unanswered = 0, 0, 0
    for row in rows:
        item_id = row["item_id"]
        your_answer = row["your_answer"].strip().upper()
        if your_answer not in ("A", "B"):
            unanswered += 1
            continue
        total += 1
        if your_answer == key[item_id]:
            agree += 1

    print(f"Items answered: {total} (unanswered/invalid: {unanswered})")
    if total > 0:
        print(f"Agreement with CoT-v2 judge: {agree}/{total} = {agree / total:.1%}")
    print()
    print("Context for interpreting this number (from Phase 2 pilot, n=250, reward split):")
    print("  Reward-split human-human agreement (leave-one-out): 64.2%")
    print("  CoT-v2 judge vs. pilot's pooled human labels:        57.2%")
    print()
    print("This is a small confirmatory check (n={}), not a statistical".format(total))
    print("validation -- report it as such, alongside the Phase 2 pilot numbers above.")


if __name__ == "__main__":
    main()
