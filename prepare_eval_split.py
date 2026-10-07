"""
Phase 5b step 1 -- prepare the held-out eval split.

In-distribution: all 252 `selfinstruct` AlpacaEval examples (Alpaca-52K is
itself self-instruct-generated, so this is the stylistic match).

Generalization-shift: 100 examples stratified proportionally across the
other four categories (oasst, koala, helpful_base, vicuna), rather than a
flat random draw from the pooled 553 -- ensures the shift set's diversity
is a designed property, not a coincidence of sampling.
"""

import json
import random

ALPACA_EVAL_PATH = "data/alpaca_eval_set.json"
SEED = 7
GENSHIFT_TOTAL = 100

IN_DIST_CATEGORY = "selfinstruct"
SHIFT_CATEGORIES = ["oasst", "koala", "helpful_base", "vicuna"]


def main():
    random.seed(SEED)

    with open(ALPACA_EVAL_PATH, encoding="utf-8") as f:
        data = json.load(f)

    in_dist = [r for r in data if r["dataset"] == IN_DIST_CATEGORY]
    print(f"In-distribution ({IN_DIST_CATEGORY}): {len(in_dist)} examples")

    shift_pool = {cat: [r for r in data if r["dataset"] == cat] for cat in SHIFT_CATEGORIES}
    shift_pool_total = sum(len(v) for v in shift_pool.values())

    genshift = []
    allocated = 0
    cats = list(shift_pool.keys())
    for i, cat in enumerate(cats):
        if i == len(cats) - 1:
            n = GENSHIFT_TOTAL - allocated  # last category takes the remainder, avoids rounding gaps
        else:
            n = round(GENSHIFT_TOTAL * len(shift_pool[cat]) / shift_pool_total)
        sample = random.sample(shift_pool[cat], min(n, len(shift_pool[cat])))
        genshift.extend(sample)
        allocated += len(sample)
        print(f"  {cat}: {len(sample)}/{len(shift_pool[cat])} sampled")

    print(f"Generalization-shift total: {len(genshift)}")

    with open("data/eval_indist.json", "w", encoding="utf-8") as f:
        json.dump(in_dist, f, indent=2)
    with open("data/eval_genshift.json", "w", encoding="utf-8") as f:
        json.dump(genshift, f, indent=2)

    print("Saved data/eval_indist.json and data/eval_genshift.json")


if __name__ == "__main__":
    main()
