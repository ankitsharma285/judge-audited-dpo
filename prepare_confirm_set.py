"""
Phase 5d -- build a CONFIRMATION prompt set the LR sweep never touched.

The 5e-6 learning rate was chosen using the in-dist (252 selfinstruct) and
shift (100) slices, so results on those slices carry selection bias. This takes
every AlpacaEval prompt that is in neither slice (805 - 252 - 100 = 453 expected;
all from oasst / koala / helpful_base / vicuna, i.e. shift-type prompts).

Evaluate pre-chosen configs on this set ONCE, with the analysis fixed in advance
(see sft-dpo-project-plan.md). Do not use it to pick another LR.
"""
import json

with open("data/alpaca_eval_set.json", encoding="utf-8") as f:
    allp = json.load(f)
with open("data/eval_indist.json", encoding="utf-8") as f:
    used = {r["instruction"] for r in json.load(f)}
with open("data/eval_genshift.json", encoding="utf-8") as f:
    used |= {r["instruction"] for r in json.load(f)}

confirm = [r for r in allp if r["instruction"] not in used]
by_cat = {}
for r in confirm:
    by_cat[r["dataset"]] = by_cat.get(r["dataset"], 0) + 1
print(f"Total AlpacaEval prompts: {len(allp)} | used by sweep slices: {len(used)} | confirmation set: {len(confirm)}")
print("By category:", by_cat)
assert "selfinstruct" not in by_cat, "unexpected: selfinstruct prompts should all be in the in-dist slice"

with open("data/eval_confirm.json", "w", encoding="utf-8") as f:
    json.dump(confirm, f, indent=2)
print("Saved data/eval_confirm.json")
