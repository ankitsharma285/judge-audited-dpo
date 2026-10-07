"""
Phase 5c -- Held-out pairwise scoring: SFT vs. each DPO checkpoint.

Length control: rather than a full AlpacaEval-style logistic regression
(out of scope), this reports win-rate STRATIFIED by whether the DPO
completion is longer or not-longer than the SFT completion for that same
instruction. If a DPO checkpoint wins mostly on the "DPO is longer" subset
and not on the "DPO is not longer" subset, that's direct evidence of a
length-driven advantage rather than a genuine quality one.

Run once per (DPO checkpoint, slice) combination -- 4 checkpoints x 2 slices
= 8 runs total.
"""

import argparse
import json
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_MODEL = "google/gemma-2-9b-it"
SEED = 99

JUDGE_PROMPT_TEMPLATE = """You are evaluating two AI responses to the same instruction. \
Think through the following steps briefly, then give a final verdict.

Step 1: Does each response correctly follow what the instruction actually asked \
for (format, structure, or specific requirements the instruction specifies)?
Step 2: Is each response factually accurate and complete?
Step 3: Weigh instruction-adherence together with overall quality and \
helpfulness. Do not let a MINOR technical compliance detail (e.g. being a few \
words over a soft length guideline) override a response that is clearly better \
overall -- but do not let length, elaborateness, or polish excuse a response \
that fails to actually do what was asked either. Judge the response as a whole.

Here are three worked examples of this evaluation process:

---
Instruction: Edit the following sentence to use a comparison structure.

A city has many buildings.

Response A: A city has more buildings than a small town.
Response B: A city is like a beehive full of buildings.

Reasoning: Step 1: The instruction specifically asks for a comparison structure \
(e.g., "more...than"). Response A uses a direct comparison, correctly following \
the instruction. Response B uses a simile/metaphor, which is not the comparison \
structure requested, even though it is more vivid. Step 2: Both are factually \
reasonable. Step 3: Response A correctly follows the specific instruction; \
Response B does not, despite sounding more elaborate. This is a real failure to \
do what was asked, not a minor technicality.
Final Answer: A

---
Instruction: What is the boiling point of water at sea level in Celsius?

Response A: Water boils at 100 degrees Celsius at sea level.
Response B: The boiling point of water depends on many complex atmospheric \
factors including pressure, altitude, humidity, and dissolved mineral content, \
making it a fascinating area of physical chemistry to explore in depth.

Reasoning: Step 1: The instruction asks for a specific factual value. Response A \
directly and correctly answers it. Response B is much longer but never actually \
states the answer. Step 2: Response A is factually correct and complete for what \
was asked. Step 3: Response A fully and correctly satisfies the instruction; \
Response B's length does not make up for failing to answer the question.
Final Answer: A

---
Instruction: Write a short story (100 words or fewer) about a stormy night.

Response A: The rain fell. The wind blew. It was a stormy night. The dog \
barked. Then it stopped raining. The end.
Response B: Lightning split the sky as Mara pressed her hands against the \
window, watching the old oak in the yard bend nearly double. The power had \
gone out an hour ago, and every gust made the house groan like it might give \
in. She counted the seconds between flash and thunder, the way her father \
taught her, and wondered if the storm would still be raging by morning.

Reasoning: Step 1: Response A is safely under the word limit. Response B is \
slightly over 100 words. Step 2: Both are on-topic. Step 3: Response B is a far \
more engaging, vivid short story with real narrative tension, while Response A \
is a flat, disconnected list of sentences that barely qualifies as a story. \
Being a few words over a soft length guideline is a minor technicality that \
does not outweigh this large difference in actual quality -- the instruction's \
core intent (write a good short story) is much better served by Response B.
Final Answer: B

---

Now evaluate this pair:

Instruction:
{instruction}

Response A:
{response_a}

Response B:
{response_b}

Give brief reasoning for Steps 1-3, then end your reply with exactly one line in \
this format (nothing after it):
Final Answer: A
or
Final Answer: B"""


def parse_verdict(text: str):
    match = re.search(r"final answer:\s*([ab])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    tokens = re.findall(r"\b([AB])\b", text.upper())
    return tokens[-1] if tokens else None


def load_and_align(sft_path, dpo_path):
    with open(sft_path, encoding="utf-8") as f:
        sft_data = json.load(f)
    with open(dpo_path, encoding="utf-8") as f:
        dpo_data = json.load(f)

    dpo_by_instr = {r["instruction"]: r for r in dpo_data}
    aligned = []
    for sft_row in sft_data:
        dpo_row = dpo_by_instr.get(sft_row["instruction"])
        if dpo_row is None:
            continue
        aligned.append(
            {
                "instruction": sft_row["instruction"],
                "sft_completion": sft_row["completion"],
                "dpo_completion": dpo_row["completion"],
            }
        )
    return aligned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo_checkpoint_name", required=True)
    parser.add_argument("--slice", required=True, choices=["indist", "genshift"])
    args = parser.parse_args()

    random.seed(SEED)

    sft_path = f"data/eval_completions_sft_{args.slice}.json"
    dpo_path = f"data/eval_completions_{args.dpo_checkpoint_name}_{args.slice}.json"
    pairs = load_and_align(sft_path, dpo_path)
    print(f"Aligned {len(pairs)} pairs for {args.dpo_checkpoint_name} on {args.slice}")

    print(f"Loading judge model: {JUDGE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, dtype=torch.bfloat16).to("cuda")
    model.eval()

    results = []
    unparseable = 0

    for i, pair in enumerate(pairs):
        swap = random.random() < 0.5
        response_a = pair["dpo_completion"] if swap else pair["sft_completion"]
        response_b = pair["sft_completion"] if swap else pair["dpo_completion"]

        prompt_text = JUDGE_PROMPT_TEMPLATE.format(
            instruction=pair["instruction"], response_a=response_a, response_b=response_b
        )
        messages = [{"role": "user", "content": prompt_text}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        verdict = parse_verdict(generated)
        if verdict is None:
            unparseable += 1
            continue

        if verdict == "A":
            dpo_won = swap
        else:
            dpo_won = not swap

        dpo_is_longer = len(pair["dpo_completion"]) > len(pair["sft_completion"])

        results.append({"dpo_won": dpo_won, "dpo_is_longer": dpo_is_longer})

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pairs)}")

    n_scored = len(results)
    raw_win_rate = sum(r["dpo_won"] for r in results) / n_scored if n_scored else 0.0

    longer_subset = [r for r in results if r["dpo_is_longer"]]
    not_longer_subset = [r for r in results if not r["dpo_is_longer"]]
    win_rate_when_longer = sum(r["dpo_won"] for r in longer_subset) / len(longer_subset) if longer_subset else None
    win_rate_when_not_longer = sum(r["dpo_won"] for r in not_longer_subset) / len(not_longer_subset) if not_longer_subset else None

    print()
    print(f"=== {args.dpo_checkpoint_name} vs SFT, {args.slice} slice ===")
    print(f"Scored: {n_scored} (unparseable: {unparseable})")
    print(f"Raw win-rate (DPO vs SFT): {raw_win_rate:.1%}")
    if win_rate_when_longer is not None:
        print(f"  -- when DPO completion IS longer ({len(longer_subset)} pairs): {win_rate_when_longer:.1%}")
    else:
        print("  -- when DPO completion IS longer: n/a (no such pairs)")
    if win_rate_when_not_longer is not None:
        print(f"  -- when DPO completion is NOT longer ({len(not_longer_subset)} pairs): {win_rate_when_not_longer:.1%}")
    else:
        print("  -- when DPO completion is NOT longer: n/a (no such pairs)")
    print()
    print("A large gap between these two conditional win-rates indicates a length-driven")
    print("advantage rather than a genuine quality improvement.")

    out_path = f"data/phase5c_scores_{args.dpo_checkpoint_name}_{args.slice}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.dpo_checkpoint_name,
                "slice": args.slice,
                "n_scored": n_scored,
                "unparseable": unparseable,
                "raw_win_rate": raw_win_rate,
                "win_rate_when_longer": win_rate_when_longer,
                "win_rate_when_not_longer": win_rate_when_not_longer,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
