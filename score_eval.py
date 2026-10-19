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

--both_orders (optional, default off so earlier results stay reproducible):
judge every pair twice, once in each A/B order. A DPO "win" requires winning in
both orders; a split verdict is a tie and counts as 0.5 in the headline
win-rate. The eval judge picks the first-shown response ~75% of the time on
similar pairs, so single-order scoring is noisy; this removes that bias. Also
reports the decisive-only win-rate and the first-slot pick rate.
"""

import argparse
import json
import math
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


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


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
    parser.add_argument("--both_orders", action="store_true",
                        help="judge each pair in both A/B orders; split verdicts count as ties (0.5)")
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
    first_slot_picks = []   # True when the judge chose the first-shown response

    def judge(instruction, resp_first, resp_second):
        prompt_text = JUDGE_PROMPT_TEMPLATE.format(
            instruction=instruction, response_a=resp_first, response_b=resp_second
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
        return parse_verdict(generated)

    for i, pair in enumerate(pairs):
        dpo_is_longer = len(pair["dpo_completion"]) > len(pair["sft_completion"])

        if args.both_orders:
            v1 = judge(pair["instruction"], pair["dpo_completion"], pair["sft_completion"])  # DPO first
            v2 = judge(pair["instruction"], pair["sft_completion"], pair["dpo_completion"])  # SFT first
            if v1 is None or v2 is None:
                unparseable += 1
                continue
            dpo_wins_1 = (v1 == "A")
            dpo_wins_2 = (v2 == "B")
            first_slot_picks += [v1 == "A", v2 == "A"]
            if dpo_wins_1 and dpo_wins_2:
                score, outcome = 1.0, "win"
            elif (not dpo_wins_1) and (not dpo_wins_2):
                score, outcome = 0.0, "loss"
            else:
                score, outcome = 0.5, "tie"
        else:
            swap = random.random() < 0.5
            response_a = pair["dpo_completion"] if swap else pair["sft_completion"]
            response_b = pair["sft_completion"] if swap else pair["dpo_completion"]
            verdict = judge(pair["instruction"], response_a, response_b)
            if verdict is None:
                unparseable += 1
                continue
            first_slot_picks.append(verdict == "A")
            dpo_won_single = (swap if verdict == "A" else not swap)
            score, outcome = (1.0, "win") if dpo_won_single else (0.0, "loss")

        results.append({"dpo_won": score == 1.0, "score": score, "outcome": outcome,
                        "dpo_is_longer": dpo_is_longer})

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pairs)}")

    n_scored = len(results)
    # score is 1/0 in single-order mode (identical to the old dpo_won rate) and 1/0.5/0 with --both_orders
    raw_win_rate = sum(r["score"] for r in results) / n_scored if n_scored else 0.0

    longer_subset = [r for r in results if r["dpo_is_longer"]]
    not_longer_subset = [r for r in results if not r["dpo_is_longer"]]
    win_rate_when_longer = sum(r["score"] for r in longer_subset) / len(longer_subset) if longer_subset else None
    win_rate_when_not_longer = sum(r["score"] for r in not_longer_subset) / len(not_longer_subset) if not_longer_subset else None

    n_win = sum(r["outcome"] == "win" for r in results)
    n_loss = sum(r["outcome"] == "loss" for r in results)
    n_tie = sum(r["outcome"] == "tie" for r in results)
    n_dec = n_win + n_loss
    ci_lo, ci_hi = wilson(n_win, n_dec) if args.both_orders else wilson(n_win, n_scored)
    first_slot_rate = (sum(first_slot_picks) / len(first_slot_picks)) if first_slot_picks else None

    print()
    print(f"=== {args.dpo_checkpoint_name} vs SFT, {args.slice} slice ===")
    print(f"Scored: {n_scored} (unparseable: {unparseable})")
    mode = "both orders (ties = 0.5)" if args.both_orders else "single randomized order"
    print(f"Scoring mode: {mode}")
    print(f"Raw win-rate (DPO vs SFT): {raw_win_rate:.1%}")
    if args.both_orders:
        print(f"  wins {n_win} / losses {n_loss} / ties {n_tie}; decisive win-rate "
              f"{(n_win / n_dec if n_dec else float('nan')):.1%}, 95% Wilson CI [{ci_lo:.1%}, {ci_hi:.1%}] (n={n_dec})")
    else:
        print(f"  95% Wilson CI [{ci_lo:.1%}, {ci_hi:.1%}] (n={n_scored})")
    if first_slot_rate is not None:
        print(f"  judge picked the first-shown response {first_slot_rate:.1%} of the time (50% = no position bias)")
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

    suffix = "_bothorders" if args.both_orders else ""
    out_path = f"data/phase5c_scores_{args.dpo_checkpoint_name}_{args.slice}{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.dpo_checkpoint_name,
                "slice": args.slice,
                "n_scored": n_scored,
                "unparseable": unparseable,
                "both_orders": args.both_orders,
                "raw_win_rate": raw_win_rate,
                "n_win": n_win, "n_loss": n_loss, "n_tie": n_tie,
                "win_rate_ci95": [ci_lo, ci_hi],
                "first_slot_pick_rate": first_slot_rate,
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
