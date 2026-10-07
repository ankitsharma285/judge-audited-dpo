"""
Phase 2 -- Labeling-Judge Pilot Validation

Validates Llama-3.1-8B-Instruct (the labeling judge for Phase 3) against the
`reward`-split human cross-annotations (327 pairs, Alpaca-sourced) -- the
subset that matches the prompt distribution the labeling judge will actually
see, per the Phase 0 investigation into the cross-annotations file structure.

Mitigations applied (must match the config used in Phase 3, not a simplified
version):
  - Position randomization: which output is shown as "A" vs "B" is randomized
    per pair, so the judge can't learn a positional shortcut.
  - Length-control instruction: the judge prompt explicitly instructs the
    model to ignore response length and judge quality/correctness only.

Decide the proceed/fix/switch threshold BEFORE running this (see THRESHOLDS
below) -- don't set it after seeing the agreement number.
"""

import json
import random
from collections import defaultdict, Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CROSS_ANNOTATIONS_PATH = "data/alpaca_farm_human_crossannotations.json"
JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SEED = 42

# Pre-set thresholds -- decided before seeing the agreement number.
THRESHOLDS = {
    "proceed": 0.70,   # >= this: proceed with the labeling judge as-is
    "fix_or_switch": 0.60,  # < this: try prompt-engineering, then API fallback
    # between the two: judgment call
}

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


def load_reward_split_pairs():
    with open(CROSS_ANNOTATIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    reward_rows = [r for r in data if r["datasplit"] == "reward"]

    groups = defaultdict(list)
    for row in reward_rows:
        key = (row["instruction"], row["output_1"], row["output_2"])
        groups[key].append(row["preference"])  # 1 or 2

    pairs = []
    for (instruction, output_1, output_2), prefs in groups.items():
        vote_counts = Counter(prefs)
        majority_pref, majority_count = vote_counts.most_common(1)[0]
        if majority_count == len(prefs) / 2:
            continue  # skip true ties (even split, no majority)
        pairs.append(
            {
                "instruction": instruction,
                "output_1": output_1,
                "output_2": output_2,
                "human_majority": majority_pref,  # 1 or 2
                "n_annotators": len(prefs),
            }
        )
    return pairs


def build_judge_prompt(instruction, output_1, output_2, swap: bool):
    # swap=True means output_2 is shown as "A" and output_1 as "B"
    response_a = output_2 if swap else output_1
    response_b = output_1 if swap else output_2
    return JUDGE_PROMPT_TEMPLATE.format(
        instruction=instruction, response_a=response_a, response_b=response_b
    )


def parse_verdict(generated_text: str):
    """Extract A/B from a CoT reply. Looks for 'Final Answer: A/B' first
    (the format the prompt asks for); falls back to scanning for a bare
    A/B token if the model didn't follow the format exactly."""
    import re

    match = re.search(r"final answer:\s*([ab])\b", generated_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # fallback: last standalone A/B token in the reply (CoT models sometimes
    # restate the letter without the exact "Final Answer:" phrasing)
    tokens = re.findall(r"\b([AB])\b", generated_text.upper())
    if tokens:
        return tokens[-1]
    return None


def verdict_to_output_pref(verdict: str, swap: bool):
    """Map the judge's A/B verdict back to output_1/output_2, undoing the position swap."""
    if verdict == "A":
        return 2 if swap else 1
    elif verdict == "B":
        return 1 if swap else 2
    return None


def main():
    random.seed(SEED)

    print("Loading reward-split cross-annotation pairs...")
    pairs = load_reward_split_pairs()
    print(f"Loaded {len(pairs)} pairs with a clear human majority vote.")

    print(f"Loading judge model: {JUDGE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL,
        dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()

    results = []
    unparseable = 0

    for i, pair in enumerate(pairs):
        swap = random.random() < 0.5  # position randomization

        prompt_text = build_judge_prompt(
            pair["instruction"], pair["output_1"], pair["output_2"], swap
        )
        messages = [{"role": "user", "content": prompt_text}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,   # transformers v5+: apply_chat_template returns a
            return_tensors="pt",  # BatchEncoding dict now, not a plain tensor
        ).to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,  # CoT needs room to reason before the final line
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        verdict = parse_verdict(generated)
        if verdict is None:
            unparseable += 1
            continue

        judge_pref = verdict_to_output_pref(verdict, swap)
        results.append(
            {
                "instruction": pair["instruction"],
                "len_output_1": len(pair["output_1"]),
                "len_output_2": len(pair["output_2"]),
                "swap": swap,
                "judge_pref": judge_pref,
                "human_majority": pair["human_majority"],
                "agree": judge_pref == pair["human_majority"],
                "raw_response": generated,
            }
        )

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(pairs)}")

    n_scored = len(results)
    n_agree = sum(r["agree"] for r in results)
    agreement = n_agree / n_scored if n_scored else 0.0

    print()
    print("=== Phase 2 Pilot Results ===")
    print(f"Pairs loaded:        {len(pairs)}")
    print(f"Unparseable verdicts: {unparseable}")
    print(f"Pairs scored:         {n_scored}")
    print(f"Judge-human agreement: {agreement:.1%}")
    print()
    print("Reference points (full cross-annotation set, not reward-split alone):")
    print("  Human-human agreement:     65.7%")
    print("  Best GPT-4-class judges:   ~69%")
    print("  Llama-3-70B judge:         67.5%")
    print()

    if agreement >= THRESHOLDS["proceed"]:
        print(f"Result >= {THRESHOLDS['proceed']:.0%}: PROCEED with labeling judge as configured.")
    elif agreement < THRESHOLDS["fix_or_switch"]:
        print(f"Result < {THRESHOLDS['fix_or_switch']:.0%}: FIX (prompt-engineering) or SWITCH judge.")
    else:
        print("Result in between thresholds: JUDGMENT CALL -- inspect disagreements before deciding.")

    with open("data/phase2_pilot_results_cot_v2.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "agreement": agreement,
                "n_scored": n_scored,
                "unparseable": unparseable,
                "results": results,
            },
            f,
            indent=2,
        )
    print("Saved raw results to data/phase2_pilot_results_cot_v2.json")


if __name__ == "__main__":
    main()
