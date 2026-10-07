"""
Phase 3b -- Dual labeling pass (Option B ablation data).

For each prompt's 4 completions, finds best-vs-worst via a 4-comparison
min-max tournament (pair up 1v2 & 3v4, compare winners for best, compare
losers for worst -- provably correct under transitive preference, cheaper
than all 6 pairwise comparisons).

Runs this TWICE per prompt with two judge configs, using the SAME
completions both times (the controlled variable for the Option B ablation):

  - MITIGATED: CoT-v2 rubric (from Phase 2), position-randomized per
    comparison, explicit length-control instruction.
  - NAIVE: bare "which is better, A or B" forced choice, fixed order
    (no position randomization), no length-control instruction.

Outputs two separate DPO pair datasets: phase3_dpo_pairs_mitigated.json
and phase3_dpo_pairs_naive.json.
"""

import json
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

COMPLETIONS_PATH = "data/phase3_completions.json"
JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SEED = 42

# --- Mitigated config: CoT-v2 rubric, identical to the Phase 2 shipped config ---
MITIGATED_PROMPT_TEMPLATE = """You are evaluating two AI responses to the same instruction. \
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

# --- Naive config: bare forced choice, no debiasing ---
NAIVE_PROMPT_TEMPLATE = """Instruction:
{instruction}

Response A:
{response_a}

Response B:
{response_b}

Which response is better, A or B? Reply with exactly one letter: A or B."""


def parse_mitigated_verdict(text: str):
    match = re.search(r"final answer:\s*([ab])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    tokens = re.findall(r"\b([AB])\b", text.upper())
    return tokens[-1] if tokens else None


def parse_naive_verdict(text: str):
    text = text.strip().upper()
    for token in text.split():
        cleaned = token.strip(".,:;\"'()")
        if cleaned in ("A", "B"):
            return cleaned
    if text.startswith("A"):
        return "A"
    if text.startswith("B"):
        return "B"
    return None


def make_compare_fn(model, tokenizer, mitigated: bool):
    """Returns compare_fn(instruction, completions, idx_a, idx_b) -> preferred index (idx_a or idx_b)."""

    def compare_fn(instruction, completions, idx_a, idx_b):
        text_a, text_b = completions[idx_a], completions[idx_b]

        if mitigated:
            swap = random.random() < 0.5
            shown_a = text_b if swap else text_a
            shown_b = text_a if swap else text_b
            prompt_text = MITIGATED_PROMPT_TEMPLATE.format(
                instruction=instruction, response_a=shown_a, response_b=shown_b
            )
            max_new_tokens = 300
        else:
            swap = False  # naive: fixed order, no randomization
            prompt_text = NAIVE_PROMPT_TEMPLATE.format(
                instruction=instruction, response_a=text_a, response_b=text_b
            )
            max_new_tokens = 10

        messages = [{"role": "user", "content": prompt_text}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        verdict = parse_mitigated_verdict(generated) if mitigated else parse_naive_verdict(generated)
        if verdict is None:
            return None  # caller decides fallback

        if verdict == "A":
            return idx_b if swap else idx_a
        else:
            return idx_a if swap else idx_b

    return compare_fn


def find_best_worst(indices, compare_pair_fn):
    """4-comparison simultaneous min-max tournament over exactly 4 indices."""
    i0, i1, i2, i3 = indices

    winner1 = compare_pair_fn(i0, i1)
    if winner1 is None:
        return None, None
    loser1 = i1 if winner1 == i0 else i0

    winner2 = compare_pair_fn(i2, i3)
    if winner2 is None:
        return None, None
    loser2 = i3 if winner2 == i2 else i2

    best = compare_pair_fn(winner1, winner2)
    if best is None:
        return None, None

    winner_of_losers = compare_pair_fn(loser1, loser2)
    if winner_of_losers is None:
        return None, None
    worst = loser2 if winner_of_losers == loser1 else loser1

    return best, worst


def label_dataset(prompts, model, tokenizer, mitigated: bool, label_name: str):
    compare_fn = make_compare_fn(model, tokenizer, mitigated)
    pairs = []
    skipped = 0

    for i, p in enumerate(prompts):
        completions = p["completions"]
        assert len(completions) == 4, f"expected 4 completions, got {len(completions)}"

        pair_compare = lambda a, b: compare_fn(p["instruction"], completions, a, b)
        best_idx, worst_idx = find_best_worst([0, 1, 2, 3], pair_compare)

        if best_idx is None or worst_idx is None or best_idx == worst_idx:
            skipped += 1
            continue

        # index-level check above doesn't catch same-text-different-index
        # duplicates (found during QA: 47/2000 prompts have all 4 completions
        # textually identical) -- a pair with identical chosen/rejected text
        # contributes zero preference signal to DPO, so skip it explicitly.
        if completions[best_idx].strip() == completions[worst_idx].strip():
            skipped += 1
            continue

        pairs.append(
            {
                "prompt_id": p["prompt_id"],
                "instruction": p["instruction"],
                "input": p.get("input", ""),
                "chosen": completions[best_idx],
                "rejected": completions[worst_idx],
            }
        )

        if (i + 1) % 100 == 0:
            print(f"  [{label_name}] {i + 1}/{len(prompts)} prompts, {skipped} skipped so far")

    return pairs, skipped


def main():
    random.seed(SEED)

    with open(COMPLETIONS_PATH, encoding="utf-8") as f:
        prompts = json.load(f)

    print(f"Loading judge model: {JUDGE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, dtype=torch.bfloat16).to("cuda")
    model.eval()

    print("=== Pass A: MITIGATED (CoT-v2 + position-randomization + length-control) ===")
    mitigated_pairs, mitigated_skipped = label_dataset(
        prompts, model, tokenizer, mitigated=True, label_name="mitigated"
    )
    with open("data/phase3_dpo_pairs_mitigated.json", "w", encoding="utf-8") as f:
        json.dump(mitigated_pairs, f, indent=2)
    print(f"Saved {len(mitigated_pairs)} pairs ({mitigated_skipped} skipped) to data/phase3_dpo_pairs_mitigated.json")

    print()
    print("=== Pass B: NAIVE (bare forced-choice, fixed order) ===")
    naive_pairs, naive_skipped = label_dataset(
        prompts, model, tokenizer, mitigated=False, label_name="naive"
    )
    with open("data/phase3_dpo_pairs_naive.json", "w", encoding="utf-8") as f:
        json.dump(naive_pairs, f, indent=2)
    print(f"Saved {len(naive_pairs)} pairs ({naive_skipped} skipped) to data/phase3_dpo_pairs_naive.json")


if __name__ == "__main__":
    main()
