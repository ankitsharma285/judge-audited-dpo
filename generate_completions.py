"""
Phase 3a -- Completion generation.

Generates 4 completions per prompt from the SFT checkpoint, one per
temperature, on the ~2K DPO prompt pool. Output feeds Phase 3b's dual
labeling pass (bias-mitigated vs. naive judge config).

NOTE on padding_side: this script uses padding_side="left", the opposite of
Phase 1's training collator (which required "right" for label alignment).
Left-padding is required for correct batched generation with a decoder-only
model -- it's a separate process/tokenizer instance, so no conflict with
Phase 1, but worth not copy-pasting this into a training context by mistake.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SFT_CHECKPOINT = "checkpoints/qwen2.5-1.5b-sft/final_sft_checkpoint"
DPO_PROMPT_POOL_PATH = "data/dpo_prompt_pool.json"
OUTPUT_PATH = "data/phase3_completions.json"

TEMPERATURES = [0.7, 0.85, 1.0, 1.15]  # one completion per temperature -> 4 completions/prompt
MAX_NEW_TOKENS = 256
BATCH_SIZE = 16
TOP_P = 0.95


def main():
    tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched generation

    model = AutoModelForCausalLM.from_pretrained(SFT_CHECKPOINT, dtype=torch.bfloat16).to("cuda")
    model.eval()

    with open(DPO_PROMPT_POOL_PATH, encoding="utf-8") as f:
        prompts = json.load(f)

    results = [{**p, "completions": []} for p in prompts]

    for temp_idx, temp in enumerate(TEMPERATURES):
        print(f"--- Temperature {temp} ({temp_idx + 1}/{len(TEMPERATURES)}) ---")
        for start in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[start:start + BATCH_SIZE]
            texts = [p["formatted_prompt"] for p in batch]
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).to("cuda")

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=temp,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.eos_token_id,
                )

            prompt_len = inputs["input_ids"].shape[1]  # uniform across the batch due to left-padding
            for i, out in enumerate(output_ids):
                text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
                results[start + i]["completions"].append(text.strip())

            if (start // BATCH_SIZE) % 10 == 0:
                print(f"  {min(start + BATCH_SIZE, len(prompts))}/{len(prompts)} prompts")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} prompts x {len(TEMPERATURES)} completions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
