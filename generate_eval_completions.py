"""
Phase 5b step 2 -- generate eval completions for one checkpoint.

Greedy decoding (do_sample=False), matching standard AlpacaEval-style
methodology -- eval scoring compares each model's single best-effort
response, not diverse samples like Phase 3's DPO-pair generation did.

Run once per checkpoint:

  python phase5b_generate_eval_completions.py \
      --checkpoint_path checkpoints/qwen2.5-1.5b-sft/final_sft_checkpoint \
      --checkpoint_name sft

  python phase5b_generate_eval_completions.py \
      --checkpoint_path checkpoints/dpo_mitigated_beta0.1 --checkpoint_name dpo_mitigated_beta0.1

  ...and similarly for dpo_naive_beta0.1, dpo_mitigated_beta0.05, dpo_mitigated_beta0.3
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_alpaca_prompt(instruction: str, input_text: str = "") -> str:
    if input_text and input_text.strip():
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction.strip()}\n\n"
            f"### Input:\n{input_text.strip()}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction.strip()}\n\n"
        "### Response:\n"
    )


def generate_for_file(model, tokenizer, input_path, output_path, batch_size, max_new_tokens):
    with open(input_path, encoding="utf-8") as f:
        prompts = json.load(f)

    results = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        texts = [format_alpaca_prompt(p["instruction"]) for p in batch]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for i, out in enumerate(output_ids):
            text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            results.append(
                {
                    "instruction": batch[i]["instruction"],
                    "dataset": batch[i]["dataset"],
                    "completion": text.strip(),
                }
            )

        if (start // batch_size) % 5 == 0:
            print(f"    {min(start + batch_size, len(prompts))}/{len(prompts)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {len(results)} completions to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--checkpoint_name", required=True, help="short name used in output filenames")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--slices", nargs="+", default=["indist", "genshift"],
                        choices=["indist", "genshift", "confirm"],
                        help="which eval slices to generate (default: indist genshift, as before)")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, dtype=torch.bfloat16).to("cuda")
    model.eval()

    slice_files = {
        "indist": "data/eval_indist.json",
        "genshift": "data/eval_genshift.json",
        "confirm": "data/eval_confirm.json",   # fresh AlpacaEval prompts never used in the sweep (phase5d)
    }
    for sl in args.slices:
        print(f"--- {sl} slice ---")
        generate_for_file(
            model, tokenizer, slice_files[sl],
            f"data/eval_completions_{args.checkpoint_name}_{sl}.json",
            args.batch_size, args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
