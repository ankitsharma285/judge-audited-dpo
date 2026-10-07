"""
Phase 0 closeout checks:
1. Confirm AlpacaEval instructions don't overlap with SFT / DPO-prompt-pool splits.
2. Confirm gated model access (Llama-3.1-8B-Instruct, Gemma-2-9b-it) is actually
   granted, not just that the HF token itself is valid.

Run from the directory containing data/dpo_prompt_pool.json, data/sft_train_data.json,
data/alpaca_eval_set.json.
"""

import hashlib
import json


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace, so near-identical formatting
    differences don't hide a real overlap or hide a real non-overlap."""
    return " ".join(text.strip().lower().split())


def instruction_hash(instruction: str) -> str:
    return hashlib.sha256(normalize(instruction).encode("utf-8")).hexdigest()


def check_alpaca_eval_overlap():
    print("=== 1. AlpacaEval overlap check ===")

    with open("data/dpo_prompt_pool.json", encoding="utf-8") as f:
        dpo_pool = json.load(f)
    with open("data/sft_train_data.json", encoding="utf-8") as f:
        sft_data = json.load(f)
    with open("data/alpaca_eval_set.json", encoding="utf-8") as f:
        alpaca_eval = json.load(f)

    dpo_hashes = {instruction_hash(r["instruction"]) for r in dpo_pool}
    sft_hashes = {instruction_hash(r["instruction"]) for r in sft_data}
    eval_hashes = {instruction_hash(r["instruction"]) for r in alpaca_eval}

    overlap_dpo = dpo_hashes & eval_hashes
    overlap_sft = sft_hashes & eval_hashes

    print(f"DPO pool size: {len(dpo_pool)} | SFT size: {len(sft_data)} | AlpacaEval size: {len(alpaca_eval)}")
    print(f"Overlap (DPO pool vs AlpacaEval):  {len(overlap_dpo)}")
    print(f"Overlap (SFT data vs AlpacaEval):  {len(overlap_sft)}")

    if overlap_dpo or overlap_sft:
        print("WARNING: overlap detected -- inspect the matching instructions below.")
        eval_by_hash = {instruction_hash(r["instruction"]): r["instruction"] for r in alpaca_eval}
        for h in list(overlap_dpo | overlap_sft)[:5]:
            print(f"  - {eval_by_hash[h][:120]}")
    else:
        print("PASS: no overlap between AlpacaEval and SFT/DPO-pool instructions.")
    print()


def check_gated_model_access():
    print("=== 2. Gated model access check ===")
    from huggingface_hub import model_info
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    models = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-2-9b-it",
    ]

    for model_id in models:
        try:
            info = model_info(model_id)
            print(f"  OK   {model_id}  (gated={info.gated})")
        except GatedRepoError:
            print(f"  FAIL {model_id}  -- access request not yet approved / not accepted")
        except RepositoryNotFoundError:
            print(f"  FAIL {model_id}  -- repo not found or token lacks visibility")
        except Exception as e:
            print(f"  FAIL {model_id}  -- unexpected error: {e}")

    print()
    print("If any FAIL: visit the model page on huggingface.co while logged in")
    print("and accept the license there -- model_info() alone won't grant access.")
    print()
    print("Stronger confirmation (actually pulls config, not just metadata):")
    from transformers import AutoTokenizer
    print("  AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')")
    AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    print("  AutoTokenizer.from_pretrained('google/gemma-2-9b-it')")
    AutoTokenizer.from_pretrained("google/gemma-2-9b-it")


if __name__ == "__main__":
    check_alpaca_eval_overlap()
    check_gated_model_access()
