import hashlib
import json
import os
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def get_prompt_hash(instruction: str, input_text: str) -> str:
    """Generates a deterministic SHA256 hash for instruction-input pairs."""
    content = f"{instruction.strip()}|||{input_text.strip()}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def format_alpaca_prompt(instruction: str, input_text: str = "") -> str:
    """Formats prompt according to standard Alpaca conditional template."""
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


def save_json_array(records, path: str):
    """Writes a list of dicts as a single pretty-printed JSON array.
    Used for ALL outputs in this script so every downstream file
    (json.load()) reads the same shape, regardless of whether the
    data originated from a local list or a HF `datasets.Dataset`.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def main():
    os.makedirs("data", exist_ok=True)
    print("--- Starting Phase 0 Data Pipeline ---")

    # 1. Pull Alpaca-52K dataset
    print("[1/4] Loading tatsu-lab/alpaca dataset...")
    full_dataset = load_dataset("tatsu-lab/alpaca", split="train")
    shuffled_ds = full_dataset.shuffle(seed=42)

    # 2. Split DPO Prompt Pool (~2K examples) & SFT Set (15K examples)
    dpo_pool_raw = shuffled_ds.select(range(2000))
    sft_pool_raw = shuffled_ds.select(range(2000, 17000))

    # Process DPO Prompts (Discard Alpaca outputs; construct formatted input)
    dpo_prompts = []
    dpo_hashes = set()
    for row in dpo_pool_raw:
        h = get_prompt_hash(row["instruction"], row["input"])
        dpo_hashes.add(h)
        dpo_prompts.append(
            {
                "prompt_id": h,
                "instruction": row["instruction"],
                "input": row["input"],
                "formatted_prompt": format_alpaca_prompt(
                    row["instruction"], row["input"]
                ),
            }
        )

    save_json_array(dpo_prompts, "data/dpo_prompt_pool.json")
    print(f"    Saved {len(dpo_prompts)} prompts to data/dpo_prompt_pool.json")

    # Process SFT Dataset
    sft_samples = []
    sft_hashes = set()
    for row in sft_pool_raw:
        h = get_prompt_hash(row["instruction"], row["input"])
        sft_hashes.add(h)
        sft_samples.append(
            {
                "prompt_id": h,
                "instruction": row["instruction"],
                "input": row["input"],
                "formatted_prompt": format_alpaca_prompt(
                    row["instruction"], row["input"]
                ),
                "output": row["output"],
            }
        )

    save_json_array(sft_samples, "data/sft_train_data.json")
    print(f"    Saved {len(sft_samples)} samples to data/sft_train_data.json")

    # 3. Confirm Zero Overlap
    overlap = dpo_hashes.intersection(sft_hashes)
    print(f"[2/4] Verifying split overlap... Hash Overlap Count: {len(overlap)}")
    assert len(overlap) == 0, "CRITICAL ERROR: Overlap detected between SFT and DPO pools!"
    print("    SUCCESS: Zero overlap between SFT split and DPO prompt pool.")

    # 4. Pull AlpacaEval Eval Set & Cross-Annotations
    #
    # NOTE: tatsu-lab/alpaca_eval is hosted via a legacy `alpaca_eval.py`
    # loading script. `datasets` >=4.0 removed support for loading-script
    # datasets entirely (RuntimeError: "Dataset scripts are no longer
    # supported"), and this isn't recoverable via trust_remote_code (also
    # removed in 4.0). The underlying data is just plain JSON sitting at
    # the repo root, so we bypass load_dataset() and pull the files
    # directly instead -- no script execution involved.
    print("[3/4] Pulling AlpacaEval eval set...")
    alpaca_eval_path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        repo_type="dataset",
        filename="alpaca_eval.json",
    )
    with open(alpaca_eval_path, encoding="utf-8") as f:
        alpaca_eval = json.load(f)
    save_json_array(alpaca_eval, "data/alpaca_eval_set.json")
    print(f"    Saved {len(alpaca_eval)} AlpacaEval prompts to data/alpaca_eval_set.json")

    print("[4/4] Pulling human cross-annotations file...")
    cross_annotations_path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        repo_type="dataset",
        filename="alpaca_farm_human_crossannotations.json",
    )
    with open(cross_annotations_path, encoding="utf-8") as f:
        cross_annotations = json.load(f)
    save_json_array(cross_annotations, "data/alpaca_farm_human_crossannotations.json")
    print(
        f"    Saved {len(cross_annotations)} cross-annotations to "
        f"data/alpaca_farm_human_crossannotations.json"
    )

    print("--- Phase 0 Complete ---")
    print("All four output files are plain JSON arrays (json.load() works uniformly on every file).")


if __name__ == "__main__":
    main()
