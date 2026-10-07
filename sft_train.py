import os
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
SFT_DATA_PATH = "data/sft_train_data.json"
OUTPUT_DIR = "checkpoints/qwen2.5-1.5b-sft"
VAL_FRACTION = 0.05  # held-out slice carved from the 15K SFT set, for eval-loss-plateau tracking
SEED = 42


class PromptLossMaskingCollator:
    """
    Custom collator that formats sequence inputs and applies label masking (-100)
    to prompt tokens so cross-entropy loss is computed ONLY on response tokens.

    Assumes tokenizer.padding_side == "right": input_ids are right-padded by
    tokenizer.pad(), and labels are right-padded here to match position-for-
    position. If padding_side were "left" these two would misalign silently,
    so the caller MUST set tokenizer.padding_side = "right" before using this
    collator (enforced in main() below).
    """

    def __init__(self, tokenizer, max_length: int = 1024):
        assert tokenizer.padding_side == "right", (
            "PromptLossMaskingCollator requires padding_side='right' for "
            "input_ids/labels alignment -- got padding_side="
            f"{tokenizer.padding_side!r}"
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        formatted_prompts = [item["formatted_prompt"] for item in batch]
        responses = [item["output"] for item in batch]

        prompt_encodings = self.tokenizer(
            formatted_prompts,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        full_input_ids = []
        full_labels = []

        for i in range(len(batch)):
            p_ids = prompt_encodings["input_ids"][i]

            # Encode response text alone (no string-concatenated EOS -- avoids
            # relying on the tokenizer's special-token string-matching behavior
            # during BPE encoding), then append the EOS id directly.
            remaining_budget = max(self.max_length - len(p_ids) - 1, 0)  # -1 reserves room for EOS
            r_ids = self.tokenizer.encode(
                responses[i],
                truncation=True,
                max_length=remaining_budget,
                add_special_tokens=False,
            )
            r_ids = r_ids + [self.tokenizer.eos_token_id]

            input_ids = p_ids + r_ids
            labels = [-100] * len(p_ids) + r_ids

            full_input_ids.append(input_ids)
            full_labels.append(labels)

        batch_encoded = self.tokenizer.pad(
            {"input_ids": full_input_ids},
            padding=True,
            return_tensors="pt",
        )

        max_pad_len = batch_encoded["input_ids"].shape[1]
        padded_labels = []
        for labels in full_labels:
            pad_len = max_pad_len - len(labels)
            padded_labels.append(labels + [-100] * pad_len)  # right-padded, matches input_ids

        batch_encoded["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch_encoded


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- Starting Phase 1 SFT Training: {MODEL_NAME} ---")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # explicit -- required for label alignment in the collator above

    # No device_map: single-GPU full-FT, let Trainer place the model.
    # device_map="auto" + Trainer is discouraged by HF and can break under
    # distributed launchers (torchrun/accelerate launch).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,  # transformers v5+: torch_dtype renamed to dtype
        attn_implementation="sdpa",
    )

    full_dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")

    # Carve a held-out validation slice so eval loss can actually be tracked
    # for the plateau-based stopping criterion from the project plan.
    split_dataset = full_dataset.train_test_split(test_size=VAL_FRACTION, seed=SEED)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"Train examples: {len(train_dataset)} | Eval examples: {len(eval_dataset)}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # effective train batch size = 32
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=0.04,  # transformers v5+: warmup_ratio removed; warmup_steps now accepts a float as ratio
        num_train_epochs=2,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,  # custom collator needs raw formatted_prompt/output columns intact
        gradient_checkpointing=False,  # single 1.5B full-FT fits comfortably in 80GB H100 VRAM
    )

    collator = PromptLossMaskingCollator(tokenizer=tokenizer, max_length=1024)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()

    final_path = os.path.join(OUTPUT_DIR, "final_sft_checkpoint")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"--- Phase 1 Complete. Checkpoint saved to: {final_path} ---")


if __name__ == "__main__":
    main()
