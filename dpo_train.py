"""
Phase 4 -- DPO training, implemented from scratch (no trl.DPOTrainer).

Loss: standard DPO (Rafailov et al. 2023):
  L = -E[log sigmoid(beta * ((logp_pi_chosen - logp_pi_rejected)
                             - (logp_ref_chosen - logp_ref_rejected)))]

Response log-probs are summed (un-normalized) over response tokens only,
matching the original paper. Reference model is a frozen copy of the SFT
checkpoint.

Usage (run each of these four for the project's DPO phase):

  python dpo_train.py --pairs_path data/dpo_pairs_mitigated.json \
      --beta 0.1 --output_dir checkpoints/dpo_mitigated_beta0.1

  python dpo_train.py --pairs_path data/dpo_pairs_naive.json \
      --beta 0.1 --output_dir checkpoints/dpo_naive_beta0.1

  python dpo_train.py --pairs_path data/dpo_pairs_mitigated.json \
      --beta 0.05 --output_dir checkpoints/dpo_mitigated_beta0.05

  python dpo_train.py --pairs_path data/dpo_pairs_mitigated.json \
      --beta 0.3 --output_dir checkpoints/dpo_mitigated_beta0.3
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

SFT_CHECKPOINT = "checkpoints/qwen2.5-1.5b-sft/final_sft_checkpoint"


def format_alpaca_prompt(instruction: str, input_text: str = "") -> str:
    """Same template used in Phase 0/3 -- reconstructed here since the saved
    DPO pairs only stored instruction/input, not formatted_prompt."""
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


class DPOPairDataset(Dataset):
    def __init__(self, pairs_path):
        with open(pairs_path, encoding="utf-8") as f:
            self.pairs = json.load(f)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class DPOCollator:
    """Builds (prompt+chosen) and (prompt+rejected) sequences with prompt
    tokens masked to -100, same masking approach as Phase 1's SFT collator,
    so response-only log-probs can be summed correctly.
    Requires tokenizer.padding_side == 'right' for label/input_ids alignment.
    """

    def __init__(self, tokenizer, max_length=1024):
        assert tokenizer.padding_side == "right", "DPOCollator requires padding_side='right'"
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _build_side(self, instruction, input_text, response_text):
        formatted_prompt = format_alpaca_prompt(instruction, input_text)
        p_ids = self.tokenizer(
            formatted_prompt, truncation=True, max_length=self.max_length, add_special_tokens=False
        )["input_ids"]

        remaining_budget = max(self.max_length - len(p_ids) - 1, 0)
        r_ids = self.tokenizer.encode(
            response_text, truncation=True, max_length=remaining_budget, add_special_tokens=False
        )
        r_ids = r_ids + [self.tokenizer.eos_token_id]

        input_ids = p_ids + r_ids
        labels = [-100] * len(p_ids) + r_ids
        return input_ids, labels

    def _pad_side(self, all_input_ids, all_labels):
        batch_encoded = self.tokenizer.pad(
            {"input_ids": all_input_ids}, padding=True, return_tensors="pt"
        )
        max_len = batch_encoded["input_ids"].shape[1]
        padded_labels = [labels + [-100] * (max_len - len(labels)) for labels in all_labels]
        batch_encoded["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch_encoded

    def __call__(self, batch):
        chosen_ids, chosen_labels = [], []
        rejected_ids, rejected_labels = [], []

        for item in batch:
            c_ids, c_lab = self._build_side(item["instruction"], item.get("input", ""), item["chosen"])
            r_ids, r_lab = self._build_side(item["instruction"], item.get("input", ""), item["rejected"])
            chosen_ids.append(c_ids)
            chosen_labels.append(c_lab)
            rejected_ids.append(r_ids)
            rejected_labels.append(r_lab)

        chosen_batch = self._pad_side(chosen_ids, chosen_labels)
        rejected_batch = self._pad_side(rejected_ids, rejected_labels)

        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "chosen_labels": chosen_batch["labels"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
            "rejected_labels": rejected_batch["labels"],
        }


def get_batch_logps(logits, labels):
    """Sum of log-probs over response tokens (labels != -100), un-normalized,
    matching the original DPO paper's implementation choice."""
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100

    safe_labels = labels.clone()
    safe_labels[~loss_mask] = 0  # dummy index for masked positions, zeroed out below anyway

    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_logps = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
    return (token_logps * loss_mask).sum(dim=1)


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = (beta * (policy_chosen_logps - ref_chosen_logps)).detach()
    rejected_rewards = (beta * (policy_rejected_logps - ref_rejected_logps)).detach()

    return losses.mean(), chosen_rewards, rejected_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"--- DPO Training ---")
    print(f"pairs_path={args.pairs_path} beta={args.beta} lr={args.lr} "
          f"epochs={args.epochs} batch_size={args.batch_size} (x{args.grad_accum_steps} accum)")

    tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    policy = AutoModelForCausalLM.from_pretrained(SFT_CHECKPOINT, dtype=torch.bfloat16).to("cuda")
    policy.train()

    reference = AutoModelForCausalLM.from_pretrained(SFT_CHECKPOINT, dtype=torch.bfloat16).to("cuda")
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    dataset = DPOPairDataset(args.pairs_path)
    collator = DPOCollator(tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    total_steps = (len(loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"Dataset size: {len(dataset)} pairs | steps/epoch: {len(loader)} | total optim steps: {total_steps}")

    metrics_log = []
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}

            policy_chosen_logits = policy(
                input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]
            ).logits
            policy_chosen_logps = get_batch_logps(policy_chosen_logits, batch["chosen_labels"])

            policy_rejected_logits = policy(
                input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]
            ).logits
            policy_rejected_logps = get_batch_logps(policy_rejected_logits, batch["rejected_labels"])

            with torch.no_grad():
                ref_chosen_logits = reference(
                    input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]
                ).logits
                ref_chosen_logps = get_batch_logps(ref_chosen_logits, batch["chosen_labels"])

                ref_rejected_logits = reference(
                    input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]
                ).logits
                ref_rejected_logps = get_batch_logps(ref_rejected_logits, batch["rejected_labels"])

            loss, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, args.beta
            )

            (loss / args.grad_accum_steps).backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    margin = (chosen_rewards - rejected_rewards).mean().item()
                    reward_acc = (chosen_rewards > rejected_rewards).float().mean().item()
                    entry = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": loss.item(),
                        "chosen_reward": chosen_rewards.mean().item(),
                        "rejected_reward": rejected_rewards.mean().item(),
                        "margin": margin,
                        "reward_acc": reward_acc,
                        "lr": scheduler.get_last_lr()[0],
                    }
                    metrics_log.append(entry)
                    print(
                        f"step {global_step}/{total_steps} epoch {epoch} loss {loss.item():.4f} "
                        f"margin {margin:.4f} reward_acc {reward_acc:.3f} "
                        f"chosen_r {chosen_rewards.mean().item():.4f} rejected_r {rejected_rewards.mean().item():.4f}"
                    )

        print(f"--- Epoch {epoch} complete ---")

    policy.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=2)

    print(f"--- Complete. Checkpoint + metrics saved to: {args.output_dir} ---")


if __name__ == "__main__":
    main()
