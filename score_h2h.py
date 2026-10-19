"""
Head-to-head pairwise scoring: DPO checkpoint A vs DPO checkpoint B.

Primary use: mitigated-label policy (A) vs naive-label policy (B) at the same
LR/beta, on the held-out eval prompts. Reuses the exact judge (Gemma-2-9B) and
CoT-v2 rubric from score_eval.py (imported, not copied, so they cannot drift).

Differences from score_eval.py (vs-SFT scoring):
  * Every pair is judged in BOTH orders (A first, then B first). A pair counts as
    a win for a model only if it wins in both orders; a split verdict is a "tie"
    (the judge's preference flipped with position, i.e. no reliable preference).
    This removes position bias instead of relying on random swapping.
  * Reports decisive wins/losses/ties, win-rate among decisive pairs with a
    Wilson 95% CI, and an exact two-sided sign test (H0: P(A beats B)=0.5).
  * Length strata: win-rate of A among decisive pairs, split by whether A's
    completion is longer than B's. Also the share of pairs where A is longer.

Usage (cluster):
  python score_h2h.py --a dpo_mitigated_beta0.1_lr5e-6 --b dpo_naive_beta0.1_lr5e-6
"""

import argparse
import json
import math

SEED_NOTE = "greedy decoding; no sampling, so results are deterministic given the inputs"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def sign_test_p(k, n):
    """Exact two-sided binomial test, H0: p=0.5."""
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def align(a_path, b_path):
    with open(a_path, encoding="utf-8") as f:
        a = json.load(f)
    with open(b_path, encoding="utf-8") as f:
        b = {r["instruction"]: r for r in json.load(f)}
    out = []
    for r in a:
        o = b.get(r["instruction"])
        if o is not None:
            out.append({"instruction": r["instruction"], "a": r["completion"], "b": o["completion"]})
    return out


def summarize(rows, name_a, name_b, slice_name):
    """rows: list of {outcome: 'A'|'B'|'tie', a_longer: bool}"""
    n = len(rows)
    a_w = [r for r in rows if r["outcome"] == "A"]
    b_w = [r for r in rows if r["outcome"] == "B"]
    ties = n - len(a_w) - len(b_w)
    dec = len(a_w) + len(b_w)
    lo, hi = wilson(len(a_w), dec)
    p = sign_test_p(len(a_w), dec)

    def strat(flag):
        sub = [r for r in rows if r["a_longer"] == flag and r["outcome"] != "tie"]
        k = sum(r["outcome"] == "A" for r in sub)
        return {"n_decisive": len(sub), "a_win_rate": (k / len(sub)) if sub else None}

    summary = {
        "a": name_a, "b": name_b, "slice": slice_name,
        "n_pairs": n, "a_wins": len(a_w), "b_wins": len(b_w), "ties": ties,
        "a_win_rate_decisive": (len(a_w) / dec) if dec else None,
        "a_win_rate_ci95": [lo, hi],
        "sign_test_p": p,
        "frac_a_longer": sum(r["a_longer"] for r in rows) / n if n else None,
        "strata_a_longer": strat(True),
        "strata_a_not_longer": strat(False),
    }

    print(f"\n=== {name_a} (A) vs {name_b} (B), {slice_name} ===")
    print(f"Pairs: {n} | A wins both orders: {len(a_w)} | B wins both orders: {len(b_w)} | split (tie): {ties}")
    if dec:
        print(f"A win-rate among decisive pairs: {len(a_w)/dec:.1%}  95% CI [{lo:.1%}, {hi:.1%}]  (n={dec})")
        print(f"Exact sign test p = {p:.3f}  (two-sided, H0: A and B equally preferred)")
    print(f"A longer than B in {summary['frac_a_longer']:.1%} of pairs")
    for label, key in (("A longer", "strata_a_longer"), ("A not longer", "strata_a_not_longer")):
        s = summary[key]
        wr = f"{s['a_win_rate']:.1%}" if s["a_win_rate"] is not None else "n/a"
        print(f"  A win-rate when {label}: {wr} (decisive n={s['n_decisive']})")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="checkpoint name A (e.g. mitigated)")
    ap.add_argument("--b", required=True, help="checkpoint name B (e.g. naive)")
    ap.add_argument("--slices", nargs="+", default=["indist", "genshift"], choices=["indist", "genshift"])
    args = ap.parse_args()

    # heavy imports here so the stats helpers above are importable without torch
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from score_eval import JUDGE_MODEL, JUDGE_PROMPT_TEMPLATE, parse_verdict

    print(f"Loading judge model: {JUDGE_MODEL} ({SEED_NOTE})")
    tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, dtype=torch.bfloat16).to("cuda")
    model.eval()

    def judge(instruction, resp_first, resp_second):
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            instruction=instruction, response_a=resp_first, response_b=resp_second
        )
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_verdict(text)  # 'A' (first shown) / 'B' (second shown) / None

    for sl in args.slices:
        pairs = align(f"data/eval_completions_{args.a}_{sl}.json",
                      f"data/eval_completions_{args.b}_{sl}.json")
        print(f"\nAligned {len(pairs)} pairs on {sl}")
        rows, unparseable = [], 0
        for i, p in enumerate(pairs):
            v1 = judge(p["instruction"], p["a"], p["b"])   # A shown first
            v2 = judge(p["instruction"], p["b"], p["a"])   # B shown first
            if v1 is None or v2 is None:
                unparseable += 1
                continue
            a_wins_1 = (v1 == "A")      # order 1: first slot = A
            a_wins_2 = (v2 == "B")      # order 2: second slot = A
            outcome = "A" if (a_wins_1 and a_wins_2) else "B" if (not a_wins_1 and not a_wins_2) else "tie"
            rows.append({"outcome": outcome, "a_longer": len(p["a"]) > len(p["b"]),
                         "first_slot_pref": [v1, v2]})
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(pairs)}")
        print(f"Unparseable pairs dropped: {unparseable}")
        summary = summarize(rows, args.a, args.b, sl)
        summary["unparseable"] = unparseable
        # position-bias diagnostic: how often did the judge pick the FIRST slot?
        firsts = [r["first_slot_pref"] for r in rows]
        if firsts:
            first_rate = sum((x[0] == "A") + (x[1] == "A") for x in firsts) / (2 * len(firsts))
            summary["first_slot_pick_rate"] = first_rate
            print(f"Judge picked the first-shown response {first_rate:.1%} of the time (50% = no position bias)")
        out = f"data/h2h_{args.a}__vs__{args.b}_{sl}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=2)
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
