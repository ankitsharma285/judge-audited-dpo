# Bias-Audited SFT + DPO for Instruction Following

An end-to-end post-training study on **Qwen2.5-1.5B**: SFT and DPO implemented from scratch (custom losses, no library trainers), preference labels produced by an LLM judge that was **validated against human annotations before being trusted**, and downstream results evaluated with confidence intervals and a length-stratified check rather than a single win-rate number.

> **Status:** active research project. Phases 0 to 5 are complete: learning-rate sweep, three training seeds at the best LR for both label types, a confirmation run on fresh prompts, and direct head-to-head tests of the two labeling methods. Remaining: a second, independent eval judge. See [Status and next steps](#status-and-next-steps).

---

## TL;DR

- Implemented **SFT** (response-only loss masking) and **DPO** (summed response log-probs, frozen reference model) from scratch in PyTorch.
- Built a **bias-aware LLM-judge labeling pipeline** (position randomization, explicit length control, CoT rubric) and validated it against the human-annotated AlpacaFarm cross-annotations with a pre-set threshold, a diagnosed failure mode, and a stopping rule.
- Three **distinct model families** fill the three roles (policy / labeling judge / eval judge) to avoid self-preference bias.
- **First result was null:** at LR = 1e-6, none of four DPO configurations beat SFT in-distribution (all win-rates 48 to 50%, CIs include 50%). A controlled LR sweep (LR the only changed variable) found the cause: 1e-6 and 2e-6 are indistinguishable from SFT, and **5e-6 beats SFT**.
- **The 5e-6 gain reproduces.** Across three training seeds the mitigated-label policy scores 58.2 to 59.4% in-distribution (single-order, 95% CIs all exclude 50%), and wins about 2 to 3 decisive pairs for every loss under both-order judging. The naive-label policy shows the same pattern more weakly in-distribution (53.4 to 56.8%).
- **It holds on prompts the sweep never saw.** On 453 fresh AlpacaEval prompts (shift-type), all six 5e-6 policies beat SFT: single-order 56.9 to 59.8% (CIs exclude 50%), and 72 to 78% of decisive both-order pairs. This was a pre-specified test; the primary policy passed with p = 3e-12. The earlier 68.0% shift-slice figure was optimistic: the same policy scores 57.1% on the fresh prompts.
- **Label debiasing did not improve quality.** Head-to-head, mitigated vs. naive policies (three seed pairs, each pair judged in both orders): no difference in-distribution (56.0%, 46.1%, 53.6% for mitigated among decisive pairs), and the shift slice leans slightly toward naive (40.0 to 43.2%). On the fresh prompts the naive policy is about 2 points ahead against SFT (59.4% vs. 57.6% mean). Debiasing did change behaviour: the naive policy writes longer answers (longer than SFT's in about 90% of fresh-prompt pairs vs. about 78% for mitigated) and its chosen-response likelihood falls further during training.
- **The eval judge has a strong position bias** (it picks the first-shown response 75 to 85% of the time), which only became visible when scoring both orders. Single-order win-rates are therefore noisy and, if anything, conservative.

---

## Key results

### Held-out win-rate vs. the SFT baseline

Judge: Gemma-2-9B-it (CoT rubric, position-randomized, one judgment per prompt). 95% Wilson intervals. These single-order numbers are noisy because of the judge's position bias; the position-controlled result for the 5e-6 policies is in [Both-order re-scoring](#both-order-re-scoring-of-the-5e-6-policies-vs-sft). "In-dist" = 252 AlpacaEval `selfinstruct` prompts; "shift" = 100 prompts stratified across `oasst`, `koala`, `helpful_base`, `vicuna`.

| DPO config | LR | In-dist win-rate | 95% CI | Shift win-rate | 95% CI |
|---|---|---|---|---|---|
| mitigated labels, β=0.05 | 1e-6 | 47.8% | [41.7, 54.0] | 51.0% | [41.3, 60.6] |
| mitigated labels, β=0.1 | 1e-6 | 48.0% | [41.8, 54.2] | 56.0% | [46.2, 65.3] |
| mitigated labels, β=0.3 | 1e-6 | 49.6% | [43.4, 55.8] | 55.0% | [45.2, 64.4] |
| naive labels, β=0.1 | 1e-6 | 49.6% | [43.4, 55.8] | 61.0% | [51.2, 70.0] |
| mitigated labels, β=0.1 | 2e-6 | 47.8% | [41.7, 54.0] | 55.0% | [45.2, 64.4] |
| naive labels, β=0.1 | 2e-6 | 50.0% | [43.8, 56.2] | 54.0% | [44.3, 63.4] |
| **mitigated labels, β=0.1** | **5e-6** | **59.4%** | **[53.2, 65.3]** | **68.0%** | **[58.3, 76.3]** |
| naive labels, β=0.1 | 5e-6 | 53.8% | [47.6, 59.8] | 65.0% | [55.3, 73.6] |

At LR = 1e-6 and 2e-6, only one of twelve comparisons (naive labels at 1e-6, shift slice) excluded 50%, and it only partially survived the length check below. At LR = 5e-6, the mitigated policy excludes 50% on both slices; the naive policy excludes it on the shift slice only. The two label types track each other across LR, and their 5e-6 intervals overlap heavily (see the head-to-head below). One seed per cell.

### Three training seeds at 5e-6 (vs. SFT)

Seeds 42 (the original run), 1234, 5678. Single-order win-rate with 95% Wilson CI; both-order = wins : losses among pairs where the judge preferred the same response in both orders (decisive win-rate in parentheses).

| Labels | Seed | In-dist, single-order | In-dist, both-order | Shift, single-order | Shift, both-order |
|---|---|---|---|---|---|
| mitigated | 42 | 59.4% [53.2, 65.3] | 64 : 27 (70.3%) | 68.0% [58.3, 76.3] | 33 : 6 (84.6%) |
| mitigated | 1234 | 59.0% [52.8, 65.0] | 60 : 30 (66.7%) | 61.0% [51.2, 70.0] | 26 : 9 (74.3%) |
| mitigated | 5678 | 58.2% [52.0, 64.2] | 63 : 22 (74.1%) | 67.0% [57.3, 75.4] | 32 : 5 (86.5%) |
| naive | 42 | 53.8% [47.6, 59.8] | 55 : 45 (55.0%) | 65.0% [55.3, 73.6] | 31 : 9 (77.5%) |
| naive | 1234 | 53.4% [47.2, 59.5] | 57 : 43 (57.0%) | 55.6% [45.7, 65.0] | 28 : 14 (66.7%) |
| naive | 5678 | 56.8% [50.6, 62.8] | 64 : 44 (59.3%) | 67.0% [57.3, 75.4] | 38 : 8 (82.6%) |

Seed-to-seed spread is small in-distribution (mitigated 58.2 to 59.4%, naive 53.4 to 56.8%) and larger on the 100-prompt shift slice (mitigated 61.0 to 68.0%, naive 55.6 to 67.0%), as expected from 100 prompts. The lower-LR rows in the first table are single seeds.

### Confirmation on fresh prompts (vs. SFT, 5e-6)

The LR was chosen using the in-dist and shift slices, so results on them carry selection bias. The 453 AlpacaEval prompts in neither slice (all from `oasst`, `koala`, `helpful_base`, `vicuna`, so shift-type) were evaluated once, with the analysis fixed beforehand: the seed-42 mitigated policy, both-order, exact sign test, "confirmed" if the decisive win-rate exceeds 50% with p < 0.05.

| Labels | Seed | Single-order | Both-order W : L : T | Decisive win-rate | Longer than SFT |
|---|---|---|---|---|---|
| mitigated | 42 (primary) | 57.1% [52.5, 61.6] | 117 : 33 : 302 | 78.0% [70.7, 83.9], p = 3e-12 | 361/452 |
| mitigated | 1234 | 58.8% [54.2, 63.2] | 114 : 34 : 303 | 77.0% [69.6, 83.1] | 353/451 |
| mitigated | 5678 | 56.9% [52.3, 61.3] | 105 : 39 : 308 | 72.9% [65.1, 79.5] | 350/452 |
| naive | 42 | 58.8% [54.3, 63.3] | 131 : 39 : 282 | 77.1% [70.2, 82.7] | 407/452 |
| naive | 1234 | 59.8% [55.2, 64.2] | 128 : 42 : 283 | 75.3% [68.3, 81.2] | 411/453 |
| naive | 5678 | 59.5% [54.9, 63.9] | 129 : 50 : 273 | 72.1% [65.1, 78.1] | 403/452 |

- **Confirmed.** The primary test passed, and so did every other policy. This confirms the shift-type result; it does not confirm in-distribution, because no fresh in-distribution prompts were used.
- **Shift numbers were optimistic.** The mitigated policies score 61 to 68% on the 100-prompt shift slice but 56.9 to 58.8% on the fresh prompts (naive: 55.6 to 67.0% vs. 58.8 to 59.8%). Selection on the slice and a favorable 100-prompt sample are both plausible causes and cannot be separated here. The fresh-prompt figure is the better estimate.
- **Naive is slightly ahead on the fresh prompts** (mean single-order 59.4% vs. 57.6%), with overlapping intervals. It also writes longer answers (about 90% of pairs longer than SFT's vs. about 78%).

### Length-stratified check (LR = 5e-6, mitigated, β=0.1)

AlpacaEval's regression-based length-controlled metric was out of scope, so length is checked by stratification: win-rate split by whether the DPO response is longer than the SFT response for the same prompt.

| Slice | DPO response longer | DPO response not longer |
|---|---|---|
| In-dist | 58.0% (n=157), CI [50.2, 65.4] | 62.0% (n=92), CI [51.8, 71.2] |
| Shift | 66.2% (n=71), CI [54.6, 76.1] | 72.4% (n=29), CI [54.3, 85.3] |

The policy wins at least as often when its answer is **not** longer. That is the opposite of what a length-driven gain would look like. (The shift/not-longer cell is small, n=29.)

For comparison, the same check on the **naive-label** 5e-6 policy:

| Slice | DPO response longer | DPO response not longer |
|---|---|---|
| In-dist | 52.5% (n=198), CI [45.6, 59.4] | 58.5% (n=53), CI [45.1, 70.7] |
| Shift | 61.4% (n=83), CI [50.7, 71.2] | 82.4% (n=17), CI [59.0, 93.8] |

The two label types differ in how often the DPO answer is longer than the SFT answer: **63% in-dist and 71% shift for mitigated, vs. 79% and 83% for naive.** The in-dist difference (about 16 points) is large relative to its standard error; the shift difference (about 12 points) is closer to the noise. The naive shift/not-longer cell has only 17 pairs and should not be read.

### Training dynamics (reward accuracy on the training pairs)

| Run | LR | Reward acc, first half → second half | Loss, first half → second half |
|---|---|---|---|
| mitigated, β=0.1 | 1e-6 | 0.500 → 0.521 | 0.6918 → 0.6907 (flat) |
| mitigated, β=0.1 | 2e-6 | 0.562 → 0.698 | 0.6875 → 0.6773 |
| mitigated, β=0.1 | 5e-6 | 0.630 → 0.792 | 0.6728 → 0.6098 |
| naive, β=0.1 | 1e-6 | 0.625 → 0.745 | 0.6852 → 0.6742 |
| naive, β=0.1 | 2e-6 | 0.745 → 0.828 | 0.6644 → 0.6266 |
| naive, β=0.1 | 5e-6 | 0.760 → 0.844 | 0.5866 → 0.4686 |
| mitigated, β=0.05 | 1e-6 | 0.521 → 0.562 | 0.6927 → 0.6910 |
| mitigated, β=0.3 | 1e-6 | 0.521 → 0.599 | 0.6911 → 0.6828 |

Training reward accuracy is measured against noisy judge labels, so it is **not** a quality metric. It is used only as a diagnostic. Held-out evaluation is the evidence. (Values are means over the 24 logged windows in each half of training. At every LR, the naive-label run reaches higher training accuracy than the mitigated run, which is consistent with, but does not demonstrate, naive labels containing an easier shortcut such as "longer is better".)

### Both-order re-scoring of the 5e-6 policies vs. SFT

The 5e-6 policies were re-scored against SFT with every pair judged in both orders (`--both_orders`). A win requires winning in both orders, and a split verdict is a tie. Ties are 60 to 63% of pairs. The first-shown response was picked 77 to 80% of the time.

| Policy | Slice | Wins / losses / ties | Decisive win-rate (95% CI) | Sign-test p | Ties counted as 0.5 |
|---|---|---|---|---|---|
| mitigated | In-dist | 64 / 27 / 158 | 70.3% [60.3, 78.7] | 0.0001 | 57.4% |
| mitigated | Shift | 33 / 6 / 61 | 84.6% [70.3, 92.8] | <0.0001 | 63.5% |
| naive | In-dist | 55 / 45 / 151 | 55.0% [45.2, 64.4] | 0.37 | 52.0% |
| naive | Shift | 31 / 9 / 60 | 77.5% [62.5, 87.7] | 0.0007 | 61.0% |

- **The 5e-6 improvement over SFT survives position-bias control.** When the judge has a position-independent preference, it favors the mitigated policy over SFT by 64:27 in-dist and 33:6 under shift. The "ties as 0.5" column (57.4%, 63.5%) is close to the single-order numbers (59.4%, 68.0%), as expected, since ties pull it toward 50% by construction. The two columns answer different questions and should not be compared as if they were the same metric.
- **The decisive-only rate conditions on the judge agreeing with itself across orders,** which drops about 60% of the pairs. It describes pairs with a clear preference and does not describe the whole eval set.
- **Naive in-dist is the one cell that does not clear 50%** (55:45, p = 0.37). The mitigated-vs-naive gap in-dist (70.3% vs. 55.0% against SFT) looks large, but it compares two different, non-paired subsets of decisive pairs. The direct head-to-head below is the proper test, and it found no significant difference. Do not cite this indirect gap as evidence for mitigation.
- Length-split win-rates (ties as 0.5): mitigated wins 55.4% when longer vs. 60.9% when not longer in-dist, and 60.6% vs. 70.7% under shift. The policy still does at least as well when its answer is shorter.

### Head-to-head: mitigated vs. naive labels (LR = 5e-6, β = 0.1)

Comparing each policy to SFT separately cannot show whether the two labeling methods differ. So the policies trained on the same seed were scored directly against each other, each pair judged in **both orders**; a policy gets a win only if it wins in both, and a split verdict counts as a tie (the judge's preference followed position, not content). A is mitigated.

| Seed pair | In-dist: A : B | Mitigated rate, sign-test p | Shift: A : B | Mitigated rate, sign-test p |
|---|---|---|---|---|
| 42 | 47 : 37 | 56.0% [45.3, 66.1], 0.33 | 15 : 20 (65 ties) | 42.9% [28.0, 59.1], 0.50 |
| 1234 | 41 : 48 (162 ties) | 46.1% [36.1, 56.4], 0.53 | 16 : 21 (62 ties) | 43.2% [28.7, 59.1], 0.51 |
| 5678 | 52 : 45 (154 ties) | 53.6% [43.7, 63.2], 0.54 | 18 : 27 (55 ties) | 40.0% [27.0, 54.5], 0.23 |
| pooled | 140 : 130 | 51.9% [45.9, 57.7], 0.58 | 49 : 68 | 41.9% [33.3, 50.9], 0.10 |

- **No difference in-distribution.** The three seeds fall on both sides of 50% and the pooled rate is 51.9%.
- **Shift leans toward naive**, in all three pairs (40.0 to 43.2%), but no single pair is significant. Pooled p = 0.10 is optimistic because the seed pairs share prompts and are not independent. Treat it as a hint. The fresh-prompt results (naive about 2 points ahead against SFT) point the same way and are also within noise.
- **About 60 to 65% of pairs are ties**, so each comparison rests on 84 to 97 decisive in-dist pairs and 35 to 45 on shift. At an effect of about 56% vs. 50%, roughly 540 decisive pairs would be needed for 80% power, which this eval set cannot supply. The conclusion is "no quality advantage for mitigated labels was found", not "no effect".
- **An earlier indirect comparison was misleading.** Against SFT, mitigated looked about 4 points ahead of naive in-distribution (58.9% vs. 54.7% mean across seeds). Compared directly there is no difference. Comparing through a shared baseline is not a substitute for a paired comparison.
- **Length.** The mitigated policy's answer was longer than the naive policy's in only 15 to 22% of pairs. Win-rates split by length are not interpretable (the mitigated-longer cells have as few as 4 decisive pairs).
- **Training behaviour differs.** In the second epoch the implicit reward of the chosen response averages about +0.03 to +0.04 for mitigated runs and −0.07 to −0.14 for naive runs, while rejected falls further for naive (about −0.7 to −0.8 vs. −0.16 to −0.18). Naive training pushes the chosen response's likelihood down relative to the reference. These rewards are summed over response tokens, so they scale with length and the naive policy's answers are longer; treat this as a clue, not a mechanism.

---

## Pipeline

```mermaid
flowchart LR
    A[Alpaca-52K] --> B[SFT on 15K<br/>Qwen2.5-1.5B]
    A --> C[2K held-back prompts]
    B --> D[4 completions / prompt<br/>temps 0.7 to 1.15]
    C --> D
    D --> E1[Judge: bias-mitigated<br/>Llama-3.1-8B]
    D --> E2[Judge: naive<br/>Llama-3.1-8B]
    E1 --> F1[1,920 pairs]
    E2 --> F2[1,923 pairs]
    F1 --> G[DPO from scratch]
    F2 --> G
    G --> H[Held-out eval<br/>Gemma-2-9B judge<br/>AlpacaEval prompts]
```

**Three roles, three model families**, so no model judges its own family's outputs:

| Role | Model |
|---|---|
| Policy (SFT, DPO) | Qwen2.5-1.5B |
| Labeling judge (builds DPO pairs) | Llama-3.1-8B-Instruct |
| Eval judge (scores SFT vs. DPO) | Gemma-2-9B-it |

---

## Design decisions worth reading

**Judge validation before use.** The labeling judge was validated on the `reward` split of the AlpacaFarm human cross-annotations (Alpaca-sourced, matching the prompts it would label). The eval judge was validated on the `eval` split (AlpacaEval-sourced, matching what it would score). The pass/fail thresholds were set before running the pilot.

| Judge config (labeling judge, n=250) | Agreement with human majority |
|---|---|
| Zero-shot forced choice | 59.6%, but with a positional skew (agreement 66.9% when humans preferred output 2, 50.5% when they preferred output 1) |
| CoT + few-shot, v1 | 56.4%, skew removed, but over-weighted minor technical compliance |
| **CoT + few-shot, v2 (shipped)** | **57.2%**, skew removed, agreement even across both directions (56.8% / 57.6%) |
| Human-human ceiling on this split | 64.2% (leave-one-out) |

The shipped judge scored below the pre-set 60% threshold. It was shipped anyway as a documented exception: the zero-shot judge's higher number came from an exploitable positional bias, and the split's own human ceiling is only 64.2%. Iteration stopped after two rounds, because tuning further against the same fixed 250-pair set risks overfitting the rubric to it. Gemma-2-9B-it as eval judge scored 63.7% against a 67.2% human ceiling on its split and was used without further tuning.

**Pair construction.** Best and worst of four completions are found with a 4-comparison min-max tournament (pair up, compare winners, compare losers), cheaper than six pairwise comparisons. Pairs where chosen and rejected are textually identical are dropped.

**Controlled labeling ablation.** The "naive" and "mitigated" datasets are built from the *same* completions. The naive judge is a bare forced-choice prompt in fixed order with no length instruction. Only the labeling method differs.

**One variable at a time.** The LR sweep changed LR and nothing else. Dataset size and epoch count were deliberately held fixed so the result stays interpretable.

---

## Training details

**SFT.** Full fine-tuning, bf16, AdamW, LR 2e-5 (cosine, 4% warmup), effective batch 32, 2 epochs, max length 1024. Prompt tokens are masked with `-100`; EOS is appended as a token id. 5% of the 15K subsample is held out for eval loss, and the best-eval-loss checkpoint is kept (epoch 1: 1.232 vs. epoch 2: 1.233, i.e. no gain from the second epoch).

**DPO.** Custom PyTorch loop. Loss is `-log σ(β · ((logπ_θ(y_w) − logπ_θ(y_l)) − (logπ_ref(y_w) − logπ_ref(y_l))))` with un-normalized summed log-probs over response tokens only. Reference model is a frozen copy of the SFT checkpoint. AdamW, cosine schedule with 10% warmup, batch 8, 2 epochs, grad-clip 1.0. Reward margins and reward accuracy are logged every 10 steps to `training_metrics.json`.

**Evaluation.** Greedy decoding (max 256 new tokens). In the SFT-vs-DPO scoring, each pair is judged once per prompt with randomized A/B position. The DPO-vs-DPO head-to-head judges every pair in both orders (see above), and `phase5c_score_eval.py` supports the same with `--both_orders`. Intervals are Wilson score intervals; the head-to-head also reports an exact sign test.

---

## Limitations

Please read these before citing any number above.

- **Seeds exist only for the two 5e-6 configurations** (three each). The LR sweep rows at 1e-6 and 2e-6 and the beta ablation are single runs.
- **LR was selected on the eval prompts.** 5e-6 was picked using the same slices the first results were reported on. The fresh-prompt confirmation addresses this for shift-type prompts (453 prompts, all pass). It does **not** cover in-distribution prompts, which still carry the selection caveat. The in-dist gain reproduced across three seeds, which addresses seed variance but not selection. The trend across LR is also not smooth (for example, mitigated in-dist is 48.0%, 47.8%, then 59.4%), so it should not be described as a scaling curve.
- **No LR above 5e-6 was tried**, so the optimum may lie beyond the grid, and over-optimization behaviour (the chosen-likelihood drop seen in the naive runs) was not explored.
- **Eval-judge position bias.** The eval judge picks the first-shown response 75 to 85% of the time. Single-order scoring is noisy because of this. The 5e-6 policies were re-scored with both-order judging and the improvement held, but only 35 to 40% of pairs are decisive on the slices (about 33% on the fresh prompts), so those rates are conditional on judge agreement. The other LR and beta configs were not re-scored.
- **One eval judge.** The eval judge shares its CoT rubric with the labeling judge (different model family, same criteria), so the two are not fully independent in what they reward. A second, independently designed judge has not yet been run, and no human evaluation of the final policies was done. Agreement between two LLM judges would reduce this concern but not prove quality.
- **Judge reliability is below the human ceiling** (57.2% vs. 64.2% for the labeling judge), so a meaningful share of training labels is noisy.
- **Small data.** About 1.9K DPO pairs; 100 prompts in the shift slice (the length-stratified shift cells are as small as n=29); all eval prompts share one benchmark.
- **Length control is a stratified check, not a regression.** It is a transparent robustness test, not a replacement for AlpacaEval's length-controlled metric.
- **Spot-check is weak evidence.** A 25-item single-annotator spot-check (72% agreement, CI roughly 54 to 90%) used best-vs-worst pairs, which are easier than typical pairs. It is a sanity check, not validation.
- **The naive-vs-mitigated question is underpowered.** About 60 to 65% of head-to-head pairs are position-dependent ties. No advantage for mitigated labels was found; a small difference in either direction cannot be excluded. Pooled head-to-head p-values across seeds are optimistic because the seed pairs share prompts.
- **The length and likelihood differences between label types are descriptions**, not evidence about quality or mechanism.
- Results are specific to this model size, dataset, and judge setup.

---

## Status and next steps

- [x] Phase 0: data prep, split verification (no overlap between SFT, DPO prompts, and eval)
- [x] Phase 1: SFT
- [x] Phase 2: labeling-judge validation and spot-check
- [x] Phase 3: completion generation and dual (mitigated / naive) labeling
- [x] Phase 4: DPO training (4 configs at LR 1e-6, plus LR sweep on mitigated)
- [x] Phase 5: eval-judge validation and held-out evaluation
- [x] LR sweep on mitigated and naive labels (1e-6, 2e-6, 5e-6) with held-out evaluation
- [x] Both-order re-scoring of the 5e-6 policies vs. SFT (improvement holds; other configs not re-scored)
- [x] Head-to-head of mitigated vs. naive labels, three seed pairs: no quality advantage for mitigated labels
- [x] Three training seeds at 5e-6, both label types: gain reproduces
- [x] Confirmation on 453 fresh AlpacaEval prompts, analysis fixed in advance: passed (shift-type prompts only)
- [ ] Second, independent eval judge (different family, plain prompt), validated against human annotations first
- [ ] Optional: LR above 5e-6 (best result is at the edge of the grid), chosen on a dev split and labeled exploratory
- [ ] Optional: head-to-head on the fresh prompts, to test the hint that naive labels do slightly better under shift
- [ ] Optional: larger preference dataset, then epoch count (not resolvable with this eval)

---

## Repository layout

```
phase0_data_prep.py              # Alpaca split, AlpacaEval + cross-annotation download
phase0_closeout_checks.py        # eval-leakage check, gated-model access check
phase1_sft_train.py              # SFT with custom response-only loss masking
phase2_judge_pilot.py            # labeling-judge validation vs. human annotations
phase2_spotcheck_prepare.py      # blinded self-labeling sheet
phase2_spotcheck_score.py        # score the spot-check
phase3a_generate_completions.py  # 4 completions per prompt from the SFT model
phase3b_label_pairs.py           # mitigated + naive labeling via min-max tournament
phase4_dpo_train.py              # DPO from scratch (--lr, --beta, --pairs_path)
phase5a_eval_judge_pilot.py      # eval-judge validation
phase5b_prepare_eval_split.py    # in-dist / shift slices of AlpacaEval
phase5b_generate_eval_completions.py
phase5c_score_eval.py            # SFT-vs-DPO scoring, win-rate + length strata (--both_orders optional)
phase5d_prepare_confirm_set.py   # fresh AlpacaEval prompts not used by the LR sweep
score_h2h.py                     # DPO-vs-DPO head-to-head, both-order judging, sign test
slurm/                           # example Slurm job scripts (cluster-specific)
```

---

## Reproducing

```bash
# environment: PyTorch (CUDA), transformers, datasets, huggingface_hub, accelerate
huggingface-cli login            # Llama-3.1 and Gemma-2 are gated: accept licenses first

python phase0_data_prep.py
python phase0_closeout_checks.py
python phase1_sft_train.py
python phase2_judge_pilot.py
python phase3a_generate_completions.py
python phase3b_label_pairs.py

python phase4_dpo_train.py --pairs_path data/phase3_dpo_pairs_mitigated.json \
    --beta 0.1 --lr 5e-6 --output_dir checkpoints/dpo_mitigated_beta0.1_lr5e-6

python phase5b_prepare_eval_split.py
python phase5b_generate_eval_completions.py --checkpoint_path checkpoints/qwen2.5-1.5b-sft/final_sft_checkpoint --checkpoint_name sft
python phase5b_generate_eval_completions.py --checkpoint_path checkpoints/dpo_mitigated_beta0.1_lr5e-6 --checkpoint_name dpo_mitigated_beta0.1_lr5e-6
python phase5c_score_eval.py --dpo_checkpoint_name dpo_mitigated_beta0.1_lr5e-6 --slice indist
python phase5c_score_eval.py --dpo_checkpoint_name dpo_mitigated_beta0.1_lr5e-6 --slice genshift

# head-to-head (needs completions for both checkpoints; naive uses --pairs_path data/phase3_dpo_pairs_naive.json)
python score_h2h.py --a dpo_mitigated_beta0.1_lr5e-6 --b dpo_naive_beta0.1_lr5e-6
```

Developed on a single H100 (80 GB). Large files (`data/`, `checkpoints/`) are not committed.

### Version notes (transformers v5 era)

These cost real debugging time, so they are recorded here:

- `TrainingArguments`: `warmup_ratio` was removed; use `warmup_steps` with a float (it acts as a ratio).
- `from_pretrained`: `torch_dtype` is deprecated in favor of `dtype`.
- `apply_chat_template` now returns a `BatchEncoding` by default; pass `return_dict=True` and unpack with `**inputs` into `generate()`.
- `datasets` 4.0+ no longer supports script-based datasets; `tatsu-lab/alpaca_eval` must be fetched as raw JSON via `hf_hub_download`.
- Qwen2.5-1.5B ties embeddings, so a "missing keys: lm_head.weight" warning on checkpoint reload is cosmetic (tie verified directly).
- Batched generation needs `padding_side="left"`; the SFT/DPO label-aligned collators need `padding_side="right"`.

---

## References

- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023)
- Dubois et al., *AlpacaFarm* (2023) and the AlpacaEval project (human cross-annotations used for judge validation)
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023) (position and verbosity bias)

## License and data

Code: MIT License. Datasets and models (Alpaca, AlpacaEval, Qwen2.5, Llama-3.1, Gemma-2) are subject to their own licenses and terms; check them before redistributing any derived data or weights.


