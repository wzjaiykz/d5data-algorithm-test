# Auto-Labeling Banking Support Intents with a Foundation Model

*Banking77 · DeepSeek-V4.1-Flash · 231 utterances · 77 intents · 12 November 2026*

**TL;DR.** A zero-shot foundation-model labeler reaches **80.9%** accuracy (macro-F1 **0.794**) on a class-balanced 231-item Banking77 sample for **$0.011** and **37 s** wall-clock. Seven label-level prompt hints lift it to **84.0%** (+3.0 pp; 13 fixed, 6 broken; McNemar p=0.17). The remaining errors cluster around near-duplicate taxonomy labels, not model capability.

## 1. Dataset and Task Description

**Dataset.** Banking77 (Casanueva et al., 2020) contains 10,003 training and 3,080 test utterances over 77 fine-grained online-banking intents; the test split has exactly 40 utterances per intent and averages 11 words. It is released under CC BY 4.0 and downloadable as two CSV files.

**Input and target.** The input is a customer-support utterance, e.g. “I still have not received my new card.” The target is exactly one intent label from the 77, e.g. `card_arrival`. This is single-label 77-way classification, representative of support-ticket routing.

**Sample.** We use 231 test utterances, 3 per intent (seed 42), so every intent is covered and the evaluation is not confounded by class imbalance. At n=231, the 95% confidence interval on accuracy is about ±5 pp. We verified that neither split contains exact-duplicate texts or duplicate texts with conflicting labels.

**Why this dataset.** It has a single gold label, a balanced test set, a business-relevant task, and a direct CSV download; no HuggingFace or GPU is required. **Limitations:** it is a public benchmark (possible pretraining contamination), its label names are unusually transparent, and the utterances are short and template-like. The measured accuracy is therefore an upper bound for production traffic.

## 2. Model and Method

**Model.** `deepseek-flash` (DeepSeek-V4.1-Flash-1017) through the OpenAI-compatible chat API. We disable thinking (`thinking: {"type": "disabled"}`), set temperature 0, request JSON mode, and cap output at 80 tokens. At off-peak prices ($0.15/M input cache-miss, $0.60/M output) the 231-item run costs about one cent.

**Prompt.** The system message contains the task, the 77 allowed labels, the JSON schema, and three rules (copy the label exactly; confidence is a probability; no explanations). The user message is only `Utterance:\n<text>`. The long identical system prefix is cache-friendly: 63% of zero-shot prompt tokens and 79% of hinted prompt tokens were cache hits. Output is `{"label": "...", "confidence": 0.0-1.0}`. We validate the label against the allowed set, normalize case/spacing, and retry malformed JSON once with a repair message (max 3 attempts, 60 s timeout, 8 concurrent requests).

**Not an agent.** There is no tool use, memory, or multi-step loop: one API call per utterance. The LLM-specific engineering is the prompt, JSON validation, and confidence analysis; the rest is ordinary concurrency and accounting.

**Hinted variant.** Seven label-level disambiguation rules were written after the zero-shot run, using the training split to define each label and the baseline confusion matrix to choose which pairs to clarify. They never mention a specific test utterance. Example: `get_physical_card` is about the card’s PIN, while `passcode_forgotten` is only for an app login passcode.

## 3. Runnable Code (Core Part)

`demo_code.py` reads JSONL, calls the API concurrently, validates each response, and writes predictions plus metrics:

```bash
export DEEPSEEK_API_KEY=...
python demo_code.py --input data/banking77_sample.jsonl \
  --output outputs/predictions_hinted.jsonl --prompt hinted --concurrency 8
```

The repository also includes `--provider mock` (no key, used by pytest/CI), `outputs/predictions_zero_shot.jsonl`, `outputs/predictions_hinted.jsonl`, `*.metrics.json`, and `*.errors.csv`.

| Utterance (truncated) | Gold | Predicted | Conf. | Correct |
|---|---|---|---|---|
| How can I get my card out of the damn ATM? | card_swallowed | card_swallowed | 0.95 | Y |
| How many disposable cards can I make in a single day? | disposable_card_limits | disposable_card_limits | 0.97 | Y |
| What is the minimum age? | age_limit | age_limit | 0.98 | Y |
| Why has my card payment been declined? | declined_card_payment | declined_card_payment | 0.97 | Y |
| What’s the deal with no cryptocurrency on your app? | beneficiary_not_allowed | fiat_currency_support | 0.60 | N |

## 4. Result Analysis

**Headline.** Zero-shot: **187/231 = 80.95%** (95% CI 75.4–85.5), macro-F1 **0.794**, 44 errors. Hinted: **194/231 = 83.98%** (78.7–88.2), macro-F1 **0.827**, 37 errors. 53 of 77 intents are perfect in the hinted run; 3 are 0/3 (`card_arrival`, `pending_transfer`, `topping_up_by_card`).

**Where it fails.** The top confusion pairs (hinted) are:

| Gold → predicted | Count |
|---|---|
| card_arrival → card_delivery_estimate | 3 |
| pending_transfer → transfer_timing | 3 |
| supported_cards_and_currencies → fiat_currency_support | 2 |
| card_payment_not_recognised → compromised_card | 2 |
| cash_withdrawal_not_recognised → compromised_card | 2 |

Four patterns recur across the errors. (i) **Near-duplicate labels**: `card_arrival` and `card_delivery_estimate` are both about a card that has not arrived; the benchmark’s own labels are inconsistent (“How long does a card delivery take?” is `card_arrival`, while “I am waiting for my card to arrive” is `card_delivery_estimate`). (ii) **Label-name traps**: `get_physical_card` actually means “help with the card PIN” (105/106 training examples mention the PIN). Zero-shot predicted `passcode_forgotten` or `change_pin` for all three PIN items; the hint fixed 3/3. (iii) **Method vs outcome**: `topping_up_by_card` is a top-up method, but the model picks `top_up_failed` when the utterance mentions a failed top-up. (iv) **Lexical attraction and label gaps**: “cryptocurrency” pulls the model to `fiat_currency_support`; the taxonomy has no crypto intent, so the gold `beneficiary_not_allowed` is itself a forced fit.

**Confidence.** The model is overconfident at the low end (ECE 0.060). The 0.9–1.0 bin contains 185/231 items (80%), averages 0.947 confidence, and is 90.3% accurate; the 0.5–0.6 bin is 37.5% accurate (n=8). Confidence still ranks errors usefully:

| Auto-label coverage | n | Accuracy | Errors caught |
|---|---|---|---|
| 100% | 231 | 83.98% | 37/37 |
| 90% | 208 | 88.46% | 24/37 (64.9%) |
| 80% | 185 | 90.27% | 18/37 (48.6%) |
| 70% | 162 | 91.36% | 14/37 (37.8%) |

At 80% coverage, a human reviews 46 items and the automatic portion is 90.3% accurate; half of all errors are in the reviewed tail. Length is not a reliable bias in the hinted run: accuracy is 85.9%, 81.4%, 87.0%, and 81.8% across word-count quartiles.

```{=typst}
#pagebreak()
```

**Three concrete errors (hinted run).**

1. **Lexical attraction + taxonomy gap.** “What’s the deal with no cryptocurrency on your app?” → `fiat_currency_support` (gold `beneficiary_not_allowed`, confidence 0.60). “Currency” in “cryptocurrency” is the strongest lexical cue; the label set has no crypto intent, so the gold label is a poor fit and the model has no valid escape hatch.
2. **Method vs outcome.** “I tried to use my card to top up but all of the money has disappeared.” → `top_up_failed` (gold `topping_up_by_card`, confidence 0.72). Both the method and the failure are present; the model selects the failure, while the gold labels the method. The taxonomy mixes two dimensions in one label set.
3. **Specific vs general.** “When will the transfer be completed?” → `transfer_timing` (gold `pending_transfer`, confidence 0.95). The model treats a specific in-flight transfer as a general speed question; the benchmark does not state which label wins when both apply. The high confidence makes this a good candidate for human review.

**Prompt ablation.** Seven hints fixed 13 items and broke 6, a net +7 (+3.03 pp); agreement between variants is 207/231 (89.6%); McNemar’s exact test gives p=0.167, so the gain is suggestive, not statistically significant at n=231. The biggest wins were `get_physical_card` 0/3→3/3, `get_disposable_virtual_card` 0/3→3/3, `transfer_into_account` 1/3→3/3, and `transfer_not_received_by_recipient` 1/3→3/3. The costs were real: a hint about the cost of ordering a physical card fixed 1 item but broke 2 (`getting_spare_card`, `card_about_to_expire`), and `pending_transfer` fell from 1/3 to 0/3 as the model collapsed it into `transfer_timing`. Hints are not free; they need a held-out dev set.

## 5. Human vs Model Labeling

| Dimension | Model (measured) | Human (assumed; not measured) |
|---|---|---|
| Speed | 1.16 s/item mean, 1.68 s p95; 231 items in 36.7 s at 8 concurrency | ~15 s/item → ~58 min for 231; ~95× slower |
| Cost | $0.010 for 231; ~$0.043 per 1,000 at off-peak | ¥30–60/h at 15 s/item → ¥29–58 for 231; ~400–800× |
| Consistency | 50/50 label agreement on a repeat run; 89.6% across prompts | Expected to vary with fatigue and guideline interpretation |
| Semantic understanding | Handles paraphrases and typos; 53/77 perfect intents; fails on label-name traps, near-duplicates, negation; never asks a question | Can ask clarifying questions and apply guidelines; slower |
| Bias | Label-name bias, broad-label attraction (`compromised_card` 8 vs 3 gold), class collapse (`pending_transfer` never predicted), overconfidence | Fatigue, order effects, cultural/linguistic bias; not measured |

The API cost excludes one-off prompt engineering and evaluation. The human speed/cost figures are illustrative assumptions, not measurements. The model is extremely stable under an identical prompt (50/50 labels; mean absolute confidence change 0.006), but prompt-sensitive: the hint set changed 10.4% of labels.

## 6. Improvement Ideas

1. **Better prompts (measured).** The seven hints gave +3.03 pp, but with 6 regressions and p=0.17. The next step is a proper dev/test split, label definitions instead of examples, and de-semanticized label IDs to separate language understanding from label-name matching.
2. **Confidence filtering / human-in-the-loop (measured).** At 80% coverage the auto-labeled portion reaches 90.3% and captures 48.6% of errors in the 20% sent to humans; at 90% coverage, 88.5% and 64.9%. Because confidence is overconfident at the low end, calibrate it on a dev set or combine it with log-probabilities or model disagreement.
3. **Multi-model voting / cascade.** Run flash and pro (or three samples at temperature 0.7) and use disagreement as an uncertainty signal; only escalate the bottom 20% to pro. Pro costs about 4.4× flash on input and 3.3× on output, so a cascade keeps the average cost close to flash while buying a second opinion on the hard tail.
4. **Fix the taxonomy (highest leverage).** The residual errors cluster at label boundaries: merge or define `card_arrival` vs `card_delivery_estimate`, `pending_transfer` vs `transfer_timing`, and `card_payment_not_recognised` vs `compromised_card`; rename `get_physical_card` to something like `card_pin_help`; add `crypto`, `other`, and `unknown` so the model can abstain. A second-annotator audit and a real production sample would test whether these boundaries matter outside the benchmark.

**Artifacts and reproducibility.** `demo_code.py` is self-contained and has a `--provider mock` mode for CI; the 231-row sample and both prediction files are committed, together with `*.metrics.json`, `*.errors.csv`, and `tests/`. Python 3.12 + `httpx`; `pytest -q` needs no API key. Rebuild this PDF with `python scripts/build_report.py` (Pandoc + Typst). The key is read from `DEEPSEEK_API_KEY` and is never written to the repository.

**Limitations.** 231 items (±5 pp), one domain, a public benchmark that may be in pretraining data, transparent label names, self-reported confidence, a single annotator for the error analysis, and hints tuned on the evaluation set. The code, sample, outputs, and tests are in the repository; `pytest -q` runs without an API key.
