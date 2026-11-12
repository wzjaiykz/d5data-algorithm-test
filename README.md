# Auto-Labeling Banking Support Intents with a Foundation Model

A small, reproducible auto-labeling pipeline for **Banking77** (77 fine-grained
online-banking intents) using DeepSeek's OpenAI-compatible chat API. It is a
batch classifier, not an agent: one API call per utterance, no tools, no memory,
no multi-step planning.

The point of the exercise is **labeling quality**, not a leaderboard score: the
repository includes the real predictions, a confidence/coverage analysis, and a
2-page report with three concrete errors.

## Results at a glance

| Variant | Correct | Accuracy | Macro-F1 | Errors | Wall clock | API cost |
|---|---|---|---|---|---|---|
| Zero-shot | 187/231 | **80.95%** | 0.7935 | 44 | 37.3 s | $0.0106 |
| + 7 prompt hints | 194/231 | **83.98%** | 0.8275 | 37 | 36.7 s | $0.0100 |

231 test utterances, 3 per intent (seed 42), 8 concurrent requests, temperature
0, `deepseek-flash` in non-thinking mode. The hint ablation fixed 13 items and
broke 6 (McNemar exact p=0.167, so the +3.03 pp is suggestive, not significant
at n=231). On a 50-item repeat run the labels were identical (50/50); mean
absolute confidence change was 0.006.

Full analysis: [`report/report.pdf`](report/report.pdf) (2 pages) and
[`report/report.md`](report/report.md) (source).

## Dataset and task

* **Dataset:** [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data)
  (Casanueva et al., 2020), CC BY 4.0. 10,003 train / 3,080 test utterances,
  77 intents, exactly 40 test utterances per intent, average 11 words.
* **Input:** a customer-support utterance, e.g. "I still have not received my
  new card."
* **Target:** exactly one intent label from the 77, e.g. `card_arrival`.
* **Sample:** 231 test utterances, 3 per intent, fixed seed 42. Every intent is
  covered, so the evaluation is not confounded by class imbalance. Provenance
  and license are recorded in `data/banking77_sample.meta.json`.

The sample is committed, so the demo runs without downloading anything. To
rebuild it from the official CSV:

```bash
python scripts/prepare_data.py
```

The test split has no exact-duplicate texts and no duplicate texts with
conflicting labels; the training split is imbalanced (35-187 examples per
intent). Banking77 is a public benchmark: pretraining contamination is possible,
and its label names are unusually transparent, so the measured accuracy is an
upper bound for production traffic.

## Quick start

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export DEEPSEEK_API_KEY=sk-...     # Windows PowerShell: $env:DEEPSEEK_API_KEY="sk-..."
```

Run the zero-shot baseline and the hinted variant:

```bash
python demo_code.py --output outputs/predictions_zero_shot.jsonl
python demo_code.py --prompt hinted --output outputs/predictions_hinted.jsonl
```

No key? The mock provider is deterministic and is what CI uses:

```bash
python demo_code.py --provider mock --limit 5 --output /tmp/mock.jsonl
python -m pytest -q
```

If you are behind a proxy, `httpx` honors the standard `HTTPS_PROXY` /
`HTTP_PROXY` environment variables; the code does not hardcode a proxy.

### Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--input` | `data/banking77_sample.jsonl` | Input JSONL with `id`, `text`, `gold_label` |
| `--output` | `outputs/predictions_zero_shot.jsonl` | Predictions JSONL |
| `--metrics` | `<output stem>.metrics.json` | Metrics JSON path |
| `--model` | `deepseek-flash` | Any OpenAI-compatible model name |
| `--prompt` | `zero_shot` | `zero_shot` or `hinted` |
| `--provider` | `deepseek` | `deepseek` or `mock` |
| `--limit` | all rows | Pilot on the first N rows |
| `--concurrency` | `8` | Concurrent HTTP requests |
| `--max-attempts` | `3` | Retries for transient errors and invalid JSON |
| `--timeout` | `60` | Per-request timeout in seconds |

## Prompt and method

The system message contains the task, the 77 allowed labels, the JSON schema,
and three rules (copy the label exactly; confidence is a probability; no
explanations). The user message is only `Utterance:\n<text>`. The long identical
system prefix is cache-friendly: 63% of zero-shot prompt tokens and 79% of
hinted prompt tokens were cache hits.

```json
{"label": "<one allowed label>", "confidence": 0.0}
```

We validate the label against the allowed set, normalize case/spacing, and retry
malformed JSON once with a repair message. The model runs with
`thinking: {"type": "disabled"}`, temperature 0, JSON mode, and `max_tokens=80`.
At the observed ~18 output tokens per item, non-thinking mode uses roughly 20x
fewer output tokens and about 2x lower latency than reasoning mode for this task.

The hinted variant adds seven **label-level** disambiguation rules derived from
the training split plus the baseline confusion matrix, e.g.:

* `get_physical_card` is about the card's PIN; `passcode_forgotten` is only for
  an app login passcode.
* `get_disposable_virtual_card` is about getting/using a disposable card;
  `disposable_card_limits` is about limits and restrictions.
* `exchange_rate` is the rate itself; `exchange_via_app` is how/where to
  exchange; `fiat_currency_support` is which currencies the account supports.

The hints never mention a specific test utterance. They were evaluated on the
same 231 items used to diagnose the confusions, so the +3.03 pp is an
in-sample, optimistic estimate; a held-out dev/test split is the next step.

## Outputs and metrics

`demo_code.py` writes:

* `<output>.jsonl` — one record per input row: prediction, confidence, latency,
  token usage, raw model output, and the gold label.
* `<output stem>.metrics.json` — accuracy, macro-F1, Wilson 95% CI, per-class
  precision/recall/F1, top confusion pairs, calibration bins and ECE,
  accuracy-coverage curve, length bins, prediction bias, latency, token usage,
  and cost.
* `<output stem>.confusions.csv` and `<output stem>.errors.csv`.

Committed results:

```
outputs/predictions_zero_shot.jsonl
outputs/predictions_zero_shot.metrics.json
outputs/predictions_zero_shot.metrics.confusions.csv
outputs/predictions_zero_shot.metrics.errors.csv
outputs/predictions_hinted.jsonl
outputs/predictions_hinted.metrics.json
outputs/predictions_hinted.metrics.confusions.csv
outputs/predictions_hinted.metrics.errors.csv
outputs/predictions_zero_shot_repeat50.jsonl   # consistency check
outputs/predictions_zero_shot_repeat50.metrics.json
```

Cost is computed from the API's token accounting and the published DeepSeek
prices on 2026-11-12 (USD per 1M tokens: flash cache-hit $0.003, cache-miss
$0.15, output $0.60). Peak hours are 01:00-04:00 and 06:00-10:00 UTC on
weekdays; off-peak is half. The committed run was off-peak, so the applicable
cost is `cost.off_peak_usd`. Prices can change; the formula is in
`estimate_cost()`.

## What the errors look like

Top confusion pairs in the hinted run:

| Gold → predicted | Count |
|---|---|
| `card_arrival` → `card_delivery_estimate` | 3 |
| `pending_transfer` → `transfer_timing` | 3 |
| `supported_cards_and_currencies` → `fiat_currency_support` | 2 |
| `card_payment_not_recognised` → `compromised_card` | 2 |
| `cash_withdrawal_not_recognised` → `compromised_card` | 2 |

The residual errors cluster around near-duplicate taxonomy labels rather than a
lack of model capability. `card_arrival` and `card_delivery_estimate` are both
about a card that has not arrived; the benchmark's own labels are inconsistent
("How long does a card delivery take?" is `card_arrival`, while "I am waiting
for my card to arrive" is `card_delivery_estimate`). `get_physical_card` is a
label-name trap: 105 of its 106 training examples mention the card PIN, but the
name does not. The hinted prompt fixes all three PIN items in this sample.

At 80% auto-label coverage, the automatic portion is 90.3% accurate and the
20% sent to humans contains 48.6% of all errors; at 90% coverage, 88.5% and
64.9%. Confidence is overconfident at the low end (ECE 0.060), so calibration
or a second signal is needed before using a threshold in production.

## Repository layout

```
demo_code.py                 self-contained pipeline (real API + mock)
data/                        231-row sample, labels, provenance/license
outputs/                     committed predictions, metrics, error/confusion CSVs
report/report.md             2-page report source (English)
report/report.pdf            rendered 2-page report
report/preamble.typ          Typst layout for the PDF
scripts/prepare_data.py      rebuild the sample from the official CSV
scripts/build_report.py      Markdown -> Pandoc -> Typst -> PDF
tests/                       pytest unit + mock end-to-end tests
.github/workflows/ci.yml     pip + pytest + mock smoke test + artifact check
```

## Tests and CI

```bash
python -m pytest -q
```

The suite covers response parsing/validation, prompt construction, macro-F1,
confusion pairs, calibration, cost math, duplicate IDs, and a full mock
end-to-end run. CI installs `requirements.txt`, runs `pytest`, runs the mock
smoke test, and validates the committed 231-row result (194 correct, 83.98%).
No API key is needed in CI.

A live API smoke test is intentionally not part of the default workflow: it
would need a repository secret and would spend money on every push. To run one,
add `DEEPSEEK_API_KEY` as an Actions secret and run:

```bash
python demo_code.py --limit 3 --output /tmp/live.jsonl
```

locally with the key exported, or add a `workflow_dispatch` job that maps the
secret to `DEEPSEEK_API_KEY`.

## Limitations

* 231 items, 3 per intent: the 95% CI on accuracy is about ±5 pp, and per-class
  accuracy is coarse (0-3 correct per intent).
* One dataset and domain; public benchmark; possibly in pretraining data;
  transparent label names. Accuracy is an upper bound for production.
* Confidence is self-reported, not calibrated; ECE is 0.060 and low-confidence
  predictions are especially overconfident.
* The error analysis is by a single annotator; no inter-annotator agreement.
* The hints were tuned on the same 231 items used for evaluation; the +3.03 pp
  is in-sample and not statistically significant (McNemar p=0.167).
* There is no `unknown`/`other` class. The model always forces one of the 77
  labels, which is the wrong behavior for out-of-scope traffic.

The highest-leverage next step is not a stronger model: it is fixing the
taxonomy (merge/define the near-duplicate labels, rename `get_physical_card`,
add `crypto`/`other`/`unknown`) and collecting a real production sample with a
second-annotator audit.

## License and citation

The code in this repository is provided for the D5 Data algorithm exercise.
Banking77 is licensed under CC BY 4.0; if you use the dataset, cite:

```bibtex
@inproceedings{Casanueva2020,
  author = {I{\~{n}}igo Casanueva and Tadas Temcinas and Daniela Gerz and Matthew Henderson and Ivan Vulic},
  title = {Efficient Intent Detection with Dual Sentence Encoders},
  year = {2020},
  booktitle = {Proceedings of the 2nd Workshop on NLP for ConvAI - ACL 2020},
  url = {https://arxiv.org/abs/2003.04807}
}
```

The API key is read from `DEEPSEEK_API_KEY` and is never written to the
repository. `.env` is gitignored.
