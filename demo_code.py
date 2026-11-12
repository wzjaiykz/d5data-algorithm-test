#!/usr/bin/env python3
"""Auto-label Banking77 customer-support utterances with a foundation model.

This is intentionally a *batch classifier*, not an agent: one API call per
utterance, no tools, no memory, no multi-step planning. The LLM-specific parts
are (1) the prompt, (2) JSON-mode output validation, and (3) confidence
calibration. Everything else is ordinary Python concurrency and accounting.

Usage (real API):
    export DEEPSEEK_API_KEY=sk-...
    python demo_code.py --input data/banking77_sample.jsonl \
        --output outputs/predictions_zero_shot.jsonl

Usage (no key, deterministic fake model for tests/CI):
    python demo_code.py --provider mock --limit 5 \
        --output /tmp/mock.jsonl

The script writes:
  * <output>.jsonl              one record per input row (prediction + usage)
  * <output stem>.metrics.json  accuracy, macro-F1, calibration, cost, etc.
  * <output stem>.confusions.csv
  * <output stem>.errors.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"

# Published list prices, USD per 1M tokens, fetched 2026-11-12 from
# https://api-docs.deepseek.com/quick_start/pricing
# Peak hours are 01:00-04:00 and 06:00-10:00 UTC on weekdays; off-peak is half.
PRICES_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "deepseek-flash": {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.60},
    "deepseek-v4-pro": {"cache_hit": 0.022, "cache_miss": 0.66, "output": 1.98},
}
PRICE_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing"
PRICE_AS_OF = "2026-11-12"

PROMPT_VARIANTS = ("zero_shot", "hinted")

# Filled in after the zero-shot run. Each hint is a label-level rule derived
# from the training split and the baseline confusion matrix (never from a
# specific test utterance). Keeping them label-level makes the hinted prompt
# reusable on new traffic.
DISAMBIGUATION_HINTS: tuple[str, ...] = (
    "`get_physical_card` is about the card's PIN: finding it, receiving it, or "
    "setting it up. `passcode_forgotten` is only for an app login password/passcode.",
    "`get_disposable_virtual_card` is about getting or using a disposable virtual "
    "card. `disposable_card_limits` is about limits, restrictions, or how many "
    "disposable cards can be created.",
    "`transfer_not_received_by_recipient` applies when the recipient has not "
    "received the money, even if the user asks 'how long'. `transfer_timing` is "
    "for general questions about transfer speed when no specific transfer is missing.",
    "`transfer_into_account` is the process of moving money into the account by "
    "bank transfer. `top_up_by_bank_transfer_charge` is specifically about a fee "
    "or charge for that top-up method.",
    "`cash_withdrawal_not_recognised` is for an unauthorized or unrecognized cash "
    "withdrawal. `compromised_card` is for broader card/account fraud such as "
    "unauthorized purchases.",
    "`exchange_rate` is only about the rate or value itself. Use `exchange_via_app` "
    "for how or where to exchange currency, `fiat_currency_support` for which "
    "currencies the account supports, and `card_payment_wrong_exchange_rate` / "
    "`wrong_exchange_rate_for_cash_withdrawal` for a disputed rate on a specific "
    "card payment or ATM withdrawal.",
    "Questions about the cost of ordering a physical card belong to "
    "`order_physical_card`, not `card_payment_fee_charged` (which is a fee charged "
    "on a card payment).",
)


class RetryableAPIError(Exception):
    """Transient API/network failure; safe to retry."""


class FatalAPIError(Exception):
    """Configuration/auth/payload failure; retrying will not help."""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_system_prompt(labels: list[str], variant: str) -> str:
    """Build the cacheable system prompt shared by every request in a run.

    Putting the 77-label list in the system message (rather than repeating it
    in every user message) gives the provider a long, identical prefix to cache.
    """
    parts = [
        "You are an intent classifier for online-banking customer-support messages.",
        "",
        "Choose exactly one intent from the allowed labels for the customer's utterance.",
        "",
        "Allowed labels:",
        "\n".join(labels),
    ]
    if variant == "hinted":
        if not DISAMBIGUATION_HINTS:
            raise SystemExit(
                "No hints configured. Run the zero-shot baseline first and fill "
                "DISAMBIGUATION_HINTS before using --prompt hinted."
            )
        parts += [
            "",
            "Disambiguation hints for frequently confused intents:",
            "\n".join(f"- {hint}" for hint in DISAMBIGUATION_HINTS),
        ]
    parts += [
        "",
        'Return one JSON object and nothing else: {"label": "<one allowed label>", '
        '"confidence": <number between 0 and 1>}',
        "",
        "Rules:",
        "- The label must be copied exactly from the allowed labels.",
        "- confidence is your estimated probability that the chosen label is correct.",
        "- Do not explain your answer or add extra keys.",
    ]
    return "\n".join(parts)


def build_repair_instruction(error: Exception) -> str:
    return (
        f"Your previous response was invalid: {error}. "
        'Return only a JSON object such as {"label": "card_arrival", "confidence": 0.8}. '
        "The label must be copied exactly from the allowed labels."
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def normalize_label(candidate: Any, labels: list[str]) -> str | None:
    """Map a model-produced label to an exact allowed label, or return None."""
    if not isinstance(candidate, str):
        return None
    cleaned = candidate.strip().strip("`\"'")
    if cleaned in labels:
        return cleaned
    # Tolerate case, spaces, and hyphen/underscore differences. Banking77 has
    # one label with a capital letter (Refund_not_showing_up), so build the
    # normalized map and return the exact original spelling.
    normalized = re.sub(r"[\s\-]+", "_", cleaned.lower())
    by_normalized = {
        re.sub(r"[\s\-]+", "_", label.lower()): label for label in labels
    }
    return by_normalized.get(normalized)


def parse_response(content: str, labels: list[str]) -> tuple[str, float | None]:
    """Parse and validate the model's JSON object."""
    if not content or not content.strip():
        raise ValueError("empty model response")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise ValueError("response is not a JSON object")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("response JSON is not an object")

    label = normalize_label(payload.get("label"), labels)
    if label is None:
        raise ValueError(f"label {payload.get('label')!r} is not in the allowed set")

    raw_confidence = payload.get("confidence")
    confidence: float | None
    if raw_confidence is None:
        confidence = None
    else:
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None:
            confidence = max(0.0, min(1.0, confidence))
    return label, confidence


# ---------------------------------------------------------------------------
# API / mock model calls
# ---------------------------------------------------------------------------


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    cache_hit = int(usage.get("prompt_cache_hit_tokens") or 0)
    # Preserve an explicit 0 (a fully cached prompt), but fall back to the total
    # when the provider does not report cache accounting at all.
    cache_miss = int(usage.get("prompt_cache_miss_tokens", prompt_tokens) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "cache_hit_tokens": cache_hit,
        "cache_miss_tokens": cache_miss,
    }


def post_chat(
    client: httpx.Client,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    try:
        response = client.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=timeout,
        )
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise RetryableAPIError(f"transport error: {exc}") from exc

    if response.status_code == 429:
        raise RetryableAPIError("HTTP 429 rate limited")
    if 500 <= response.status_code < 600:
        raise RetryableAPIError(f"HTTP {response.status_code} server error")
    if response.status_code >= 400:
        raise FatalAPIError(f"HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


def sleep_backoff(attempt: int) -> None:
    time.sleep(min(2 ** (attempt - 1), 8) + random.random() * 0.5)


def process_one(
    client: httpx.Client,
    item: dict[str, Any],
    system_prompt: str,
    labels: list[str],
    args: argparse.Namespace,
    url: str,
    api_key: str,
) -> dict[str, Any]:
    user_message = {"role": "user", "content": f"Utterance:\n{item['text']}"}
    messages = [
        {"role": "system", "content": system_prompt},
        user_message,
    ]
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
        # Classification does not need chain-of-thought. Disabling it cuts
        # output tokens by roughly 20x and latency by about 2x on these short
        # utterances (measured on one sample: 320 -> 16 output tokens,
        # 4.8 s -> 2.1 s).
        "thinking": {"type": "disabled"},
    }

    started = time.perf_counter()
    last_error: str | None = None
    raw_output = ""

    for attempt in range(1, args.max_attempts + 1):
        try:
            data = post_chat(client, url, api_key, payload, args.timeout)
            choices = data.get("choices") or [{}]
            message = choices[0].get("message") or {}
            raw_output = message.get("content") or ""
            label, confidence = parse_response(raw_output, labels)
            latency_ms = (time.perf_counter() - started) * 1000
            return make_record(
                item=item,
                predicted_label=label,
                confidence=confidence,
                correct=label == item["gold_label"],
                error=None,
                error_type=None,
                args=args,
                latency_ms=latency_ms,
                attempts=attempt,
                usage=normalize_usage(data.get("usage")),
                raw_output=raw_output,
            )
        except FatalAPIError as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            return make_error_record(
                item, args, latency_ms, attempt, f"fatal: {exc}", "fatal", raw_output
            )
        except (RetryableAPIError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= args.max_attempts:
                break
            if isinstance(exc, ValueError):
                # Give the model a chance to repair invalid JSON/labels. This
                # costs an extra call only for malformed outputs (rare).
                payload = dict(payload)
                payload["messages"] = [
                    {"role": "system", "content": system_prompt},
                    user_message,
                    {"role": "assistant", "content": raw_output or "(empty)"},
                    {"role": "user", "content": build_repair_instruction(exc)},
                ]
            sleep_backoff(attempt)

    latency_ms = (time.perf_counter() - started) * 1000
    return make_error_record(
        item,
        args,
        latency_ms,
        args.max_attempts,
        last_error or "unknown error",
        "parse_or_retry",
        raw_output,
    )


def mock_process_one(
    item: dict[str, Any], labels: list[str], args: argparse.Namespace
) -> dict[str, Any]:
    """Deterministic fake model used by pytest/CI; never used for results."""
    digest = int(hashlib.sha256(item["text"].encode("utf-8")).hexdigest()[:8], 16)
    if digest % 100 < 82:
        predicted = item["gold_label"]
    else:
        index = labels.index(item["gold_label"])
        predicted = labels[(index + 1 + digest % 5) % len(labels)]
    confidence = round(0.55 + (digest % 40) / 100, 2)
    return make_record(
        item=item,
        predicted_label=predicted,
        confidence=confidence,
        correct=predicted == item["gold_label"],
        error=None,
        error_type=None,
        args=args,
        latency_ms=0.0,
        attempts=1,
        usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
        },
        raw_output='{"label": "%s", "confidence": %.2f}' % (predicted, confidence),
    )


def make_record(
    *,
    item: dict[str, Any],
    predicted_label: str | None,
    confidence: float | None,
    correct: bool,
    error: str | None,
    error_type: str | None,
    args: argparse.Namespace,
    latency_ms: float,
    attempts: int,
    usage: dict[str, int],
    raw_output: str,
) -> dict[str, Any]:
    return {
        "id": item["id"],
        "order": item["order"],
        "text": item["text"],
        "gold_label": item["gold_label"],
        "predicted_label": predicted_label,
        "confidence": confidence,
        "correct": correct,
        "error": error,
        "error_type": error_type,
        "model": "mock" if args.provider == "mock" else args.model,
        "prompt_variant": args.prompt,
        "provider": args.provider,
        "latency_ms": round(latency_ms, 1),
        "attempts": attempts,
        "usage": usage,
        "raw_output": raw_output[:500],
        "word_count": len(item["text"].split()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def make_error_record(
    item: dict[str, Any],
    args: argparse.Namespace,
    latency_ms: float,
    attempts: int,
    error: str,
    error_type: str,
    raw_output: str,
) -> dict[str, Any]:
    return make_record(
        item=item,
        predicted_label=None,
        confidence=None,
        correct=False,
        error=error,
        error_type=error_type,
        args=args,
        latency_ms=latency_ms,
        attempts=attempts,
        usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
        },
        raw_output=raw_output,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def compute_metrics(
    records: list[dict[str, Any]],
    labels: list[str],
    *,
    model: str,
    prompt_variant: str,
    wall_clock_s: float,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    scored = [r for r in records if r.get("predicted_label") in labels]
    n_total = len(records)
    n_scored = len(scored)
    n_errors = n_total - n_scored
    correct = sum(1 for r in scored if r["correct"])
    accuracy = correct / n_scored if n_scored else 0.0
    low_ci, high_ci = wilson_interval(correct, n_scored)

    per_class: dict[str, dict[str, Any]] = {}
    for label in labels:
        gold_rows = [r for r in scored if r["gold_label"] == label]
        predicted_rows = [r for r in scored if r["predicted_label"] == label]
        true_positive = sum(1 for r in gold_rows if r["predicted_label"] == label)
        precision = true_positive / len(predicted_rows) if predicted_rows else 0.0
        recall = true_positive / len(gold_rows) if gold_rows else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "support": len(gold_rows),
            "correct": true_positive,
            "accuracy": true_positive / len(gold_rows) if gold_rows else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    classes_with_support = [v for v in per_class.values() if v["support"] > 0]
    macro_f1 = (
        sum(v["f1"] for v in classes_with_support) / len(classes_with_support)
        if classes_with_support
        else 0.0
    )
    perfect_classes = sum(
        1 for v in per_class.values() if v["support"] and v["correct"] == v["support"]
    )
    zero_classes = sum(1 for v in per_class.values() if v["support"] and v["correct"] == 0)

    confusion_counter = Counter(
        (r["gold_label"], r["predicted_label"]) for r in scored if not r["correct"]
    )
    top_confusions = [
        {"gold": gold, "predicted": predicted, "count": count}
        for (gold, predicted), count in confusion_counter.most_common(20)
    ]

    # Calibration: only rows that returned a confidence value.
    with_confidence = [r for r in scored if r["confidence"] is not None]
    calibration_bins: list[dict[str, Any]] = []
    ece = 0.0
    bin_edges = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0001)]
    for low, high in bin_edges:
        bucket = [r for r in with_confidence if low <= r["confidence"] < high]
        if not bucket:
            continue
        avg_confidence = sum(r["confidence"] for r in bucket) / len(bucket)
        bucket_accuracy = sum(1 for r in bucket if r["correct"]) / len(bucket)
        calibration_bins.append(
            {
                "low": low,
                "high": min(high, 1.0),
                "n": len(bucket),
                "avg_confidence": round(avg_confidence, 4),
                "accuracy": round(bucket_accuracy, 4),
            }
        )
        if with_confidence:
            ece += len(bucket) / len(with_confidence) * abs(bucket_accuracy - avg_confidence)

    # Accuracy-coverage curve: if the lowest-confidence items are sent to a
    # human, how accurate is the auto-labeled remainder?
    ordered = sorted(
        scored,
        key=lambda r: (
            r["confidence"] is not None,
            r["confidence"] if r["confidence"] is not None else -1.0,
        ),
        reverse=True,
    )
    total_errors = n_scored - correct
    coverage_curve: list[dict[str, Any]] = []
    for coverage in ((1.0, 0.9, 0.8, 0.7, 0.6, 0.5) if n_scored else ()):
        k = max(1, math.ceil(coverage * n_scored))
        subset = ordered[:k]
        subset_correct = sum(1 for r in subset if r["correct"])
        errors_caught = len(subset) - subset_correct
        coverage_curve.append(
            {
                "coverage": coverage,
                "n_auto_labeled": k,
                "accuracy": round(subset_correct / len(subset), 4),
                "errors_caught": errors_caught,
                "error_recall": round(errors_caught / total_errors, 4)
                if total_errors
                else 0.0,
            }
        )

    # Length bias: accuracy by word-count quartile.
    length_bins: list[dict[str, Any]] = []
    if scored:
        lengths = sorted(r["word_count"] for r in scored)
        q1 = lengths[len(lengths) // 4]
        q2 = lengths[len(lengths) // 2]
        q3 = lengths[(3 * len(lengths)) // 4]

        def bucket_name(words: int) -> str:
            if words <= q1:
                return "Q1_shortest"
            if words <= q2:
                return "Q2"
            if words <= q3:
                return "Q3"
            return "Q4_longest"

        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in scored:
            grouped.setdefault(bucket_name(record["word_count"]), []).append(record)
        for name in ("Q1_shortest", "Q2", "Q3", "Q4_longest"):
            group = grouped.get(name, [])
            if not group:
                continue
            group_correct = sum(1 for r in group if r["correct"])
            length_bins.append(
                {
                    "bucket": name,
                    "n": len(group),
                    "accuracy": round(group_correct / len(group), 4),
                    "avg_confidence": round(
                        sum(r["confidence"] for r in group if r["confidence"] is not None)
                        / max(1, sum(1 for r in group if r["confidence"] is not None)),
                        4,
                    ),
                }
            )

    # Prediction bias: which labels are over/under-predicted relative to gold?
    prediction_counts = Counter(r["predicted_label"] for r in scored)
    prediction_bias = []
    for label in labels:
        predicted_count = prediction_counts.get(label, 0)
        gold_count = per_class[label]["support"]
        prediction_bias.append(
            {
                "label": label,
                "predicted_count": predicted_count,
                "gold_count": gold_count,
                "delta": predicted_count - gold_count,
            }
        )
    prediction_bias.sort(key=lambda row: row["delta"], reverse=True)

    latencies = [r["latency_ms"] for r in scored if r["latency_ms"] is not None]
    latency_summary = {}
    if latencies:
        latencies_sorted = sorted(latencies)
        latency_summary = {
            "mean_ms": round(sum(latencies) / len(latencies), 1),
            "p50_ms": round(latencies_sorted[len(latencies_sorted) // 2], 1),
            "p95_ms": round(
                latencies_sorted[min(len(latencies_sorted) - 1, int(0.95 * len(latencies_sorted)))],
                1,
            ),
            "min_ms": round(min(latencies), 1),
            "max_ms": round(max(latencies), 1),
        }

    usage_totals = {
        key: sum(int(r["usage"].get(key, 0)) for r in records)
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
        )
    }
    cost = estimate_cost(usage_totals, model)
    actual_errors = sum(1 for r in records if r.get("error"))
    valid_labels = sum(1 for r in records if r.get("predicted_label") in labels)
    # n_errors is the count of rows without a valid label (API/parse failure).
    # It is *not* the number of wrong labels; keep the two concepts separate.
    n_api_errors = n_errors
    n_misclassified = n_total - correct

    return {
        "run": {
            **run_metadata,
            "model": model,
            "prompt_variant": prompt_variant,
            "wall_clock_s": round(wall_clock_s, 2),
            "n_total": n_total,
            "n_scored": n_scored,
            "n_invalid_label_or_parse": n_errors,
            "n_api_errors": actual_errors,
            "n_valid_labels": valid_labels,
            "accuracy_ci95_wilson": [round(low_ci, 4), round(high_ci, 4)],
            "notes": (
                "accuracy excludes rows with no valid label; "
                "n_invalid_label_or_parse counts them. "
                "n_misclassified in summary counts wrong labels. "
                "The sample is 3 examples per intent, so per-class accuracy is coarse."
            ),
        },
        "summary": {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "correct": correct,
            "n_scored": n_scored,
            "n_total": n_total,
            "n_api_errors": n_api_errors,
            "n_misclassified": n_misclassified,
            "perfect_classes": perfect_classes,
            "zero_accuracy_classes": zero_classes,
            "n_classes": len(labels),
        },
        "per_class": per_class,
        "top_confusions": top_confusions,
        "calibration": {
            "bins": calibration_bins,
            "ece": round(ece, 4) if with_confidence else None,
            "n_with_confidence": len(with_confidence),
        },
        "coverage_curve": coverage_curve,
        "length_bins": length_bins,
        "prediction_bias_top_over": prediction_bias[:10],
        "prediction_bias_top_under": list(reversed(prediction_bias[-10:])),
        "latency": latency_summary,
        "usage": usage_totals,
        "cost": cost,
    }


def estimate_cost(usage_totals: dict[str, int], model: str) -> dict[str, Any] | None:
    prices = PRICES_USD_PER_MTOK.get(model)
    if not prices:
        return None
    hit = usage_totals["cache_hit_tokens"]
    miss = usage_totals["cache_miss_tokens"]
    output = usage_totals["completion_tokens"]
    off_peak = (
        hit * prices["cache_hit"] + miss * prices["cache_miss"] + output * prices["output"]
    ) / 1_000_000
    return {
        "off_peak_usd": round(off_peak, 6),
        "peak_usd": round(off_peak * 2, 6),
        "prices_usd_per_mtok": prices,
        "source": PRICE_SOURCE,
        "as_of": PRICE_AS_OF,
        "note": "Peak = 01:00-04:00 and 06:00-10:00 UTC on weekdays; off-peak is half.",
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON ({exc})") from exc
            for key in ("id", "text", "gold_label"):
                if key not in row:
                    raise SystemExit(f"{path}:{line_number}: missing {key!r}")
            row["order"] = len(rows)
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path}: duplicate ids")
    return rows


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda r: r["order"]):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_artifacts(
    metrics_path: Path,
    metrics: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    stem = metrics_path.with_suffix("")
    write_csv(
        Path(f"{stem}.confusions.csv"),
        [
            {
                "gold_label": row["gold"],
                "predicted_label": row["predicted"],
                "count": row["count"],
            }
            for row in metrics["top_confusions"]
        ],
        ["gold_label", "predicted_label", "count"],
    )
    error_rows = [
        {
            "id": record["id"],
            "text": record["text"],
            "gold_label": record["gold_label"],
            "predicted_label": record["predicted_label"] or "",
            "confidence": record["confidence"] if record["confidence"] is not None else "",
            "word_count": record["word_count"],
            "error": record["error"] or "",
            "raw_output": record["raw_output"],
        }
        for record in sorted(records, key=lambda r: r["order"])
        if not record["correct"]
    ]
    write_csv(
        Path(f"{stem}.errors.csv"),
        error_rows,
        [
            "id",
            "text",
            "gold_label",
            "predicted_label",
            "confidence",
            "word_count",
            "error",
            "raw_output",
        ],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/banking77_sample.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/predictions_zero_shot.jsonl")
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Defaults to <output stem>.metrics.json",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--prompt", choices=PROMPT_VARIANTS, default="zero_shot")
    parser.add_argument("--provider", choices=("deepseek", "mock"), default="deepseek")
    parser.add_argument("--limit", type=int, default=None, help="Pilot on the first N rows")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.provider == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY"):
        parser.error(
            "DEEPSEEK_API_KEY is not set. Export it, or use --provider mock for a "
            "network-free smoke test."
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    all_rows = load_jsonl(args.input)
    labels = sorted({row["gold_label"] for row in all_rows})
    rows = all_rows[: args.limit] if args.limit else all_rows

    system_prompt = build_system_prompt(labels, args.prompt)
    metrics_path = args.metrics or args.output.with_name(
        f"{args.output.stem}.metrics.json"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = time.perf_counter()
    records: list[dict[str, Any]] = []

    if not args.quiet:
        print(
            f"provider={args.provider} model={args.model} prompt={args.prompt} "
            f"n={len(rows)} labels={len(labels)} concurrency={args.concurrency}",
            file=sys.stderr,
        )

    if args.provider == "mock":
        records = [mock_process_one(row, labels, args) for row in rows]
    else:
        api_key = os.environ["DEEPSEEK_API_KEY"]
        url = args.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        limits = httpx.Limits(
            max_connections=max(4, args.concurrency + 4),
            max_keepalive_connections=max(4, args.concurrency),
        )
        with httpx.Client(limits=limits, follow_redirects=True) as client:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [
                    pool.submit(
                        process_one,
                        client,
                        row,
                        system_prompt,
                        labels,
                        args,
                        url,
                        api_key,
                    )
                    for row in rows
                ]
                error_count = 0
                for index, future in enumerate(as_completed(futures), start=1):
                    record = future.result()
                    records.append(record)
                    if record.get("error"):
                        error_count += 1
                    if not args.quiet and (index % 20 == 0 or index == len(rows)):
                        elapsed = time.perf_counter() - started
                        rate = index / elapsed if elapsed else 0.0
                        print(
                            f"[{index}/{len(rows)}] {rate:.1f} items/s errors={error_count}",
                            file=sys.stderr,
                        )

    wall_clock_s = time.perf_counter() - started
    finished_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics = compute_metrics(
        records,
        labels,
        model=args.model,
        prompt_variant=args.prompt,
        wall_clock_s=wall_clock_s,
        run_metadata={
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "input": str(args.input),
            "output": str(args.output),
            "provider": args.provider,
            "concurrency": args.concurrency,
            "temperature": 0,
            "python": sys.version.split()[0],
        },
    )

    write_jsonl(args.output, records)
    write_artifacts(metrics_path, metrics, records)

    summary = metrics["summary"]
    if not args.quiet:
        print(
            f"accuracy={summary['accuracy']:.4f} macro_f1={summary['macro_f1']:.4f} "
            f"correct={summary['correct']}/{summary['n_scored']} "
            f"misclassified={summary['n_misclassified']} "
            f"api_errors={summary['n_api_errors']} wall={wall_clock_s:.1f}s",
            file=sys.stderr,
        )
        cost = metrics.get("cost") or {}
        if cost:
            print(
                f"tokens={metrics['usage']} cost_off_peak=${cost['off_peak_usd']:.6f} "
                f"(peak ${cost['peak_usd']:.6f})",
                file=sys.stderr,
            )
        print(f"wrote {args.output} and {metrics_path}", file=sys.stderr)

    if summary["n_api_errors"] == len(records) and records:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
