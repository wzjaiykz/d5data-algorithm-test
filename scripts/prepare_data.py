#!/usr/bin/env python3
"""Build a deterministic, class-balanced sample from the official Banking77 test split.

Banking77 is distributed as two CSV files (text,category) by PolyAI-LDN:
https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data

The script downloads the test split, takes PER_INTENT examples per intent with a
fixed seed, shuffles them, and writes JSONL plus a metadata file. The committed
sample lets `demo_code.py` run without network access.
"""

from __future__ import annotations

import csv
import io
import json
import random
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SOURCE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/test.csv"
)
LICENSE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/LICENSE"
)
SEED = 42
PER_INTENT = 3

CITATION = (
    "Casanueva et al. (2020), Efficient Intent Detection with Dual Sentence "
    "Encoders, https://arxiv.org/abs/2003.04807"
)


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "d5data-demo/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = download(SOURCE_URL).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(raw)))
    if not rows or set(rows[0]) != {"text", "category"}:
        raise SystemExit("Unexpected CSV header; expected text,category")

    by_label: dict[str, list[str]] = {}
    for row in rows:
        by_label.setdefault(row["category"], []).append(row["text"])

    if len(by_label) != 77:
        raise SystemExit(f"Expected 77 intents, found {len(by_label)}")
    too_small = {label: len(texts) for label, texts in by_label.items() if len(texts) < PER_INTENT}
    if too_small:
        raise SystemExit(f"Not enough examples for: {too_small}")

    rng = random.Random(SEED)
    sample: list[dict[str, str]] = []
    for label in sorted(by_label):
        # Sort first so the selection does not depend on upstream CSV row order.
        for text in rng.sample(sorted(set(by_label[label])), PER_INTENT):
            sample.append({"text": text, "gold_label": label})

    rng.shuffle(sample)
    for index, item in enumerate(sample, start=1):
        item["id"] = f"b77-{index:04d}"
        item["split"] = "test"

    sample_path = DATA_DIR / "banking77_sample.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for item in sample:
            ordered = {
                "id": item["id"],
                "text": item["text"],
                "gold_label": item["gold_label"],
                "split": item["split"],
            }
            handle.write(json.dumps(ordered, ensure_ascii=False) + "\n")

    labels = sorted(by_label)
    (DATA_DIR / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    metadata = {
        "dataset": "Banking77",
        "source_url": SOURCE_URL,
        "license": "CC BY 4.0",
        "license_url": LICENSE_URL,
        "citation": CITATION,
        "split": "test",
        "source_test_rows": len(rows),
        "n_intents": len(labels),
        "per_intent_total_in_test": sorted({len(v) for v in by_label.values()}),
        "seed": SEED,
        "per_intent_sampled": PER_INTENT,
        "n_samples": len(sample),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (DATA_DIR / "banking77_sample.meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(sample)} samples ({len(labels)} intents x {PER_INTENT}) to {sample_path}")
    print(f"Metadata: {DATA_DIR / 'banking77_sample.meta.json'}")


if __name__ == "__main__":
    main()
