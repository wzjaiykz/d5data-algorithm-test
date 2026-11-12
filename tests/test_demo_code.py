import json

import pytest

import demo_code

LABELS = ["card_arrival", "card_not_working", "Refund_not_showing_up"]


def test_parse_valid_json():
    label, confidence = demo_code.parse_response(
        '{"label": "card_arrival", "confidence": 0.91}', LABELS
    )
    assert label == "card_arrival"
    assert confidence == pytest.approx(0.91)


def test_parse_normalizes_case_and_spaces():
    label, _ = demo_code.parse_response(
        '{"label": " refund_not_showing_up ", "confidence": 0.5}', LABELS
    )
    assert label == "Refund_not_showing_up"


def test_parse_accepts_json_inside_prose():
    label, _ = demo_code.parse_response(
        'Here you go: {"label": "card_not_working", "confidence": 0.7}', LABELS
    )
    assert label == "card_not_working"


def test_parse_rejects_unknown_label():
    with pytest.raises(ValueError, match="not in the allowed set"):
        demo_code.parse_response('{"label": "not_a_label", "confidence": 0.5}', LABELS)


def test_parse_rejects_empty_response():
    with pytest.raises(ValueError, match="empty"):
        demo_code.parse_response("", LABELS)


def test_prompt_contains_all_labels_and_schema():
    prompt = demo_code.build_system_prompt(LABELS, "zero_shot")
    for label in LABELS:
        assert label in prompt
    assert "Allowed labels" in prompt
    assert '"label"' in prompt
    assert '"confidence"' in prompt


def test_hinted_prompt_adds_disambiguation_section():
    prompt = demo_code.build_system_prompt(LABELS, "hinted")
    assert "Disambiguation hints" in prompt
    assert "get_physical_card" in prompt


def _record(gold: str, predicted: str, confidence: float, index: int) -> dict:
    return {
        "id": f"t{index}",
        "order": index,
        "text": f"text {index}",
        "gold_label": gold,
        "predicted_label": predicted,
        "confidence": confidence,
        "correct": gold == predicted,
        "error": None,
        "error_type": None,
        "model": "mock",
        "prompt_variant": "zero_shot",
        "provider": "mock",
        "latency_ms": 1.0,
        "attempts": 1,
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 10,
        },
        "raw_output": "{}",
        "word_count": 2,
        "timestamp_utc": "2026-11-12T00:00:00+00:00",
    }


def test_metrics_basic():
    records = [
        _record("card_arrival", "card_arrival", 0.9, 0),
        _record("card_not_working", "card_not_working", 0.8, 1),
        _record("Refund_not_showing_up", "Refund_not_showing_up", 0.7, 2),
        _record("Refund_not_showing_up", "card_arrival", 0.4, 3),
    ]
    metrics = demo_code.compute_metrics(
        records,
        LABELS,
        model="deepseek-flash",
        prompt_variant="zero_shot",
        wall_clock_s=1.0,
        run_metadata={},
    )
    assert metrics["summary"]["accuracy"] == pytest.approx(0.75)
    assert metrics["summary"]["correct"] == 3
    assert metrics["summary"]["n_total"] == 4
    assert metrics["top_confusions"][0] == {
        "gold": "Refund_not_showing_up",
        "predicted": "card_arrival",
        "count": 1,
    }
    assert metrics["calibration"]["ece"] is not None
    assert metrics["cost"]["off_peak_usd"] > 0


def test_cost_estimate_uses_cache_and_output_prices():
    usage = {
        "cache_hit_tokens": 0,
        "cache_miss_tokens": 1_000_000,
        "completion_tokens": 1_000_000,
    }
    cost = demo_code.estimate_cost(usage, "deepseek-flash")
    assert cost["off_peak_usd"] == pytest.approx(0.75)
    assert cost["peak_usd"] == pytest.approx(1.50)


def test_mock_end_to_end(work_dir):
    rows = [
        {"id": f"t{i}", "text": f"synthetic utterance {i}", "gold_label": LABELS[i % 3]}
        for i in range(6)
    ]
    input_path = work_dir / "input.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    output_path = work_dir / "predictions.jsonl"

    exit_code = demo_code.main(
        [
            "--provider",
            "mock",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--quiet",
        ]
    )

    assert exit_code == 0
    predictions = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(predictions) == 6
    assert all(row["provider"] == "mock" for row in predictions)
    assert all(row["id"].startswith("t") for row in predictions)

    metrics_path = work_dir / "predictions.metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["summary"]["n_total"] == 6
    assert metrics["summary"]["n_scored"] == 6
    assert (work_dir / "predictions.metrics.errors.csv").exists()
    assert (work_dir / "predictions.metrics.confusions.csv").exists()


def test_load_jsonl_rejects_duplicate_ids(work_dir):
    path = work_dir / "bad.jsonl"
    path.write_text(
        '{"id": "x", "text": "a", "gold_label": "card_arrival"}\n'
        '{"id": "x", "text": "b", "gold_label": "card_arrival"}\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        demo_code.load_jsonl(path)
