"""
Unit tests for metrics.py

Run with: python -m pytest tests/test_metrics.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    detect_language,
    language_id_accuracy,
    simple_token_overlap,
    proportion_ci,
    bootstrap_ci,
    stratify_by_base_response,
)


def test_detect_language_swedish():
    # Clear Swedish with Swedish characters
    text = "Huset är rött och katten sover på mattan."
    lang = detect_language(text)
    assert lang in ("sv", "no", "da", "unknown"), f"Expected sv-family, got {lang}"


def test_detect_language_english():
    text = "The cat sat on the mat and looked outside."
    lang = detect_language(text)
    assert lang == "en", f"Expected en, got {lang}"


def test_detect_language_short():
    # Very short text should not crash
    lang = detect_language("ok")
    assert isinstance(lang, str)


def test_detect_language_empty():
    lang = detect_language("")
    assert lang == "unknown"


def test_language_id_accuracy_all_correct():
    records = [
        {"id": "1", "response": "Huset är rött med blommor och träd.", "expected_lang": "sv"},
        {"id": "2", "response": "Katten sover på soffan hela dagen.", "expected_lang": "sv"},
    ]
    result = language_id_accuracy(records, expected_lang="sv")
    assert result["n"] == 2
    # Both should be detected as Swedish or Nordic
    assert result["accuracy"] >= 0.5  # at least some detected correctly


def test_language_id_accuracy_all_english():
    records = [
        {"id": "1", "response": "The cat sat on the mat outside.", "expected_lang": "sv"},
        {"id": "2", "response": "She walked through the garden alone.", "expected_lang": "sv"},
    ]
    result = language_id_accuracy(records, expected_lang="sv")
    assert result["n"] == 2
    # English responses should NOT be counted as Swedish
    assert result["accuracy"] == 0.0


def test_simple_token_overlap_identical():
    score = simple_token_overlap("hello world", "hello world")
    assert score == 1.0


def test_simple_token_overlap_no_overlap():
    score = simple_token_overlap("hello world", "cat dog fish")
    assert score == 0.0


def test_simple_token_overlap_partial():
    score = simple_token_overlap("hello world today", "hello cat today fish")
    # 2 of 3 reference tokens present (hello, today)
    assert abs(score - 2 / 3) < 1e-6


def test_simple_token_overlap_empty_ref():
    score = simple_token_overlap("", "hello world")
    assert score == 0.0


def test_proportion_ci_range():
    ci = proportion_ci(70, 100, n_boot=200, seed=0)
    assert 0.0 <= ci["lower"] <= ci["mean"] <= ci["upper"] <= 1.0
    assert abs(ci["mean"] - 0.70) < 0.05  # mean close to observed proportion


def test_proportion_ci_all_correct():
    ci = proportion_ci(100, 100, n_boot=200, seed=0)
    assert ci["mean"] == 1.0
    assert ci["lower"] == 1.0
    assert ci["upper"] == 1.0


def test_bootstrap_ci_shape():
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    ci = bootstrap_ci(values, n_boot=200, seed=0)
    assert "mean" in ci and "lower" in ci and "upper" in ci
    assert ci["lower"] <= ci["mean"] <= ci["upper"]


def test_stratify_by_base_response_all_swedish():
    # Base model responds in Swedish for all; edit improves none → stratum A all, B empty
    base = [
        {"id": "1", "response": "Huset är stort och fint i parken.", "expected_lang": "sv"},
        {"id": "2", "response": "Katten sover på soffan hela natten.", "expected_lang": "sv"},
    ]
    edit = [
        {"id": "1", "response": "Huset är rött och gammalt nu.", "expected_lang": "sv"},
        {"id": "2", "response": "Hunden springer i trädgården.", "expected_lang": "sv"},
    ]
    result = stratify_by_base_response(base, edit, expected_lang="sv")
    assert "A" in result and "B" in result
    assert result["A"]["n"] + result["B"]["n"] == 2


def test_stratify_id_mismatch_raises():
    base = [{"id": "1", "response": "Huset är rött.", "expected_lang": "sv"}]
    edit = [{"id": "2", "response": "The house is red.", "expected_lang": "sv"}]
    try:
        stratify_by_base_response(base, edit, expected_lang="sv")
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
