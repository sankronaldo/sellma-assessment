"""
Tests for data split disjointness and integrity.

Run with: python -m pytest tests/test_splits.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import build_qa_prompts


def test_qa_prompts_swedish():
    sentences = [{"id": "s1", "text": "Huset är rött och katten sover."}, {"id": "s2", "text": "Hunden springer i parken."}]
    prompts = build_qa_prompts(sentences, lang="sv")
    assert len(prompts) == 2
    for p in prompts:
        assert "svenska" in p["prompt"].lower() or "sv" in p.get("expected_lang", "")
        assert p["expected_lang"] == "sv"
        assert p["id"] in ("s1", "s2")


def test_qa_prompts_english():
    sentences = [{"id": "e1", "text": "The house is red."}]
    prompts = build_qa_prompts(sentences, lang="en")
    assert prompts[0]["expected_lang"] == "en"
    assert "English" in prompts[0]["prompt"]


def test_qa_prompts_no_id_collision():
    sv_sentences = [{"id": f"sv_{i}", "text": f"Mening {i}."} for i in range(5)]
    en_sentences = [{"id": f"en_{i}", "text": f"Sentence {i}."} for i in range(5)]
    sv_prompts = build_qa_prompts(sv_sentences, lang="sv")
    en_prompts = build_qa_prompts(en_sentences, lang="en")
    sv_ids = {p["id"] for p in sv_prompts}
    en_ids = {p["id"] for p in en_prompts}
    assert len(sv_ids & en_ids) == 0, "Swedish and English prompts share IDs"


def test_qa_prompts_contains_source():
    sentences = [{"id": "x1", "text": "Katten sitter på taket."}]
    prompts = build_qa_prompts(sentences, lang="sv")
    assert "Katten sitter på taket." in prompts[0]["prompt"]
    assert prompts[0]["source_text"] == "Katten sitter på taket."


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
