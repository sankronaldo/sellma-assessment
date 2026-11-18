"""
Data loading and split utilities.

Creates disjoint dev and test splits from FLORES-200 Swedish/English data
and Swedish Wikipedia perplexity sentences. Saves splits to disk and verifies
no overlap between dev and test example IDs.
"""

import json
import random
from pathlib import Path
from typing import Optional

import jsonlines

SPLIT_SEED = 42
DATA_DIR = Path(__file__).parent.parent / "results"
CONFIG_DIR = Path(__file__).parent.parent / "configs"


def load_flores_sentences(lang_code: str, split: str = "dev") -> list[dict]:
    """Load FLORES-200 sentences for a given language code and split."""
    from datasets import load_dataset
    dataset = load_dataset("facebook/flores", lang_code, trust_remote_code=True)
    rows = []
    for i, item in enumerate(dataset[split]):
        rows.append({"id": f"flores_{lang_code}_{split}_{i}", "text": item["sentence"], "lang": lang_code})
    return rows


def load_swedish_wikipedia(n_sentences: int = 200, seed: int = SPLIT_SEED) -> list[dict]:
    """Load n_sentences from Swedish Wikipedia for perplexity evaluation."""
    from datasets import load_dataset
    dataset = load_dataset("wikipedia", "20231101.sv", trust_remote_code=True)
    rng = random.Random(seed)
    # Sample n_sentences from the first 50k articles, pulling one sentence each
    articles = list(dataset["train"])
    rng.shuffle(articles)
    sentences = []
    for art in articles:
        # Split on period; take first clean sentence of reasonable length
        for sent in art["text"].split(". "):
            sent = sent.strip()
            if 20 < len(sent) < 300 and sent.isascii() is False:
                sentences.append({"id": f"wiki_sv_{len(sentences)}", "text": sent})
                if len(sentences) >= n_sentences:
                    break
        if len(sentences) >= n_sentences:
            break
    return sentences


def build_splits(
    dev_size: int = 50,
    test_size: int = 100,
    seed: int = SPLIT_SEED,
    out_dir: Optional[Path] = None,
) -> dict:
    """
    Build and save dev/test splits.

    Returns dict with keys dev_sv, test_sv, dev_en, test_en, perplexity_sv.
    Checks for ID overlap between dev and test before returning.
    """
    out_dir = out_dir or CONFIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    split_file = out_dir / "data_split.json"
    if split_file.exists():
        with open(split_file) as f:
            return json.load(f)

    rng = random.Random(seed)

    sv_dev_raw = load_flores_sentences("swe_Latn", "dev")
    sv_test_raw = load_flores_sentences("swe_Latn", "devtest")
    en_dev_raw = load_flores_sentences("eng_Latn", "dev")
    en_test_raw = load_flores_sentences("eng_Latn", "devtest")

    rng.shuffle(sv_dev_raw)
    rng.shuffle(en_dev_raw)

    dev_sv = sv_dev_raw[:dev_size]
    test_sv = sv_test_raw[:test_size]
    dev_en = en_dev_raw[:dev_size]
    test_en = en_test_raw[:test_size]

    # Leakage check
    dev_ids = {r["id"] for r in dev_sv} | {r["id"] for r in dev_en}
    test_ids = {r["id"] for r in test_sv} | {r["id"] for r in test_en}
    overlap = dev_ids & test_ids
    assert len(overlap) == 0, f"Dev/test overlap: {overlap}"

    perplexity_sv = load_swedish_wikipedia(n_sentences=200, seed=seed)

    splits = {
        "dev_sv": dev_sv,
        "test_sv": test_sv,
        "dev_en": dev_en,
        "test_en": test_en,
        "perplexity_sv": perplexity_sv,
        "seed": seed,
        "dev_size": dev_size,
        "test_size": test_size,
    }

    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    print(f"Splits saved to {split_file}")
    print(f"  dev_sv: {len(dev_sv)}, test_sv: {len(test_sv)}")
    print(f"  dev_en: {len(dev_en)}, test_en: {len(test_en)}")
    print(f"  perplexity_sv: {len(perplexity_sv)}")
    return splits


def load_splits(out_dir: Optional[Path] = None) -> dict:
    """Load pre-built splits from disk."""
    out_dir = out_dir or CONFIG_DIR
    split_file = out_dir / "data_split.json"
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found at {split_file}. Run build_splits() first."
        )
    with open(split_file, encoding="utf-8") as f:
        return json.load(f)


def build_qa_prompts(sentences: list[dict], lang: str = "sv") -> list[dict]:
    """
    Turn plain sentences into QA-style prompts.

    For Swedish: prompt asks for a Swedish response.
    For English: prompt asks for an English response.
    The sentence is used as context; the model is asked to summarise it.
    This keeps the task simple and ensures an exact-language answer is expected.
    """
    prompts = []
    for item in sentences:
        if lang == "sv":
            prompt = f"Sammanfatta följande mening på svenska:\n\n{item['text']}\n\nSvar:"
        else:
            prompt = f"Summarize the following sentence in English:\n\n{item['text']}\n\nAnswer:"
        prompts.append({
            "id": item["id"],
            "prompt": prompt,
            "source_text": item["text"],
            "expected_lang": "sv" if lang == "sv" else "en",
        })
    return prompts


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(str(path), mode="w") as writer:
        writer.write_all(records)


def load_jsonl(path: Path) -> list[dict]:
    with jsonlines.open(str(path), mode="r") as reader:
        return list(reader)
