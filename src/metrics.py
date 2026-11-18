"""
Evaluation metrics.

Language-ID accuracy, task accuracy (token overlap), perplexity, bootstrap CI.
"""

import math
import random
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Return ISO 639-1 language code, or 'unknown' on failure."""
    if not text or len(text.strip()) < 3:
        return "unknown"
    try:
        from langdetect import detect, LangDetectException
        try:
            return detect(text)
        except LangDetectException:
            return "unknown"
    except ImportError:
        pass
    # Fallback: simple heuristic — Swedish chars
    sv_chars = set("åäöÅÄÖ")
    if any(c in sv_chars for c in text):
        return "sv"
    return "unknown"


def language_id_accuracy(records: list[dict], expected_lang: str = "sv") -> dict:
    """
    Compute fraction of records where detected language == expected_lang.

    Returns dict with n, correct, accuracy, plus per-record labels.
    """
    labels = []
    for r in records:
        response = r.get("response", "")
        detected = detect_language(response)
        correct = detected == expected_lang or (
            expected_lang == "sv" and detected in ("sv", "no", "da")  # Nordic overlap
        )
        labels.append({
            "id": r.get("id"),
            "detected_lang": detected,
            "correct": correct,
        })
    n = len(labels)
    n_correct = sum(l["correct"] for l in labels)
    return {
        "n": n,
        "correct": n_correct,
        "accuracy": n_correct / n if n > 0 else 0.0,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Task accuracy (simple token-overlap / contains-answer)
# ---------------------------------------------------------------------------

def simple_token_overlap(reference: str, hypothesis: str) -> float:
    """Return fraction of reference tokens present in hypothesis."""
    ref_tokens = set(reference.lower().split())
    hyp_tokens = set(hypothesis.lower().split())
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & hyp_tokens) / len(ref_tokens)


def task_accuracy(records: list[dict], threshold: float = 0.3) -> dict:
    """
    Compute task accuracy using token overlap >= threshold as correct.

    Records must have 'source_text' and 'response' fields.
    """
    scores = []
    for r in records:
        overlap = simple_token_overlap(r.get("source_text", ""), r.get("response", ""))
        scores.append({
            "id": r.get("id"),
            "overlap": overlap,
            "correct": overlap >= threshold,
        })
    n = len(scores)
    n_correct = sum(s["correct"] for s in scores)
    return {
        "n": n,
        "correct": n_correct,
        "accuracy": n_correct / n if n > 0 else 0.0,
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    device: str = "cpu",
    batch_size: int = 4,
) -> float:
    """Compute mean per-token perplexity over list of texts."""
    import torch

    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
            # out.loss is mean NLL per token (mean over non-padding tokens)
            n_tokens = enc["attention_mask"].sum().item()
            total_nll += out.loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """
    Bootstrap CI for mean of a list of floats.

    Returns dict with mean, lower, upper.
    """
    rng = random.Random(seed)
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}

    boot_means = []
    n = len(arr)
    for _ in range(n_boot):
        sample = rng.choices(range(n), k=n)
        boot_means.append(arr[sample].mean())

    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2
    return {
        "mean": float(arr.mean()),
        "lower": float(np.quantile(boot_means, alpha)),
        "upper": float(np.quantile(boot_means, 1 - alpha)),
    }


def proportion_ci(successes: int, n: int, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap CI for a proportion (bernoulli)."""
    values = [1.0] * successes + [0.0] * (n - successes)
    return bootstrap_ci(values, n_boot=n_boot, seed=seed)


# ---------------------------------------------------------------------------
# Stratified analysis: failure removal vs new capability
# ---------------------------------------------------------------------------

def stratify_by_base_response(
    base_records: list[dict],
    edit_records: list[dict],
    expected_lang: str = "sv",
) -> dict:
    """
    Given base and edited condition records (same IDs, same order),
    split into:
      - Stratum A: base already in correct language
      - Stratum B: base drifted (wrong language)

    For each stratum, compute accuracy under the edit condition.
    """
    assert len(base_records) == len(edit_records)

    strata = {"A": [], "B": []}
    for b, e in zip(base_records, edit_records):
        assert b["id"] == e["id"], f"ID mismatch: {b['id']} vs {e['id']}"
        base_lang = detect_language(b.get("response", ""))
        base_correct_lang = base_lang == expected_lang or (
            expected_lang == "sv" and base_lang in ("sv", "no", "da")
        )
        edit_lang = detect_language(e.get("response", ""))
        edit_correct_lang = edit_lang == expected_lang or (
            expected_lang == "sv" and edit_lang in ("sv", "no", "da")
        )
        stratum = "A" if base_correct_lang else "B"
        strata[stratum].append({
            "id": b["id"],
            "base_lang": base_lang,
            "edit_lang": edit_lang,
            "base_correct": base_correct_lang,
            "edit_correct": edit_correct_lang,
        })

    result = {}
    for stratum, items in strata.items():
        n = len(items)
        n_edit_correct = sum(x["edit_correct"] for x in items)
        n_base_correct = sum(x["base_correct"] for x in items)
        result[stratum] = {
            "n": n,
            "base_lang_correct": n_base_correct,
            "edit_lang_correct": n_edit_correct,
            "base_accuracy": n_base_correct / n if n > 0 else 0.0,
            "edit_accuracy": n_edit_correct / n if n > 0 else 0.0,
        }

    return result
