"""
Evaluation pipeline.

Runs all conditions on a set of prompts, saves results to JSONL,
and computes per-condition metrics including bootstrap CIs.
"""

import copy
import json
import time
from pathlib import Path
from typing import Optional

import jsonlines
import torch

from .metrics import (
    language_id_accuracy,
    proportion_ci,
    stratify_by_base_response,
    compute_perplexity,
)
from .model_utils import generate, get_device


RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_condition(
    model,
    tokenizer,
    prompts: list[dict],
    condition_name: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 100,
    device: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> list[dict]:
    """
    Run a single condition on all prompts, save results to JSONL, return records.
    """
    device = device or get_device()
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{condition_name}.jsonl"

    # Build prompt texts
    prompt_texts = []
    for p in prompts:
        if system_prompt:
            # Inject system prompt via chat template
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": p["prompt"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = f"{system_prompt}\n\n{p['prompt']}"
        else:
            text = p["prompt"]
        prompt_texts.append(text)

    responses = generate(
        model,
        tokenizer,
        prompt_texts,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        device=device,
    )

    records = []
    for p, resp in zip(prompts, responses):
        records.append({
            "id": p["id"],
            "condition": condition_name,
            "prompt": p["prompt"],
            "source_text": p.get("source_text", ""),
            "expected_lang": p.get("expected_lang", "sv"),
            "response": resp,
        })

    with jsonlines.open(str(out_path), mode="w") as writer:
        writer.write_all(records)

    print(f"  {condition_name}: {len(records)} records → {out_path}")
    return records


def evaluate_records(records: list[dict], expected_lang: str = "sv") -> dict:
    """Compute metrics for a set of records."""
    lang_result = language_id_accuracy(records, expected_lang=expected_lang)
    ci = proportion_ci(lang_result["correct"], lang_result["n"])
    return {
        "n": lang_result["n"],
        "lang_id_accuracy": ci["mean"],
        "lang_id_ci_lower": ci["lower"],
        "lang_id_ci_upper": ci["upper"],
        "lang_id_correct": lang_result["correct"],
    }


def run_all_conditions(
    base_model,
    tokenizer,
    test_prompts: list[dict],
    edit_conditions: dict,  # name -> model (or None for prompt-only conditions)
    system_prompts: dict,   # name -> system_prompt string (or None)
    expected_lang: str = "sv",
    max_new_tokens: int = 100,
    device: Optional[str] = None,
    out_dir: Optional[Path] = None,
    perplexity_texts: Optional[list[str]] = None,
) -> dict:
    """
    Run all conditions and return a summary table.

    edit_conditions: dict of condition_name -> model to use (None means base_model)
    system_prompts: dict of condition_name -> system prompt (None means no system prompt)
    """
    device = device or get_device()
    out_dir = out_dir or RESULTS_DIR
    summary = {}
    base_records = None

    # Run baseline first (needed for stratification)
    all_records = {}
    for cond_name, model in edit_conditions.items():
        sys_prompt = system_prompts.get(cond_name)
        m = model if model is not None else base_model
        records = run_condition(
            m, tokenizer, test_prompts, cond_name,
            system_prompt=sys_prompt,
            max_new_tokens=max_new_tokens,
            device=device,
            out_dir=out_dir,
        )
        all_records[cond_name] = records
        summary[cond_name] = evaluate_records(records, expected_lang=expected_lang)

        if cond_name == "C0_baseline":
            base_records = records

        # Perplexity regression
        if perplexity_texts and m is not None:
            ppl = compute_perplexity(m, tokenizer, perplexity_texts, device=device)
            summary[cond_name]["perplexity_sv"] = ppl

    # Stratified analysis (failure removal vs new capability)
    if base_records is not None:
        for cond_name, records in all_records.items():
            if cond_name == "C0_baseline":
                continue
            strata = stratify_by_base_response(base_records, records, expected_lang)
            summary[cond_name]["stratum_A"] = strata.get("A", {})
            summary[cond_name]["stratum_B"] = strata.get("B", {})

    # Save summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return summary


def print_summary_table(summary: dict) -> None:
    """Print a formatted results table to stdout."""
    print("\n" + "=" * 80)
    print(f"{'Condition':<25} {'LangID%':>8} {'CI_lo':>8} {'CI_hi':>8} {'PPL_sv':>8}")
    print("-" * 80)
    for cond, m in summary.items():
        lang_pct = m.get("lang_id_accuracy", float("nan")) * 100
        ci_lo = m.get("lang_id_ci_lower", float("nan")) * 100
        ci_hi = m.get("lang_id_ci_upper", float("nan")) * 100
        ppl = m.get("perplexity_sv", float("nan"))
        print(f"{cond:<25} {lang_pct:>7.1f}% {ci_lo:>7.1f}% {ci_hi:>7.1f}% {ppl:>8.1f}")
    print("=" * 80)
