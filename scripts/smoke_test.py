"""
CPU smoke test for the full pipeline.

Uses Qwen2.5-0.5B-Instruct on a tiny synthetic subset (20 examples) to verify
that every code path runs without errors before spending Colab compute.

Run with:
  python -I scripts/smoke_test.py

Expected runtime: 5-15 minutes on CPU (depends on hardware).
"""

import sys
import json
import random
import copy
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

RESULTS_DIR = ROOT / "results" / "smoke_test"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR = ROOT / "configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def make_tiny_dataset(n: int = 20) -> tuple[list[dict], list[dict]]:
    """Create tiny synthetic dataset: n Swedish + n English prompt pairs."""
    sv_sentences = [
        "Huset på kullen är stort och vitt.",
        "Katten sover på den röda soffan.",
        "Barnen leker i parken varje dag.",
        "Himlen är blå och solen skiner.",
        "Boken på bordet är mycket intressant.",
        "Sverige är ett land i norra Europa.",
        "Fågeln sjunger i trädet utanför.",
        "Vattnet i sjön är kallt och klart.",
        "Maten på bordet luktar gott.",
        "Hunden springer längs stranden.",
        "Läraren förklarar matematik för eleverna.",
        "Regnet faller mjukt utanför fönstret.",
        "Staden är full av människor på morgonen.",
        "Bilen stannade vid trafikljuset.",
        "Blomman öppnar sina kronblad i solen.",
        "Biblioteket har tusentals böcker.",
        "Floden rinner mot havet i söder.",
        "Skolan börjar klockan åtta varje dag.",
        "Fiskarna simmar i det klara vattnet.",
        "Vinden blåser starkt från norr idag.",
    ]
    en_sentences = [
        "The house on the hill is large and white.",
        "The cat is sleeping on the red sofa.",
        "The children play in the park every day.",
        "The sky is blue and the sun is shining.",
        "The book on the table is very interesting.",
        "Sweden is a country in northern Europe.",
        "The bird is singing in the tree outside.",
        "The water in the lake is cold and clear.",
        "The food on the table smells delicious.",
        "The dog runs along the beach.",
        "The teacher explains mathematics to students.",
        "The rain falls softly outside the window.",
        "The city is full of people in the morning.",
        "The car stopped at the traffic light.",
        "The flower opens its petals in the sun.",
        "The library has thousands of books.",
        "The river flows toward the sea in the south.",
        "School starts at eight o'clock every day.",
        "The fish swim in the clear water.",
        "The wind blows strongly from the north today.",
    ]
    sv_prompts = []
    en_prompts = []
    for i, (sv, en) in enumerate(zip(sv_sentences[:n], en_sentences[:n])):
        sv_prompts.append({
            "id": f"smoke_sv_{i}",
            "prompt": f"Sammanfatta följande mening på svenska:\n\n{sv}\n\nSvar:",
            "source_text": sv,
            "expected_lang": "sv",
        })
        en_prompts.append({
            "id": f"smoke_en_{i}",
            "prompt": f"Summarize the following sentence in English:\n\n{en}\n\nAnswer:",
            "source_text": en,
            "expected_lang": "en",
        })
    return sv_prompts, en_prompts


def main():
    print("=" * 60)
    print("SMOKE TEST — CPU pipeline check")
    print("=" * 60)

    # 1. Load model on CPU
    print("\n[1/6] Loading model on CPU...")
    from src.model_utils import load_model_and_tokenizer, generate
    model, tokenizer = load_model_and_tokenizer(device="cpu", dtype=torch.float32)

    # 2. Build tiny dataset
    print("\n[2/6] Building tiny synthetic dataset (20 examples)...")
    sv_prompts, en_prompts = make_tiny_dataset(n=20)

    # Verify no ID overlap
    sv_ids = {p["id"] for p in sv_prompts}
    en_ids = {p["id"] for p in en_prompts}
    assert len(sv_ids & en_ids) == 0, "ID overlap in smoke data"
    print(f"  {len(sv_prompts)} Swedish + {len(en_prompts)} English prompts, no ID overlap.")

    # Save smoke config
    smoke_config = {"sv_ids": sorted(sv_ids), "en_ids": sorted(en_ids), "seed": SEED}
    with open(CONFIG_DIR / "smoke_split.json", "w") as f:
        json.dump(smoke_config, f, indent=2)

    # 3. Baseline generation (C0)
    print("\n[3/6] Running C0 baseline (no system prompt)...")
    from src.eval_pipeline import run_condition, evaluate_records
    c0_records = run_condition(
        model, tokenizer, sv_prompts[:5], "C0_smoke_baseline",
        max_new_tokens=40, device="cpu", out_dir=RESULTS_DIR
    )
    c0_metrics = evaluate_records(c0_records, expected_lang="sv")
    print(f"  C0 lang-ID accuracy: {c0_metrics['lang_id_accuracy']:.2%} (n={c0_metrics['n']})")

    # C1: Swedish system prompt
    print("\n  Running C1 (Swedish system prompt)...")
    c1_records = run_condition(
        model, tokenizer, sv_prompts[:5], "C1_smoke_sysprompt",
        system_prompt="Svara alltid på svenska.",
        max_new_tokens=40, device="cpu", out_dir=RESULTS_DIR
    )
    c1_metrics = evaluate_records(c1_records, expected_lang="sv")
    print(f"  C1 lang-ID accuracy: {c1_metrics['lang_id_accuracy']:.2%}")

    # 4. Language direction extraction (feedback-based)
    print("\n[4/6] Computing language direction (feedback-based step)...")
    from src.edit_utils import compute_language_direction, layer_direction_scores, apply_direction_ablation

    sv_texts = [p["prompt"] for p in sv_prompts[:10]]
    en_texts = [p["prompt"] for p in en_prompts[:10]]
    directions = compute_language_direction(model, tokenizer, sv_texts, en_texts, device="cpu")
    scores = layer_direction_scores(directions, model)
    print(f"  Top-3 layers by direction score: {scores[:3]}")
    top_layer = scores[0][0]
    print(f"  Chosen layer for edit: {top_layer}")

    # 5. Apply edit (C3)
    print("\n[5/6] Applying direction-ablation edit (alpha=1.0, 64 columns)...")
    edited_model = apply_direction_ablation(
        model, directions, target_layers=[top_layer], alpha=1.0, n_columns=64
    )

    from src.edit_utils import count_edited_params
    edit_info = count_edited_params(model, edited_model)
    print(f"  Changed {edit_info['changed_params']:,} / {edit_info['total_params']:,} params "
          f"({edit_info['fraction_changed']:.4%})")

    c3_records = run_condition(
        edited_model, tokenizer, sv_prompts[:5], "C3_smoke_edit",
        max_new_tokens=40, device="cpu", out_dir=RESULTS_DIR
    )
    c3_metrics = evaluate_records(c3_records, expected_lang="sv")
    print(f"  C3 lang-ID accuracy: {c3_metrics['lang_id_accuracy']:.2%}")

    # 6. Random edit control (C4)
    print("\n[6/6] Random-direction control (C4, 1 seed for smoke)...")
    from src.edit_utils import apply_random_direction_edit
    rand_model = apply_random_direction_edit(
        model, directions[top_layer], target_layers=[top_layer], alpha=1.0, n_columns=64, seed=0
    )
    c4_records = run_condition(
        rand_model, tokenizer, sv_prompts[:5], "C4_smoke_random",
        max_new_tokens=40, device="cpu", out_dir=RESULTS_DIR
    )
    c4_metrics = evaluate_records(c4_records, expected_lang="sv")
    print(f"  C4 lang-ID accuracy: {c4_metrics['lang_id_accuracy']:.2%}")

    # Quick metric unit tests
    print("\n--- Running metric unit tests inline ---")
    from src.metrics import detect_language, simple_token_overlap, proportion_ci
    assert detect_language("Huset är rött med katten.") in ("sv", "no", "da", "unknown")
    assert detect_language("The cat sat on the mat.") == "en"
    assert simple_token_overlap("hello world", "hello world") == 1.0
    assert simple_token_overlap("hello world", "cat dog") == 0.0
    ci = proportion_ci(70, 100, n_boot=200, seed=0)
    assert 0.5 < ci["mean"] < 0.9
    print("  All inline metric tests passed.")

    # Summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    for name, m in [("C0_baseline", c0_metrics), ("C1_sysprompt", c1_metrics),
                     ("C3_edit", c3_metrics), ("C4_random", c4_metrics)]:
        print(f"  {name:<20} lang-ID: {m['lang_id_accuracy']:.2%}")
    print("\nAll code paths passed. Ready for Colab full run.")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
