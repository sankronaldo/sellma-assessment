"""
Main experiment script.

Runs all 7 conditions on the held-out test set, saves results, and prints summary.
Expected to run on Colab T4 (~30-60 min total) or locally with GPU.

Run with:
  python -I scripts/run_experiment.py [--smoke] [--alpha ALPHA] [--n-layers N]

Flags:
  --smoke     Run on 20 synthetic examples (for quick local verification)
  --alpha     Direction ablation scale (default: auto-tuned on dev, see dev_tune.py)
  --n-layers  Number of top layers to edit (default: 1)
"""

import argparse
import json
import random
import sys
import copy
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

RESULTS_DIR = ROOT / "results"
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Use synthetic smoke data instead of FLORES")
    p.add_argument("--alpha", type=float, default=None, help="Direction ablation scale")
    p.add_argument("--n-layers", type=int, default=1, help="Number of top layers to edit")
    p.add_argument("--n-cols", type=int, default=64, help="Columns per layer to edit")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--random-seeds", type=int, default=5, help="Seeds for C4 random control")
    return p.parse_args()


def load_or_build_data(smoke: bool = False):
    """Return (dev_sv, dev_en, test_sv, test_en, ppl_texts)."""
    if smoke:
        # Use tiny synthetic data (same as smoke_test.py)
        from scripts.smoke_test import make_tiny_dataset
        sv, en = make_tiny_dataset(n=20)
        return sv[:10], en[:10], sv[10:], en[10:], [r["source_text"] for r in sv[:20]]

    # Try to load saved splits first
    split_file = CONFIG_DIR / "data_split.json"
    if split_file.exists():
        from src.data_utils import load_splits, build_qa_prompts
        splits = load_splits()
    else:
        from src.data_utils import build_splits, build_qa_prompts
        splits = build_splits(dev_size=50, test_size=100, seed=SEED)

    from src.data_utils import build_qa_prompts
    dev_sv = build_qa_prompts(splits["dev_sv"], lang="sv")
    dev_en = build_qa_prompts(splits["dev_en"], lang="en")
    test_sv = build_qa_prompts(splits["test_sv"], lang="sv")
    test_en = build_qa_prompts(splits["test_en"], lang="en")
    ppl_texts = [s["text"] for s in splits["perplexity_sv"]]
    return dev_sv, dev_en, test_sv, test_en, ppl_texts


def dev_tune_alpha(
    model, tokenizer, dev_sv, dev_en, directions, target_layers, n_cols, device
):
    """Grid-search alpha on dev set language-ID accuracy. Returns best alpha."""
    from src.edit_utils import apply_direction_ablation
    from src.eval_pipeline import run_condition, evaluate_records

    best_alpha = 1.0
    best_acc = -1.0
    for alpha in [0.5, 1.0, 2.0]:
        edited = apply_direction_ablation(
            model, directions, target_layers=target_layers, alpha=alpha, n_columns=n_cols
        )
        records = run_condition(
            edited, tokenizer, dev_sv[:20], f"dev_tune_a{alpha:.1f}",
            max_new_tokens=60, device=device, out_dir=RESULTS_DIR / "dev_tune"
        )
        metrics = evaluate_records(records, expected_lang="sv")
        acc = metrics["lang_id_accuracy"]
        print(f"  alpha={alpha:.1f}: dev lang-ID = {acc:.2%}")
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
    print(f"  Best alpha: {best_alpha} (dev lang-ID={best_acc:.2%})")
    return best_alpha


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load model
    print("\n[1] Loading model...")
    from src.model_utils import load_model_and_tokenizer
    dtype = torch.float16 if device == "cuda" else torch.float32
    model, tokenizer = load_model_and_tokenizer(device=device, dtype=dtype)

    # 2. Load data
    print("\n[2] Loading data...")
    dev_sv, dev_en, test_sv, test_en, ppl_texts = load_or_build_data(smoke=args.smoke)
    print(f"  dev_sv={len(dev_sv)}, dev_en={len(dev_en)}, test_sv={len(test_sv)}, test_en={len(test_en)}")

    # 3. Compute language direction (feedback-based — declared)
    print("\n[3] Computing language direction (FEEDBACK-BASED step)...")
    from src.edit_utils import compute_language_direction, layer_direction_scores, apply_direction_ablation
    from src.edit_utils import apply_random_direction_edit, apply_sign_flipped_edit, count_edited_params

    sv_texts = [p["prompt"] for p in dev_sv]
    en_texts = [p["prompt"] for p in dev_en]
    directions = compute_language_direction(model, tokenizer, sv_texts, en_texts, device=device)
    scores = layer_direction_scores(directions, model)
    print(f"  Top-5 layers by alignment: {scores[:5]}")

    # Choose target layers
    top_layers = [scores[i][0] for i in range(min(args.n_layers, len(scores)))]
    print(f"  Target layers: {top_layers}")

    # 4. Tune alpha on dev (unless specified)
    alpha = args.alpha
    if alpha is None:
        print("\n[4] Tuning alpha on dev set...")
        alpha = dev_tune_alpha(
            model, tokenizer, dev_sv, dev_en, directions, top_layers, args.n_cols, device
        )
    else:
        print(f"\n[4] Using pre-specified alpha={alpha}")

    # Save edit config
    edit_cfg = {
        "target_layers": top_layers,
        "alpha": alpha,
        "n_cols": args.n_cols,
        "seed": SEED,
        "layer_scores": scores[:10],
    }
    with open(CONFIG_DIR / "edit_config.json", "w") as f:
        json.dump(edit_cfg, f, indent=2)

    # 5. Build all condition models
    print("\n[5] Building condition models...")

    # C3: targeted edit
    c3_model = apply_direction_ablation(
        model, directions, target_layers=top_layers, alpha=alpha, n_columns=args.n_cols
    )
    edit_info = count_edited_params(model, c3_model)
    print(f"  C3 edit: {edit_info['changed_params']:,} params changed ({edit_info['fraction_changed']:.4%})")
    with open(RESULTS_DIR / "edit_info.json", "w") as f:
        json.dump(edit_info, f, indent=2)

    # C4: random-direction controls (5 seeds)
    c4_models = []
    for seed in range(args.random_seeds):
        c4_models.append(apply_random_direction_edit(
            model, directions[top_layers[0]], target_layers=top_layers,
            alpha=alpha, n_columns=args.n_cols, seed=seed
        ))
    print(f"  C4 random controls: {len(c4_models)} seeds")

    # C5: sign-flipped
    c5_model = apply_sign_flipped_edit(
        model, directions, target_layers=top_layers, alpha=alpha, n_columns=args.n_cols
    )

    # C6: LoRA baseline
    print("\n[6] Training LoRA baseline (C6)...")
    try:
        from src.lora_utils import train_lora_baseline
        c6_model = train_lora_baseline(
            copy.deepcopy(model), tokenizer, dev_sv,
            num_epochs=3, lr=2e-4, r=8, seed=SEED, device=device
        )
    except Exception as e:
        print(f"  LoRA training failed: {e}. Skipping C6.")
        c6_model = None

    # 7. Run all conditions on TEST SET (touched once, here)
    print("\n[7] Running all conditions on TEST SET (touched once)...")
    from src.eval_pipeline import run_condition, evaluate_records
    from src.metrics import stratify_by_base_response

    sys_sv = "Svara alltid på svenska."
    sys_sv2 = "Du är en svensk assistent. Svara alltid på svenska och använd korrekt svensk grammatik."

    # --- Swedish prompt test ---
    test_out = RESULTS_DIR / "test_sv"
    test_out.mkdir(exist_ok=True)

    all_summary = {}

    # C0: baseline (no system prompt)
    c0 = run_condition(model, tokenizer, test_sv, "C0_baseline",
                       max_new_tokens=args.max_new_tokens, device=device, out_dir=test_out)
    all_summary["C0_baseline"] = evaluate_records(c0, "sv")

    # C1: Swedish system prompt
    c1 = run_condition(model, tokenizer, test_sv, "C1_sysprompt",
                       system_prompt=sys_sv,
                       max_new_tokens=args.max_new_tokens, device=device, out_dir=test_out)
    all_summary["C1_sysprompt"] = evaluate_records(c1, "sv")

    # C2: Stronger Swedish prompt
    c2 = run_condition(model, tokenizer, test_sv, "C2_strong_sysprompt",
                       system_prompt=sys_sv2,
                       max_new_tokens=args.max_new_tokens, device=device, out_dir=test_out)
    all_summary["C2_strong_sysprompt"] = evaluate_records(c2, "sv")

    # C3: targeted edit
    c3 = run_condition(c3_model, tokenizer, test_sv, "C3_direction_ablation",
                       max_new_tokens=args.max_new_tokens, device=device, out_dir=test_out)
    all_summary["C3_direction_ablation"] = evaluate_records(c3, "sv")

    # Stratified analysis (failure removal vs new capability)
    strata = stratify_by_base_response(c0, c3, expected_lang="sv")
    all_summary["C3_direction_ablation"]["stratum_A"] = strata.get("A", {})
    all_summary["C3_direction_ablation"]["stratum_B"] = strata.get("B", {})

    # C4: random controls
    c4_accs = []
    for seed_i, rand_m in enumerate(c4_models):
        c4_recs = run_condition(rand_m, tokenizer, test_sv, f"C4_random_s{seed_i}",
                                max_new_tokens=args.max_new_tokens, device=device, out_dir=test_out)
        c4_m = evaluate_records(c4_recs, "sv")
        c4_accs.append(c4_m["lang_id_accuracy"])
    import numpy as np
    all_summary["C4_random_control"] = {
        "lang_id_accuracy": float(np.mean(c4_accs)),
        "lang_id_std": float(np.std(c4_accs)),
        "n_seeds": len(c4_accs),
    }

    # C5: sign-flipped
    c5 = run_condition(c5_model, tokenizer, test_sv, "C5_sign_flipped",
                       max_new_tokens=args.max_new_tokens, device=device, out_dir=test_out)
    all_summary["C5_sign_flipped"] = evaluate_records(c5, "sv")

    # C6: LoRA
    if c6_model is not None:
        c6 = run_condition(c6_model, tokenizer, test_sv, "C6_lora",
                           max_new_tokens=args.max_new_tokens, device=device, out_dir=test_out)
        all_summary["C6_lora"] = evaluate_records(c6, "sv")

    # --- English regression test ---
    print("\n[8] English regression test...")
    test_en_out = RESULTS_DIR / "test_en"
    test_en_out.mkdir(exist_ok=True)
    for cond_name, m in [("C0_baseline", model), ("C3_direction_ablation", c3_model)]:
        en_recs = run_condition(m, tokenizer, test_en, f"{cond_name}_en",
                                max_new_tokens=args.max_new_tokens, device=device, out_dir=test_en_out)
        en_metrics = evaluate_records(en_recs, "en")
        all_summary[cond_name]["english_lang_id"] = en_metrics["lang_id_accuracy"]

    # --- Perplexity regression ---
    print("\n[9] Perplexity regression...")
    from src.metrics import compute_perplexity
    for cond_name, m in [("C0_baseline", model), ("C3_direction_ablation", c3_model)]:
        ppl = compute_perplexity(m, tokenizer, ppl_texts[:50], device=device)
        all_summary[cond_name]["perplexity_sv"] = ppl
        print(f"  {cond_name} PPL_sv: {ppl:.2f}")

    # 10. Save full summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nFull summary saved to {summary_path}")

    # 11. Save example outputs (sample of successes and failures)
    from src.metrics import detect_language
    examples_dir = RESULTS_DIR / "examples"
    examples_dir.mkdir(exist_ok=True)
    for cond_name, records in [("C0_baseline", c0), ("C3_direction_ablation", c3)]:
        successes = [r for r in records if detect_language(r.get("response","")) in ("sv","no","da")]
        failures = [r for r in records if detect_language(r.get("response","")) == "en"]
        import jsonlines
        with jsonlines.open(str(examples_dir / f"{cond_name}_successes.jsonl"), "w") as w:
            w.write_all(successes[:20])
        with jsonlines.open(str(examples_dir / f"{cond_name}_failures.jsonl"), "w") as w:
            w.write_all(failures[:20])
    print(f"Example outputs saved to {examples_dir}")

    # Print table
    from src.eval_pipeline import print_summary_table
    print_summary_table(all_summary)

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
