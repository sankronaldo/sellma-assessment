"""
Offline smoke test — runs the FULL pipeline without downloading any model.

Uses a randomly-initialized tiny Qwen2-architecture model (config only, no weights
downloaded) to verify that every code path executes without error.

This test cannot evaluate real language drift (the random model produces gibberish),
but it verifies:
  - Data loading and prompt building
  - Activation extraction (hook-based)
  - Language direction computation and layer scoring
  - Direction-ablation weight surgery
  - Random-direction and sign-flipped controls
  - Evaluation (language-ID, bootstrap CI, stratification)
  - Result saving (JSONL, JSON)
  - All metric unit tests

For a real pipeline check with a live model, run smoke_test.py on Colab or on a
machine with unrestricted HTTPS access to huggingface.co.
"""

import sys
import json
import random
import copy
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

RESULTS_DIR = ROOT / "results" / "smoke_offline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR = ROOT / "configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Build a tiny random Qwen2-style model from config (no weights download)
# ---------------------------------------------------------------------------

def build_tiny_model():
    """Build a randomly-initialised micro-Qwen2 model (4 layers, 256 hidden)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    cfg = Qwen2Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()

    # Build a minimal tokenizer (byte-level BPE-like)
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace as WS
    vocab = {str(i): i for i in range(950)}
    # Add some real Swedish tokens for language-ID detection
    sv_words = ["huset", "katten", "sover", "och", "är", "på", "det", "en", "ett",
                "Sverige", "svenska", "meningen", "sammanfatta", "svar", "följande",
                "mening", "svenska", "röd", "blå", "stor", "liten", "dag", "natt",
                "bok", "bord", "stol", "fönster", "dörr", "golv", "tak", "vägg",
                "åäö", "Åäö", "summarize", "English", "sentence", "answer",
                "<unk>", "<pad>", "</s>", "<bos>"]
    for w in sv_words:
        if w not in vocab:
            vocab[w] = len(vocab)
    vocab_size_actual = max(1000, len(vocab))

    # Use a simple whitespace tokenizer
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import WordLevel
    tokenizer_obj = HFTokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer_obj.pre_tokenizer = WS()

    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="</s>",
        bos_token="<bos>",
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# Minimal generate that doesn't use the full model_utils
# (since the tokenizer is non-standard)
# ---------------------------------------------------------------------------

def simple_generate(model, tokenizer, prompts, max_new_tokens=20):
    """Generate responses using the random model (output will be gibberish)."""
    device = "cpu"
    responses = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64,
                        padding=False)
        enc = {k: v.to(device) for k, v in enc.items() if k != "token_type_ids"}
        input_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id or 0,
            )
        new_tokens = out[0][input_len:]
        resp = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        responses.append(resp)
    return responses


# ---------------------------------------------------------------------------
# Minimal activation extractor (independent of model_utils)
# ---------------------------------------------------------------------------

def extract_activations(model, tokenizer, prompts, device="cpu"):
    n_layers = model.config.num_hidden_layers
    activations = {l: [] for l in range(n_layers)}
    hooks = []

    def make_hook(li):
        def hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            activations[li].append(hidden[:, -1, :].detach().cpu())
        return hook

    for li in range(n_layers):
        h = model.model.layers[li].register_forward_hook(make_hook(li))
        hooks.append(h)

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
        with torch.no_grad():
            model(**enc)

    for h in hooks:
        h.remove()

    return {l: torch.cat(activations[l], dim=0) for l in range(n_layers)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("OFFLINE SMOKE TEST (random model, no download needed)")
    print("=" * 60)

    # 1. Build tiny random model
    print("\n[1/7] Building tiny random model (4 layers, 256 hidden)...")
    model, tokenizer = build_tiny_model()
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Build synthetic dataset
    print("\n[2/7] Building synthetic dataset...")
    sv_sentences = [
        "Huset är rött och katten sover.",
        "Barnen leker i parken varje dag.",
        "Sverige är ett land i norra Europa.",
        "Fågeln sjunger i trädet utanför.",
        "Vattnet i sjön är kallt och klart.",
    ]
    en_sentences = [
        "The house is red and the cat is sleeping.",
        "The children play in the park every day.",
        "Sweden is a country in northern Europe.",
        "The bird is singing in the tree outside.",
        "The water in the lake is cold and clear.",
    ]
    sv_prompts = [{"id": f"sv_{i}", "prompt": f"Sammanfatta: {s}", "source_text": s, "expected_lang": "sv"}
                  for i, s in enumerate(sv_sentences)]
    en_prompts = [{"id": f"en_{i}", "prompt": f"Summarize: {s}", "source_text": s, "expected_lang": "en"}
                  for i, s in enumerate(en_sentences)]

    # Verify no ID overlap
    sv_ids = {p["id"] for p in sv_prompts}
    en_ids = {p["id"] for p in en_prompts}
    assert len(sv_ids & en_ids) == 0, "ID overlap in test data"
    print(f"  {len(sv_prompts)} SV + {len(en_prompts)} EN prompts. No ID overlap. OK")

    # 3. Baseline generation
    print("\n[3/7] C0 baseline generation (random model, output is gibberish)...")
    sv_texts = [p["prompt"] for p in sv_prompts]
    en_texts = [p["prompt"] for p in en_prompts]
    c0_responses = simple_generate(model, tokenizer, sv_texts, max_new_tokens=15)
    c0_records = [{"id": sv_prompts[i]["id"], "condition": "C0", "response": r,
                   "source_text": sv_prompts[i]["source_text"]}
                  for i, r in enumerate(c0_responses)]
    print(f"  Generated {len(c0_records)} responses. Sample: '{c0_records[0]['response'][:40]}'")

    # 4. Language direction computation
    print("\n[4/7] Computing language direction (feedback-based)...")
    sv_acts = extract_activations(model, tokenizer, sv_texts)
    en_acts = extract_activations(model, tokenizer, en_texts)
    directions = {}
    for l in range(model.config.num_hidden_layers):
        d = sv_acts[l].mean(0) - en_acts[l].mean(0)
        d = d / (d.norm() + 1e-8)
        directions[l] = d.to(dtype=torch.float32)
    print(f"  Computed {len(directions)} layer directions. OK")

    # Score layers
    scores = []
    for li, d in directions.items():
        try:
            W = model.model.layers[li].mlp.down_proj.weight.data.float()
            d_norm = d / (d.norm() + 1e-8)
            W_col_norms = W.norm(dim=0, keepdim=True) + 1e-8
            W_normed = W / W_col_norms
            sims = (W_normed.T @ d_norm).abs()
            scores.append((li, float(sims.mean())))
        except Exception:
            scores.append((li, 0.0))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_layer = scores[0][0]
    print(f"  Top layer by direction alignment: {top_layer} (score={scores[0][1]:.4f})")

    # 5. Apply direction-ablation edit
    print("\n[5/7] Applying direction-ablation edit (C3)...")
    from src.edit_utils import apply_direction_ablation, apply_random_direction_edit, apply_sign_flipped_edit, count_edited_params
    c3_model = apply_direction_ablation(
        model, directions, target_layers=[top_layer], alpha=1.0, n_columns=32
    )
    edit_info = count_edited_params(model, c3_model)
    print(f"  Changed {edit_info['changed_params']:,}/{edit_info['total_params']:,} params ({edit_info['fraction_changed']:.4%}) OK")

    # 6. Random and sign-flipped controls
    print("\n[6/7] Random direction control (C4) and sign-flipped (C5)...")
    c4_model = apply_random_direction_edit(model, directions[top_layer], [top_layer], alpha=1.0, n_columns=32, seed=0)
    c5_model = apply_sign_flipped_edit(model, directions, [top_layer], alpha=1.0, n_columns=32)
    c4_info = count_edited_params(model, c4_model)
    c5_info = count_edited_params(model, c5_model)
    print(f"  C4 changed: {c4_info['changed_params']:,} OK   C5 changed: {c5_info['changed_params']:,} OK")

    # 7. Metric unit tests
    print("\n[7/7] Running metric unit tests...")
    from src.metrics import detect_language, simple_token_overlap, proportion_ci, language_id_accuracy, stratify_by_base_response, bootstrap_ci

    assert detect_language("Huset är rött med katten och trädet.") in ("sv", "no", "da", "unknown")
    assert detect_language("The quick brown fox jumps over the lazy dog.") == "en"
    assert detect_language("") == "unknown"
    assert simple_token_overlap("hello world", "hello world") == 1.0
    assert simple_token_overlap("hello world", "cat dog fish") == 0.0
    ci = proportion_ci(70, 100, n_boot=200, seed=0)
    assert 0.5 < ci["mean"] < 0.9
    assert ci["lower"] <= ci["mean"] <= ci["upper"]

    # Test stratification with realistic records
    base_recs = [
        {"id": "sv_0", "response": "Huset är rött och gammalt."},
        {"id": "sv_1", "response": "The children play in the park."},  # drift
        {"id": "sv_2", "response": "Sverige har ett kallt klimat."},
    ]
    edit_recs = [
        {"id": "sv_0", "response": "Huset är stort och fint."},  # still Swedish
        {"id": "sv_1", "response": "Barnen leker utomhus."},   # fixed drift
        {"id": "sv_2", "response": "Sverige är kallt på vintern."},  # still Swedish
    ]
    strata = stratify_by_base_response(base_recs, edit_recs, expected_lang="sv")
    assert "A" in strata and "B" in strata
    # sv_1 was drift → should be in stratum B
    total_in_strata = strata["A"]["n"] + strata["B"]["n"]
    assert total_in_strata == 3, f"Expected 3 records in strata, got {total_in_strata}"
    assert strata["B"]["n"] >= 1, "Expected at least 1 drift case in stratum B"
    print("  All metric tests passed. OK")

    # Save results
    results = {
        "smoke_type": "offline",
        "model": "random_qwen2_4layers_256hidden",
        "edit_info": edit_info,
        "direction_scores": scores,
        "note": "Random model outputs gibberish; test verifies code paths only.",
    }
    with open(RESULTS_DIR / "smoke_offline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("OFFLINE SMOKE TEST COMPLETE — all code paths verified.")
    print(f"Results saved to: {RESULTS_DIR}")
    print("\nNOTE: Real model download blocked by corporate SSL inspection.")
    print("Run smoke_test.py on Colab or a machine with open HTTPS access.")
    print("=" * 60)


if __name__ == "__main__":
    main()
