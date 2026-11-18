# SE.LLMA Assignment: Weight-Space Programming for Swedish LMs

**Hypothesis**: In Qwen2.5-0.5B-Instruct, language drift (responding in English to
Swedish prompts) can be reduced by targeted weight surgery derived from activation
analysis, but any improvement reflects failure removal rather than new Swedish capability.

This is a submission for the SE.LLMA take-home assignment on Weight-Space Programming.
See [`RESEARCH_NOTE.md`](RESEARCH_NOTE.md) for the main write-up and
[`PLAN.md`](PLAN.md) for the pre-registered experimental plan.

---

## Results summary

All 7 conditions run on 99 held-out Swedish test examples (Qwen2.5-0.5B-Instruct, fp16, greedy decoding):

| Condition | Swedish lang-ID% | 95% CI | PPL_sv | English% |
|---|---|---|---|---|
| C0 baseline (no prompt) | 97.98% | [94.9, 100] | 541.7 | 100.0% |
| C1 system prompt | **100.00%** | [100, 100] | — | — |
| C2 strong system prompt | 98.99% | [97.0, 100] | — | — |
| C3 direction ablation (edit) | 96.97% | [92.9, 100] | 910.3 | 97.98% |
| C4 random direction (5 seeds) | 97.78% ± 0.99% | — | — | — |
| C5 sign-flipped edit | 90.91% | [84.8, 96.0] | — | — |
| C6 LoRA (2 epochs, r=8) | **100.00%** | [100, 100] | — | — |

**Key finding**: the direction-ablation edit (C3) is worse than the unedited baseline,
worse than random edits (C4), and causes a 68% perplexity regression (541→910). A
system prompt (C1) achieves 100% with no side effects. See `RESEARCH_NOTE.md` for the
full interpretation.

---

## Reproduce

### 1. Clone and install

```bash
git clone https://github.com/sankronaldo/sellma-assessment.git
cd sellma-assessment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate.bat      # Windows
pip install -r requirements.txt
```

### 2. Download the model

The experiment uses `Qwen/Qwen2.5-0.5B-Instruct` (Apache 2.0). Clone it via ModelScope
(no account required):

```bash
git clone https://modelscope.cn/qwen/Qwen2.5-0.5B-Instruct.git models/Qwen2.5-0.5B-Instruct
```

### 3. Run unit tests

```bash
python -m pytest tests/ -v
# Expected: 19 passed
```

### 4. Run offline smoke test (no model needed)

```bash
python -I scripts/smoke_test_offline.py
```

### 5. Run the full experiment

```bash
python -I scripts/run_local.py
# Requires the model downloaded in step 2.
# Expected runtime: ~20 min on a mid-range GPU (RTX 2050 used here).
# Results saved to results/summary.json
```

### 6. Regenerate figures and CSV

```bash
python -I scripts/analyze_results.py
# Writes results/figures/*.png and results/metrics_table.csv
```

All seeds are fixed (`SEED = 42`; random-direction control seeds 0–4).
Edit config is saved to `configs/edit_config.json`. Data split is in `configs/data_split.json`.

---

## Repository layout

```
RESEARCH_NOTE.md            # main write-up (~2 pages)
report/report.pdf           # PDF version
PLAN.md                     # pre-registered plan (written before any test-set evaluation)
DATA_AND_MODELS.md          # licenses and model sources
AI_USAGE.md                 # AI tools used and their contributions
requirements.txt            # pinned dependencies

src/
  metrics.py                # language-ID, perplexity, bootstrap CI, stratification
  edit_utils.py             # direction computation, ablation, controls
  model_utils.py            # model loading and generation
  data_utils.py             # splits and prompt building
  eval_pipeline.py          # run conditions, save results
  lora_utils.py             # LoRA baseline (C6)

scripts/
  run_local.py              # main experiment (reproduces results/summary.json)
  analyze_results.py        # figures and CSV from summary.json
  smoke_test_offline.py     # offline pipeline check (random model, no download)

tests/
  test_metrics.py           # 15 unit tests for metric functions
  test_splits.py            # 4 unit tests for data utilities

configs/                    # edit config, data split, model revision
results/                    # summary.json, metrics CSV, figures, per-condition JSONL
```

---

## Hardware used

RTX 2050 (4 GB VRAM), Windows, fp16. Total experiment runtime: ~25 minutes.
