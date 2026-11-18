# Language Drift Direction Ablation — SE.LLMA Experiment

**Hypothesis**: In Qwen2.5-0.5B-Instruct, language drift (responding in English to
Swedish prompts) can be reduced by targeted weight surgery derived from activation
analysis, but the improvement represents failure removal rather than new Swedish
language capability.

This is a submission for the SE.LLMA take-home assignment on Weight-Space Programming.
See [`PLAN.md`](PLAN.md) for the pre-registered experimental plan and
[`RESEARCH_NOTE.md`](RESEARCH_NOTE.md) for the main write-up.

---

## Quick start

### 1. Clone and install

```bash
git clone <this repo>
cd sellma_experiment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate.bat    # Windows
pip install -r requirements.txt
```

### 2. Run unit tests

```bash
python -m pytest tests/ -v
# Expected: 19 passed
```

### 3. Offline code-path smoke test (no model download needed)

Verifies every pipeline component with a randomly-initialised micro-model.

```bash
python -I scripts/smoke_test_offline.py
```

### 4. Run the full experiment (requires a machine with HuggingFace access)

**Option A — Colab (recommended)**: Open `colab/run_experiment_colab.ipynb` in Google
Colab (Free T4 GPU). The notebook installs dependencies, downloads the model, runs all
7 conditions, and saves results to Google Drive.

Estimated Colab runtime: 30–60 minutes.

**Option B — local GPU**:

```bash
python -I scripts/run_experiment.py
```

**Option C — CPU smoke (20 synthetic examples)**:

```bash
python -I scripts/run_experiment.py --smoke
```

### 5. Analyse results and generate figures

```bash
python -I scripts/analyze_results.py
# Writes results/figures/*.png and results/metrics_table.csv
```

---

## Reproducing the main result

The single key number in the report is the Swedish language-ID accuracy for each
condition, as reported in `results/summary.json`. To reproduce:

1. Run the Colab notebook (cells 1–10) with `USE_DRIVE = True`.
2. After completion, download `results/summary.json` from Drive.
3. Run `scripts/analyze_results.py` to regenerate the figures and CSV.

All seeds are fixed (`SEED = 42`; random-direction control seeds 0–4).
Model revision is pinned in `configs/model_config.json` after the first download.
Dataset revision is pinned in `configs/data_split.json`.

---

## Repository layout

```
README.md                   # this file
RESEARCH_NOTE.md            # 1-2 page write-up (main deliverable)
report/report.pdf           # PDF built from the note
PLAN.md                     # pre-registered plan
DATA_AND_MODELS.md          # licenses, revisions, sources
AI_USAGE.md                 # AI tools and contributions
requirements.txt            # pinned dependencies

src/
  data_utils.py             # FLORES-200 loading, splits, prompt building
  metrics.py                # language-ID, perplexity, bootstrap CI, stratification
  model_utils.py            # model loading and generation
  edit_utils.py             # direction computation, ablation, controls
  eval_pipeline.py          # run conditions, save results
  lora_utils.py             # LoRA fine-tuning baseline (C6)

scripts/
  smoke_test.py             # CPU smoke test (requires model download)
  smoke_test_offline.py     # offline smoke test (random model, no download)
  run_experiment.py         # main experiment entry point
  analyze_results.py        # post-run analysis and figures

colab/
  run_experiment_colab.ipynb  # end-to-end Colab notebook

tests/
  test_metrics.py           # 15 tests for metrics functions
  test_splits.py            # 4 tests for data utilities

configs/                    # run configs (populated at runtime)
results/                    # raw outputs, metrics, figures (populated at runtime)
```

---

## Expected results (placeholder until Colab run completes)

| Condition | Description | Expected lang-ID |
|---|---|---|
| C0 baseline | No system prompt | Unknown (may show drift) |
| C1 sys prompt | "Svara alltid på svenska" | Higher than C0 |
| C2 strong prompt | Detailed Swedish instruction | At least as high as C1 |
| C3 direction ablation | Targeted weight edit | To be measured |
| C4 random direction | 5-seed control | Null hypothesis |
| C5 sign-flipped | Wrong direction | Should not improve |
| C6 LoRA | Fine-tuned on dev data | Conventional baseline |

The key scientific question is not which number is highest, but whether C3 improves
only on cases where C0 already drifted to English (failure removal), or whether it also
improves cases where C0 already responded in Swedish (new capability).

---

## Notes on SSL / corporate network

The model download from HuggingFace (`Qwen/Qwen2.5-0.5B-Instruct`) requires HTTPS
access to `huggingface.co`. On corporate networks with SSL inspection, this may fail
with an SSL certificate error. Use Colab or a machine with unrestricted HTTPS access.

The offline smoke test (`scripts/smoke_test_offline.py`) runs without any downloads.

---

## Hardware and expected runtime

| Environment | Expected runtime | Notes |
|---|---|---|
| Colab Free (T4) | 30–60 min | fp16, greedy decoding, max 100 tokens |
| Local GPU (comparable) | 20–40 min | |
| CPU only (--smoke flag) | 15–30 min | 20 synthetic examples |
| Offline smoke test | <1 min | No model, verifies code only |
