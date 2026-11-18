---
title: "Weight-Space Programming for Swedish Language Models: A Directed Investigation"
subtitle: "SE.LLMA Take-Home Assignment"
date: "2026-11-18"
geometry: margin=2.5cm
fontsize: 11pt
---

# Abstract

We critically evaluate the "Sparks of Weight-Space Programming" blog post and investigate
whether its key technique — targeted weight editing to remove model failure modes — can
reduce Swedish language drift in a small multilingual language model. We pre-register a
hypothesis and design a controlled experiment with seven conditions, including prompt-only
controls, a random-direction control, and a LoRA fine-tuning baseline. The core scientific
question is whether any observed improvement reflects failure removal (the model could
already answer correctly, but was prevented by producing output in English) or genuinely
new Swedish capability. We detail which parts of the method are direct weight surgery and
which require model feedback, clearly distinguishing ourselves from the paper's claim of
pure weight-space reasoning.

---

*(This report expands on RESEARCH_NOTE.md. All results sections are placeholders to be
filled after the Colab run. See RESEARCH_NOTE.md for the complete write-up structure.)*

---

# 1. Critical Reading

## 1.1 What the paper claims and what it demonstrates

[*Expand RESEARCH_NOTE.md §1 here with more detail on each experiment.*]

The paper is a blog post, not peer-reviewed. It presents evidence of "sparks" of an
emerging capability, explicitly not a proposed method. The key claim is that a closed
frontier LLM ("Astra", inaccessible to outside researchers) can read and write neural
network weights without gradient training. The experiments span a wide range: toy MLPs
(Exps 1–5), a Go-playing network (Exp 6), a sycophancy detection task (Exp 7), and a
production LLM (Exp 8).

## 1.2 The feedback loop problem

The critical distinction the paper does not cleanly draw is between:

- **Direct weight reasoning**: the LLM reads weight tensors and produces an edit in one
  shot, without running the network.
- **Feedback optimization**: the LLM runs the network, sees results, and iterates.

Experiments 1–5 are closer to pure weight reasoning (small networks, one shot). Exp 6 and
Exp 8 use explicit forward-pass feedback loops (up to 50 passes). Exp 8 is the headline
result. Its 50-pass condition, the most dramatic (+40 pp MATH-500), is unambiguously
feedback-based optimization. "No gradients" is accurate but misleading; iterative
evaluation-and-edit without gradients is still feedback search.

## 1.3 Experiment 8 specifically

The base model loops on 93% of MATH-500 answers under the stated settings. Any edit that
stops looping will appear to produce a large accuracy improvement, because the previous
outputs were not answers at all. The authors' own conclusion — that the edit removed a
failure, not added math ability — is well-supported by the Gemma 270M null result
(a model that rarely loops showed no improvement). What is not established is whether
the specific direction found by Astra is better than a random edit, a standard
repetition-penalty fix, or a LoRA on the same practice problems.

## 1.4 Summary assessment

| Claim | Assessment |
|---|---|
| LLM can read small network weights to identify behavior | Supported (Exps 1–3, with caveats) |
| One-shot edit can fix a stuck reinforcement learning policy | Supported (Exp 4, single case) |
| Qwen 0.8B math improvement via 50-forward-pass edit | Demonstrated, but: feedback search, no controls |
| Weight-editing is better than inference fixes or LoRA | Not established |
| Results transfer to Swedish or production models | Not established |

---

# 2. Hypothesis

[*Same as RESEARCH_NOTE.md §2 — repeat for self-contained report.*]

**H**: In Qwen2.5-0.5B-Instruct, language drift (responding in English to Swedish prompts)
can be reduced by a targeted weight edit derived from activation analysis. Any improvement
will be explained by failure removal (Stratum B: C0 drifted to English) rather than new
capability (Stratum A: C0 already responded in Swedish).

This is directly relevant to SE.LLMA because language drift is an operational failure mode
in multilingual model adaptation. Knowing whether targeted weight edits add value over
prompting, and whether gains represent failure removal or new capability, informs the
project's investment in interpretability-driven approaches versus standard fine-tuning.

---

# 3. Experimental Design

[*Expand RESEARCH_NOTE.md §3 with more technical detail.*]

## 3.1 Model

`Qwen/Qwen2.5-0.5B-Instruct` (Apache 2.0). Exact revision pinned in
`configs/model_config.json`. Loaded in fp16 on T4 GPU in Colab.

## 3.2 Data

FLORES-200 (CC-BY-SA 4.0) Swedish and English sentences. Dev: 50 per language.
Test: 100 per language. Splits fixed with `seed=42` and saved before any model runs.
Swedish Wikipedia perplexity corpus: 200 sentences.

Leakage check: FLORES dev split and devtest split are disjoint by construction; verified
programmatically in `src/data_utils.py::build_splits()`.

## 3.3 Edit method

The direction-finding step is feedback-based: 50+50 model runs on dev prompts. The mean
activation difference (Swedish - English) is computed at each transformer layer. The layer
whose MLP down-projection columns most align with the direction is selected. Alpha (scale)
is grid-searched on dev language-ID accuracy over {0.5, 1.0, 2.0}. The edit applies:

```
W[:,j] -= alpha * d/||d||   for top-64 columns j by alignment score
```

This modifies <<1% of parameters, once, with no feedback.

## 3.4 Controls

All controls use identical decoding and the same test set. Random-direction control uses
5 seeds to report mean ± SD and assess whether targeted direction matters.

---

# 4. Results

[*Placeholder — to be filled from results/summary.json after Colab run.*]

## 4.1 Primary metric: Swedish language-ID accuracy

[Insert table from `results/metrics_table.csv`]

## 4.2 Failure removal analysis

[Insert stratification table and `results/figures/stratification.png`]

## 4.3 Regression metrics

[Insert English lang-ID and Swedish perplexity by condition]

## 4.4 Edit accounting

[Insert from `results/edit_info.json` and `configs/edit_config.json`]

---

# 5. Interpretation

[*To be written after results. See RESEARCH_NOTE.md §5 for the pre-registered
interpretation structure.*]

---

# 6. Limitations

[*Full list in RESEARCH_NOTE.md §6. Key additional points for the report:*]

- The activation-direction method is **not** pure weight-space reasoning. It is
  activation steering with a one-shot weight-surgery implementation. The WSP paper's
  strongest evidence (Exps 1–3) involves directly computing properties of weight
  matrices without model runs. This experiment requires model runs to find the direction.

- Language drift is a coarse failure mode. At Qwen2.5-0.5B scale, the model may already
  respond in Swedish with a system prompt, making the weight edit unnecessary. At larger
  or less instruction-tuned scales, the finding may differ.

- Swedish-specific linguistics (grammatical gender agreement, definiteness morphology,
  V2 word order, Swedish character set) are not evaluated in this experiment. Language-
  of-response is a necessary but not sufficient criterion for Swedish language quality.

---

# 7. Recommendation

[*To be written after results. See RESEARCH_NOTE.md §7.*]

---

# Appendix A: Deviation Log

No deviations from the pre-registered plan recorded. See `PLAN.md` §13.

# Appendix B: File Manifest

| File | Contents |
|---|---|
| `PLAN.md` | Pre-registered plan |
| `RESEARCH_NOTE.md` | Main 1-2 page write-up |
| `report/report_source.md` | This file (builds to report.pdf) |
| `configs/data_split.json` | Fixed dev/test split |
| `configs/model_config.json` | Pinned model revision |
| `configs/edit_config.json` | Chosen alpha and target layers |
| `results/summary.json` | Per-condition metrics |
| `results/metrics_table.csv` | Tabular results |
| `results/test_sv/*.jsonl` | Per-condition Swedish test outputs |
| `results/test_en/*.jsonl` | Per-condition English test outputs |
| `results/examples/` | Sampled successes and failures |
| `results/figures/` | PNG figures |
| `results/edit_info.json` | Edit parameter counts |
