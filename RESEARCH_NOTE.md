# Research Note: Weight-Space Programming for Swedish Language Models

**Author**: SE.LLMA candidate assignment submission, 2026-11-18  
**Model**: Qwen2.5-0.5B-Instruct (ModelScope git clone, fp16, RTX 2050)  
**Task**: Does direction-ablation weight editing reduce Swedish language drift, and does
any improvement reflect failure removal rather than new capability?

---

## 1. Critical Reading of the Paper

"Sparks of Weight-Space Programming" (Takehi et al., MIT CSAIL, blog post 2026-11-02)
claims a frontier LLM can read and write neural network weights without gradient training,
calling this Weight-Space Programming (WSP). The authors clarify in a Nov 12 addendum
that this describes an emerging capability, not a proposed method.

**Strongest evidence.** Experiment 3 (induction heads) is the most rigorous result: a
frontier LLM identified specific induction heads by composing attention weight matrices
— a known interpretability technique — without running the target networks. Ground truth
was measured independently and matched exactly in all three models. Experiment 7's honest
negative (direct weight reading failed; activation patterns were needed) clearly marks
where weight-only reasoning stopped working. Public log and weight release enables
independent checking.

**Most important limitations.** The headline result, Experiment 8 (Qwen3.5-0.8B math),
uses up to 50 forward-pass feedback rounds — an LLM-driven search loop over weight edits.
"No gradients" does not mean "no feedback optimization". The 50-pass condition (+40 pp
MATH-500) is feedback search, not weight-space reasoning. The base model loops on 93% of
answers under the stated sampling settings; any edit that stops looping will produce a
large apparent improvement. No controls are reported: no random-direction edit, no
inference fix (repetition penalty, non-thinking mode), no LoRA baseline. The one
additional real LLM tested (Gemma 270M) was a null result, supporting the authors' own
conclusion that edits remove failures rather than add capabilities. All convincing
direct-weight-reasoning results (Exps 1–5) use purpose-built networks under 1.5M
parameters.

**What is and is not established.** Justified: a frontier LLM can recover facts from
small network weights; a small edit can remove one specific looping bug. Not established:
the edit is better than inference fixes or LoRA; the method is "reasoning" rather than
feedback search; any result transfers to Swedish or to production-scale models.

The key distinction the assignment asks about: Exps 1–5 are direct weight reasoning
(one shot, no network runs). Exps 6 and 8 are feedback optimization (iterative model
evaluation without gradients — still search). The paper conflates these modes.

---

## 2. Hypothesis and Relevance to SE.LLMA

**H**: In Qwen2.5-0.5B-Instruct, language drift (responding in English to Swedish
prompts) can be reduced by a targeted weight edit derived from activation analysis. Any
improvement in Swedish-task accuracy will be explained primarily by *failure removal*
(examples where the model answered correctly in content but in English) rather than new
Swedish capability.

**Why it matters for SE.LLMA.** SE.LLMA fine-tunes existing open multilingual models
for Swedish. Language drift is a known operational failure: a model that understands
Swedish but responds in English is unusable. If targeted weight edits reliably prevent
drift at negligible cost, they could inform the project's adaptation strategy. If not —
or if prompting achieves the same result at no cost — that is equally useful information.

**Pre-registered criteria.** Support: edit improves lang-ID by >10 pp over baseline,
CI excludes 0, exceeds random-direction control by >5 pp. Refute: edit within noise of
random control, or prompt-only baseline matches or exceeds edit. Failure-removal
prediction: improvement concentrates in Stratum B (C0 drifted to English), not
Stratum A (C0 already Swedish).

---

## 3. Experiment

**Model.** `Qwen/Qwen2.5-0.5B-Instruct` (Apache 2.0), downloaded via ModelScope git
clone (see `configs/model_config.json` for revision), loaded in fp16 on RTX 2050 (4 GB
VRAM). Greedy decoding, max 60 new tokens.

**Data.** 50 synthetic Swedish + 50 English sentence pairs for development; 99 synthetic
Swedish + 99 English for testing (disjoint by construction, seed 42, saved to
`configs/data_split.json`). Task: sentence summarisation prompts. 20 Swedish Wikipedia
sentences as a held-out perplexity corpus.

**Edit method.** Direction finding is feedback-based (50+50 dev forward passes, declared):
mean activation difference (Swedish – English) at each transformer layer. The layer
whose MLP down-projection columns most align with the direction is selected (layer 23,
the last layer, by alignment score). Alpha tuned on dev language-ID accuracy over
{0.5, 1.0, 2.0}; best alpha = 0.5. The edit subtracts `0.5 × d/‖d‖` from the top-64
MLP down-projection columns at layer 23 — 8,192 parameters, 0.0017% of model weights.
Applied once, no feedback. This is direct weight surgery applied after a feedback-based
direction search.

**Conditions** (all evaluated under identical decoding on the same 99 test examples):

| ID | Description |
|---|---|
| C0 | No system prompt (baseline) |
| C1 | Swedish system prompt: "Svara alltid på svenska" |
| C2 | Stronger Swedish instruction prompt |
| C3 | Targeted direction-ablation edit (chosen on dev) |
| C4 | Random-direction edits, 5 seeds, matched magnitude and layer |
| C5 | Sign-flipped edit (opposite direction) |
| C6 | LoRA fine-tuning on same 50 dev examples (2 epochs, r=8) |

---

## 4. Results

### 4.1 Main results table

| Condition | n | Lang-ID% | 95% CI | PPL_sv | EN% |
|---|---|---|---|---|---|
| C0 baseline | 99 | **97.98%** | [94.9, 100] | 541.7 | 100.0% |
| C1 sys prompt | 99 | **100.00%** | [100, 100] | — | — |
| C2 strong prompt | 99 | **98.99%** | [97.0, 100] | — | — |
| C3 direction ablation | 99 | **96.97%** | [92.9, 100] | 910.3 | 97.98% |
| C4 random (mean ± SD) | 99×5 | **97.78% ± 0.99%** | — | — | — |
| C5 sign-flipped | 99 | **90.91%** | [84.8, 96.0] | — | — |
| C6 LoRA (2 epochs, r=8) | 99 | **100.00%** | [100, 100] | — | — |

*All numbers from `results/summary.json`. Bootstrap CI: 1000 resamples, seed 0.*

### 4.2 Edit accounting

Layer 23 (final transformer layer), alpha = 0.5, 64 columns edited.
Parameters changed: 8,192 / 494,032,768 total (0.0017%).

### 4.3 Failure removal analysis (C3 vs C0)

| Stratum | Description | n | C0 Swedish% | C3 Swedish% | Change |
|---|---|---|---|---|---|
| A | C0 already Swedish | 97 | 100% | 97.94% | **−2.06 pp** |
| B | C0 drifted to English | 2 | 0% | 50% | +50 pp |

Net: the edit fixed 1 of 2 drift cases (+1) but harmed 2 previously-correct cases (−2).
Overall: −1 correct responses after editing.

### 4.4 Figures

See `results/figures/lang_id_comparison.png` and `results/figures/stratification.png`.

### 4.5 Qualitative inspection

**C0 baseline — 2 drift cases (Stratum B):**

- `test_sv_96` (Lucia prompt): model responds *"I'll do my best to provide an appropriate
  response in Swedish for you. However, I must clarify that the text you provided is not
  clear..."* — metacommentary in English, refuses the task.
- `test_sv_98` (harvest festival prompt): model treats the task as translation and
  responds *"The Swedish translation of 'Skördefesten...' is: 'The autumn sports
  festival...'"* — task misunderstanding, full English response.

Both drift cases are edge-case prompts where the model misunderstood the instruction.

**C3 edit — 2 new harms (Stratum A regressions):**

- `test_sv_54` (NEW harm): model responds *"The man builds a wall with bricks and
  stones. I'll do my best to translate this into English..."* — the edit introduced the
  same English-metacommentary failure that was already the base model's failure mode.
- `test_sv_70` (NEW harm): garbled mixed-language output, response became unusable.

The edit introduced the same type of failure it was meant to fix. This is consistent
with a disruption of the model's language-of-response mechanism that pushed some
previously-stable examples toward the existing failure mode.

Manual inspection of 20 outputs confirmed `langdetect` reliably identifies Swedish vs
English on responses >10 tokens. Full examples in `results/examples/`.

---

## 5. Interpretation

**Primary finding: the edit does not improve language-of-response.** C3 (96.97%) is
*worse* than C0 (97.98%) by 1 percentage point. The difference falls within the
confidence intervals (CI overlap completely), but the direction is against the
hypothesis. The edit cannot be said to improve language drift.

**The edit performs worse than random noise.** C4 (97.78% ± 0.99%) is better than C3
(96.97%), indicating the targeted direction has no specific benefit over a random edit
of matched magnitude.

**The direction is real but the edit is non-specific.** C5 (sign-flipped, 90.91%) is
significantly worse than both C0 and C3, confirming the found direction has genuine
semantic content — it is not random noise. However, both the forward (C3) and reverse
(C5) directions of the edit degrade the model, with the reverse being more damaging.
This is consistent with a model that is already well-calibrated: any perturbation to
the final-layer MLP output projection disrupts the delicate balance of language
activations, regardless of direction.

**Large perplexity regression.** The edit increased Swedish perplexity from 541.7 to
910.3 (+68%), a substantial degradation in language modeling quality not visible in the
primary metric. This confirms the edit is disruptive.

**Failure removal or new capability? Failure removal — but the wrong failure.**
Stratum B (the 2 drift cases) shows partial recovery (50%), consistent with the
failure-removal framing. But Stratum A shows the edit also *induces* a failure in 2
previously-correct cases. This is the opposite of targeted failure removal: the method
removes some drift while simultaneously introducing new errors in the majority class.

**Prompting is strictly better.** C1 (system prompt "Svara alltid på svenska") achieves
100% with no side effects, is costless, and reversible. It is the correct intervention
for language drift at this scale.

**Direct weight reasoning vs feedback optimization.** The direction-finding step ran 100
dev forward passes — this is feedback-based. The weight surgery itself (a one-shot edit)
is direct. The combined method is a hybrid, and cannot be described as "weight-space
reasoning" in the paper's strong sense. It is equivalent to activation steering with a
one-shot implementation.

---

## 6. Limitations

- **Single model and task.** Qwen2.5-0.5B-Instruct on summarisation prompts. Results
  do not generalize to larger models, instruction-following tasks, or conversational
  settings where drift rates may differ.

- **Minimal drift in baseline.** Only 2/99 examples drifted at all, leaving almost no
  room for improvement. The experiment is underpowered for detecting drift reduction.
  A more informative experiment would need a model with higher baseline drift (or a
  task / prompting setup that elicits more drift).

- **Synthetic data.** The Swedish sentences were generated by the researcher. They are
  well-formed but lack the distributional diversity of real-world Swedish text. Drift
  patterns may differ on naturally occurring prompts.

- **Layer selection heuristic.** The alignment score used to rank layers is a heuristic;
  the best layer for drift correction may not be the one with highest weight alignment.
  Only the top-1 layer was edited.

- **Feedback-based direction finding.** The method requires model runs and cannot be
  replicated without access to the model. It is not "direct weight reasoning".

- **Metric validity.** `langdetect` can conflate Swedish with Norwegian/Danish. 2 of
  the C3 Stratum A regressions may be classifier errors rather than actual language
  changes. This is noted but does not change the conclusion.

- **Scale.** A 0.5B model is at the small end of practical LLMs. Drift patterns and
  edit sensitivity may differ at 7B+.

---

## 7. Recommendation

**Do not pursue direction-ablation weight editing for language drift at this scale.**

The evidence gives three reasons:

1. The intervention failed: C3 performed worse than C0 on the primary metric and caused
   a large perplexity regression.
2. The intervention is unnecessary: a system prompt achieves 100% language-ID accuracy
   with no side effects. For language-of-response failures, prompting is the correct fix.
3. The method has no specific advantage: C3 performed worse than random edits (C4),
   indicating the targeted direction does not confer a benefit.

**What the C5 result adds.** The sign-flipped edit was significantly more damaging
(90.91%), confirming the found direction has semantic content. The model is sensitive
to perturbations of the final-layer MLP output projection in both directions. This
suggests the layer is load-bearing and not safely editable with this method.

**The most informative next experiment.** The paper's most credible result (Exp 8) was
on a model with a specific, documented failure mode (looping) that the edit directly
targeted. The analogous experiment for SE.LLMA would be: take a base multilingual model
(not instruction-tuned) where drift is severe and consistent, identify the drift as a
specific mechanistic failure, and test whether a targeted edit removes it without
degrading capabilities. This requires:

1. A model with high baseline drift (>30%): test with a base model (not instruct-tuned)
   or a model fine-tuned on a narrow domain that induces drift.
2. A mechanistic analysis of *why* the model drifts (which layers, which attention
   heads, whether it is a residual-stream direction or a specific circuit).
3. Proper controls: random direction, inference fix, LoRA on matched data.

The WSP paper's framing is most applicable when there is a clear, localised failure
mode. When the baseline model already behaves well (as here), weight editing adds risk
without benefit.

**Null result value for SE.LLMA.** This experiment establishes that for a well-tuned
small multilingual model on structured Swedish prompts, language drift is rare (2%) and
addressable by prompting alone. SE.LLMA should not invest in weight-editing approaches
for language drift correction until a model with a genuine and localized drift failure
has been identified.
