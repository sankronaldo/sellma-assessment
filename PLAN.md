# PLAN.md — Pre-registered Experimental Plan
*Written before any test-set evaluation. Do not edit after test-set results are obtained
except in the **Deviation log** section.*

---

## 1  Paper summary

**What the paper claims**
"Sparks of Weight-Space Programming" (Takehi et al., MIT CSAIL, blog post published
2026-11-02, addendum 2026-11-12) claims that a frontier LLM can *read* neural network
weights to infer what a network does, and *write or edit* weights to change behavior —
without gradient-based training by the LLM. The authors call this Weight-Space
Programming (WSP). The "programmer" is a closed frontier model ("GPT-6 Astra").

The addendum says WSP is "a concept and a statement about an emerging model capability,
not a method". The experiments are described as evidence of *sparks* of this capability,
not a proposed editing technique.

**Critical distinction: weight reasoning vs feedback optimization**

The paper conflates two very different operations:

| Mode | What happens | Experiments |
|---|---|---|
| **Direct weight reasoning** | The LLM reads weight tensors and produces a single edit without running the network | Exp 1–5 (all toy/small networks); Exp 7 partial |
| **Feedback optimization** | The LLM runs the network, sees results, and iterates — an LLM-driven search loop | Exp 6 (Go, with >0 forward passes), Exp 8 (Qwen math, 10/25/50 forward passes) |

Experiment 8 is the only experiment on a production-scale LLM and also the one with the most
dramatic result (+40 pp on MATH-500). It uses explicit forward-pass feedback loops. The
authors acknowledge this but argue there are "no gradients", which is accurate but
orthogonal: feedback-driven search without gradients is still feedback optimization, not
direct weight reasoning. This distinction is central to this investigation.

**Strongest evidence in the paper**
- Exp 3 (induction heads): ground truth is independently measurable, the LLM was not
  allowed to run the network, and the method it used (composing attention weight matrices)
  is a known interpretability technique. The match is exact in 3 of 3 models.
- Exp 7 honest negative: from weights alone the LLM failed; it succeeded only after
  seeing attention patterns from model runs. This is the clearest boundary between direct
  weight reading and behavior-derived analysis.
- Gemma 270M null: the model that rarely loops showed no improvement, supporting the
  authors' own framing that edits remove failures, not add capabilities.
- Public logs and edited weights enable independent checking.

**Most important limitations**
1. Feedback loop in Exp 8 is the headline result. The jump from 25→50 forward passes
   (+34 pp MATH-500) is unexplained and may be one trajectory with no seed variance
   reported.
2. The base model loops on 93% of MATH-500 answers. A large gain from loop removal is
   expected to look dramatic; it says nothing about math ability.
3. Missing controls in Exp 8: no random-direction edit, no steering-vector baseline, no
   inference-only fix (repetition penalty), no LoRA on the same practice problems.
4. All convincing non-feedback results (Exp 1–5) use purpose-built toy networks (up to
   1.5M params). The jump to Qwen 0.8B (500M+) with feedback is not supported by the
   toy results.
5. Cherry-picking risk throughout: best-of-3 seeds in Exp 1, two attempts in Exp 3,
   3 pairs in Exp 7, one edit in Exp 4.
6. Closed editor (Astra) is irreproducible by external researchers.
7. Scale and generality: real-LM evidence is one model, one failure type.
8. No Swedish or multilingual evidence at all.

**Justified vs. unjustified conclusions**
- *Justified*: a frontier LLM can, in small structured settings, recover facts from
  weights and produce useful edits to small stuck networks.
- *Justified*: in one case (Qwen 0.8B), a small weight edit removes a looping failure.
- *Justified*: the authors themselves say edits removed a failure, not added a capability.
- *Not established*: Exp 8 edit is better than simpler alternatives; the process is
  "weight reasoning" rather than feedback search; gains represent new capability.
- *Not established at all*: any of this transfers to Swedish or to larger models.

---

## 2  Candidate hypotheses considered

**Candidate A — Loop-removal transfer to Swedish**
Test if an English-found looping-direction edit also reduces looping on Swedish prompts.
*Rejected*: Qwen3.5-0.8B in thinking mode generates up to 32k tokens. This is incompatible
with Colab Free T4 at reasonable throughput. A shortened cap changes the experiment.

**Candidate B — Locating Swedish behavior via weight-diff**
Compare weight-diff analysis of a base vs. Swedish-adapted checkpoint.
*Rejected*: No openly licensed Swedish-adapted checkpoint of the exact same base
architecture was confirmed available at plan-writing time.

**Candidate C — Wrong-language drift correction via targeted weight edit**  ← **CHOSEN**
In a small multilingual model that sometimes answers Swedish prompts in English, can a
targeted weight edit derived from weight/activation analysis reduce the drift rate? And is
any resulting accuracy improvement explained by failure removal (the model already knew
the answer but produced it in the wrong language) rather than new Swedish capability?
*Chosen because*: (a) cleanly testable, (b) directly replicates the failure-removal
distinction the paper makes but never properly controls for, (c) relevant to SE.LLMA
whose models must reliably produce Swedish output, (d) feasible on CPU + T4.

---

## 3  Pre-registered hypothesis

**H**: In Qwen2.5-0.5B-Instruct, language drift (responding in English to Swedish
prompts) can be reduced by a targeted weight edit derived from activation analysis on a
small development set. Any improvement in Swedish-task accuracy will be explained
primarily by *failure removal* — examples the model answered correctly in content but in
the wrong language (English) — rather than by new Swedish language capability.

**Why it matters for SE.LLMA**: SE.LLMA fine-tunes existing open multilingual models for
Swedish. Language drift is a known and costly failure mode in this setting. WSP claims
targeted weight edits can remove failures cheaply. If confirmed, this could complement
or reduce the need for full fine-tuning. If the improvement is only failure removal,
it validates the paper's framing and suggests that prompting fixes may work equally well.
If neither works, that is useful negative evidence.

---

## 4  Primary and secondary metrics (pre-registered)

**Primary metric**: Language-ID accuracy on the Swedish prompt test set.
- Defined as: fraction of model responses correctly identified as Swedish by `langdetect`
  (or `fasttext`) language classifier.
- Computed on the held-out test set (touched once).
- Bootstrap 95% CI over examples (N=200 bootstrap resamples, seed fixed).

**Secondary metric**: Task accuracy on Swedish simple QA.
- Defined as: exact-match or token-overlap accuracy on a held-out Swedish question set.
- Split into two strata: (A) responses already in Swedish by the base model; (B)
  responses that drifted to English in the base model.
- If edit helps only stratum B → failure removal evidence.
- If edit helps stratum A → new capability evidence.

**Regression metrics** (checked for every condition):
- R1: English-language accuracy on a matched English subset (same tasks, same size).
- R2: Swedish perplexity on held-out Swedish Wikipedia sentences (10K tokens).

---

## 5  Support and refute criteria (pre-registered)

| Outcome | Interpretation |
|---|---|
| Edit improves language-ID rate by >10 pp over baseline, CI excludes 0, and exceeds random-direction control by >5 pp | **Supports**: targeted edit reduces drift |
| Accuracy gain is confined to stratum B (drift cases) with p<0.1 by bootstrap | **Supports**: failure removal, not new capability |
| Accuracy gain appears in stratum A (already-Swedish cases) | **Evidence of new capability** (less expected; would challenge WSP framing) |
| Edit effect is within noise of random-direction control | **Refutes**: targeted edit adds no specific benefit |
| Prompt-only baseline matches or exceeds edit on language-ID | **Refutes**: weight edit unnecessary |
| Edit degrades R1 (English accuracy) by >5 pp | **Regression** — edit has unacceptable side effects |

---

## 6  Experimental conditions

All conditions evaluated under identical decoding settings (temperature 0.0, greedy, max
100 new tokens) on the same test examples in the same order with fixed seed.

| ID | Condition | Purpose |
|---|---|---|
| C0 | Unchanged base model | Baseline |
| C1 | Base + Swedish system prompt | Prompt-only control — cheapest fix |
| C2 | Base + Swedish system prompt + constrained-language instruction | Stronger prompt control |
| C3 | Targeted direction-ablation edit (chosen on dev set) | Main method |
| C4 | Random-direction edit (5 draws, matched norm, matched layers) | Null hypothesis for editing |
| C5 | Sign-flipped edit (wrong direction) | Direction-specificity control |
| C6 | LoRA fine-tune on same dev examples (same data budget) | Conventional alternative |

---

## 7  Edit method

**Step 1 — Identify candidate direction (feedback-based, declared)**
- Run model on 50 Swedish dev prompts and 50 English dev prompts (prompt-only, no system
  prompt to maximize drift signal).
- Extract residual-stream activations at each transformer layer.
- Compute mean-activation difference: `d_L = mean(act_swedish_L) - mean(act_english_L)`.
- Rank layers by the cosine similarity between the edit direction and `d_L`.
- Choose the top-1 or top-2 layers on dev language-ID accuracy.

This step *uses model forward passes* and is therefore **feedback-based**, not direct
weight reasoning. This will be explicitly stated in the report.

**Step 2 — Apply edit to weights (direct weight surgery)**
- For the chosen layer(s), subtract `alpha * d_L / ||d_L||` from the MLP output projection
  weight columns that have the highest alignment with `d_L`.
  `W_new = W_old - alpha * (d_L / ||d_L||) * mask`
- `alpha` is a scalar tuned on dev language-ID rate (grid over {0.5, 1.0, 2.0}).
- `mask` selects the top-64 output neurons by alignment score.
- Number of weight parameters changed: at most 64 × hidden_dim per layer, typically <<1%
  of model weights.

**Step 3 — Evaluate on held-out test set**
- Run all conditions on test set exactly once.

---

## 8  Data

**Dev set**: 50 Swedish prompt pairs (factual questions from FLORES-200 Swedish dev,
augmented with synthetic templates) + 50 English prompt pairs (same tasks in English).
Used only to find and tune the edit direction and alpha.

**Test set**: 100 Swedish questions (held-out FLORES-200 or SuperLim subset, disjoint
from dev) + 100 English questions (matched English subset).
Touched once.

**Swedish perplexity corpus**: 200 sentences from Swedish Wikipedia (CC-BY-SA), held out
from all training or tuning steps.

Split seed: 42. Split written to `configs/data_split.json` before any training/editing.
Leakage check: verify zero overlap between dev and test example IDs before running.

---

## 9  Model

- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- Exact revision (commit hash): to be pinned when downloaded and recorded in
  `DATA_AND_MODELS.md`.
- License: Apache 2.0.
- Size: ~0.5B parameters, ~1 GB in fp16. Fits on T4 and runs on CPU for smoke tests.

---

## 10  Statistical plan

- Bootstrap CI: 1000 resamples over examples, seed 0, reported as (mean, 2.5%, 97.5%).
- Paired comparison: McNemar's test for binary outcomes (correct/incorrect) where N≥50.
- Multiple conditions: report all without correction; note which differences are within
  noise.
- Random-direction control: report mean ± SD over 5 seeds.
- Effect size: absolute percentage point difference and Cohen's h for proportions.
- Pre-registered threshold: 10 pp over C0 on primary metric, CI excludes 0.

---

## 11  Decision rule for recommendation

| Evidence | Recommendation |
|---|---|
| Edit >10 pp improvement AND exceeds random control AND <5 pp regression | Investigate further: weight editing is a viable low-data Swedish intervention |
| Edit >10 pp improvement BUT within noise of random control | No: magnitude matters but direction specificity does not; just prompting is probably enough |
| Edit >10 pp AND improvement only in stratum B (drift cases) AND prompting matches | No specific advantage over prompting; failure removal only |
| Edit <10 pp OR within noise | No: weight editing does not reliably reduce language drift at this scale |
| Any condition degrades English by >5 pp | Flag as risk; temper any positive recommendation |

---

## 12  Timeline

| Phase | Activity | Target duration |
|---|---|---|
| 0 | Plan and setup | 20 min |
| 1 | Data + metric code + tests | 30 min |
| 2 | CPU smoke test (20 examples, tiny or base model) | 20 min |
| 3 | Colab notebook + scripts | 30 min |
| 4 | Full run on T4 | 60–90 min |
| 5 | Analysis + figures | 30 min |
| 6 | Write-up: note, PDF, README, AI_USAGE | 30 min |
| 7 | Self-audit | 10 min |

---

## 13  Deviation log

*No deviations recorded yet. Any change after test-set evaluation must be logged here
with the reason.*
