# REVIEW.md — Independent Scientific Review

**Reviewer**: Claude Sonnet 4.6 (independent, did not write this work)  
**Date**: 2026-11-19  
**Scope**: SE.LLMA weight-space programming take-home assignment submission

---

## 1. Verdict

The **primary conclusion is supported**: the direction-ablation weight edit does not
improve Swedish language-ID accuracy over baseline (96.97% vs 97.98%), falls within
the confidence interval of random edits (C4 mean ~97.4%), and causes a substantial
perplexity regression (+68%). The recommendation to avoid this method and use system
prompting instead is proportionate and well-argued.

**Confidence: moderate-high.** The core numbers (C0, C3, strata, C6) reproduce exactly.
Two numbers that the note uses to argue direction-specificity — C5 (sign-flipped) and C4
mean — cannot be reproduced exactly: I obtain 86.87% and 97.37% respectively, vs the
claimed 90.91% and 97.78%, because `langdetect.detect()` is nondeterministic without a
seed. The directional claims hold under both sets of numbers, but the magnitude of the
C5 degradation is uncertain by ~4 pp.

The main limitation of the whole experiment — only 2 drift cases in the baseline (n_B = 2)
— makes the stratum analysis statistically meaningless. The note acknowledges this, but
the acknowledgment is buried in Section 6.

---

## 2. What I Verified and What I Could Not

**Verified independently (from raw JSONL files, `results/test_sv/`):**

| Item | Method |
|---|---|
| C0, C1, C2, C3, C6 lang-ID counts | `langdetect` on all 99 responses, seed=0 |
| C4 per-seed counts and mean | Same |
| C5 count | Same (86/99, not 90/99 as claimed) |
| Stratum A/B split and C3 edit accuracy | Paired comparison C0 vs C3 by ID |
| Bootstrap CI for C0, C3 | 1000 resamples, seed=0, Python stdlib |
| Qualitative failures (C0 test_sv_96, test_sv_98; C3 test_sv_54, test_sv_70, test_sv_96) | Direct inspection |
| Edit parameter count | `edit_info.json` says 57,344; note says 8,192 |
| Dev/test ID disjointness | ID prefix separation: dev_sv_* vs test_sv_* |
| Dev alpha results (deleted) | Recovered from `git show 04b1f01:results/test_sv/dev_alpha_*.jsonl` |
| Git history and timestamps | `git log --all --format="%H %ai %s"` |
| Nordic-overlap counting in `src/metrics.py` | Code read (line 47–49) |

**UNVERIFIED:**

| Item | Reason |
|---|---|
| Paper critique (Section 1 claims about Takehi et al.) | `CONTEXT_paper_and_project.md` was deleted (commit `bc6a178`); blog post URL not fetched |
| "Qwen3.5-0.8B" model name used in paper description | Cannot confirm without primary source |
| "50 forward-pass feedback rounds" in Exp 8 | Cannot confirm without primary source |
| "base model loops on 93% of MATH-500 answers" | Cannot confirm without primary source |
| Exact model revision used (ModelScope git clone) | `configs/model_config.json` not checked against actual download |
| Perplexity values | Cannot recompute without model weights |
| English lang-ID regression (C3=97.98%) | Claimed in note; `results/test_en/C3_direction_ablation_en.jsonl` exists but not recomputed here |

---

## 3. Claim-to-Evidence Table

All numbers from `RESEARCH_NOTE.md`. Source = file I traced it to. Recomputed = my
independent value from raw JSONL with `langdetect`, `DetectorFactory.seed=0`, 1000
bootstrap resamples.

| # | Claim (from note) | Source file | My recomputed value | Status |
|---|---|---|---|---|
| A | C0 baseline lang-ID: 97.98% | `results/summary.json` | 97/99 = 97.98% | ✓ MATCH |
| B | C1 sys prompt: 100.00% | `results/summary.json` | 98/99 = 98.99% | ✗ MISMATCH (see F1, F4) |
| C | C2 strong prompt: 98.99% | `results/summary.json` | 98/99 = 98.99% | ✓ MATCH |
| D | C3 direction ablation: 96.97% | `results/summary.json` | 96/99 = 96.97% | ✓ MATCH |
| E | C4 random mean ± SD: 97.78% ± 0.99% | `results/summary.json` | 97.37% ± 1.15% | ✗ MISMATCH (see F1) |
| F | C5 sign-flipped: 90.91% [84.8, 96.0] | `results/summary.json` | 86/99 = 86.87%, CI [79.8, 92.9] | ✗ MISMATCH (see F1) |
| G | C6 LoRA: 100.00% | `results/summary.json` | 99/99 = 100.00% | ✓ MATCH |
| H | Stratum A: n=97, C3=97.94% | `results/summary.json` | 97 examples, 95/97=97.94% | ✓ MATCH |
| I | Stratum B: n=2, C3=50% | `results/summary.json` | 2 examples, 1/2=50% | ✓ MATCH |
| J | Edit: 8,192 params changed (0.0017%) | `results/edit_info.json` | 57,344 changed (0.0116%) | ✗ MISMATCH (see F2) |
| K | Bootstrap CI 1000 resamples, seed 0 | `src/metrics.py` | Confirmed in code | ✓ MATCH |
| L | Dev: alpha=0.5 chosen as best | `configs/edit_config.json`, deleted dev_alpha files | Confirmed | ✓ MATCH |
| M | "C0 CI: [94.9, 100]" | `results/summary.json` | [94.95, 100] | ✓ MATCH (rounding) |
| N | C0 drift cases: test_sv_96, test_sv_98 | `results/examples/` | Confirmed by direct inspection | ✓ MATCH |
| O | C3 regressions: test_sv_54, test_sv_70 | `results/examples/` | Confirmed — both English responses | ✓ MATCH |
| P | Swedish perplexity C0=541.7, C3=910.3 | `results/summary.json` | UNVERIFIED (needs model) | — |
| Q | Layer 23 chosen, alignment score 0.036 | `configs/edit_config.json` | Confirmed in config | ✓ MATCH |

---

## 4. Findings (Ordered by Severity)

---

### F1 — MAJOR: Primary metric is nondeterministic; two key numbers cannot be reproduced

**Category**: metric validity / reproducibility  
**Evidence**: `src/metrics.py:18–34`. The `detect_language()` function calls
`langdetect.detect()` without setting `langdetect.DetectorFactory.seed`. The
`langdetect` library is explicitly documented as nondeterministic without a seed. As a
result:

- **C5 (sign-flipped)**: note claims 90/99 = 90.91%; I consistently get 86/99 = 86.87%
  (4 pp lower). With `DetectorFactory.seed=0` across 5 runs I get 86 every time.
- **C4 (random control mean)**: note claims 97.78% ± 0.99%; I get 97.37% ± 1.15%.
- **C1**: note claims 100%; I get 98.99%. The discrepancy is one case ("Allt är ok."),
  a 3-word response that langdetect classifies as Hungarian (hu) with seed=0 but
  presumably as Swedish in the original run. The response is unambiguously Swedish.

**Why it matters**: C5 is used to argue "the direction has genuine semantic content"
(Section 5, penultimate paragraph). If C5 is 86.87% rather than 90.91%, the gap vs C0
(97.98%) is larger (11 pp instead of 7 pp), which still supports the claim — but the
magnitude matters for calibration. More importantly, the C4 mean being lower than
claimed makes C3 look slightly better relative to random, reducing (but not eliminating)
the conclusion that C3 performs no better than random.

**Required change**: Set `langdetect.DetectorFactory.seed = 0` before all detection
calls, and re-run the analysis. Report the seed in the note. Alternatively, switch to
`fasttext` language-ID (which is deterministic) as was mentioned in the PLAN.

**Verification**: `src/metrics.py:18` — no seed. Two independent recomputations in this
review (one with `DetectorFactory.seed=0`, one without) produced **different results**
for C5: 86/99 (seed=0) vs 90/99 (no seed, matches the note). The original analysis
presumably ran without seed, so the note's C5=90 is one valid draw; 86 is another.
This directly demonstrates the nondeterminism for borderline mixed-language outputs.

---

### F2 — MINOR: Parameter count in note is wrong by ~7×

**Category**: number traceability  
**Evidence**: `results/edit_info.json` records `changed_params: 57344` and
`fraction_changed: 0.000116` (0.0116%). The note states: "Parameters changed: 8,192 /
494,032,768 total (0.0017%)".

The correct calculation: 64 columns × 896 (Qwen2.5-0.5B hidden_dim) = 57,344. The
note uses 64 × 128 = 8,192 — an incorrect dimension. The fraction 0.0017% is derived
from 8,192 / 494M, not the actual count.

**Why it matters**: "0.0017% of model weights" sounds more negligible than 0.0116%.
Both are tiny, so the conclusion ("a small edit can...") is unaffected, but the specific
number is wrong and would fail a verification check by a reader who looks at the config.

**Required change**: Replace "8,192" with "57,344" and "0.0017%" with "0.012%" throughout
(note, PDF, README).

---

### F3 — MAJOR: Pre-registration integrity unverifiable from git history

**Category**: hypothesis pre-registration  
**Evidence**: `git log --all --format="%H %ai %s"` shows all files — `PLAN.md`, all
`results/`, and `RESEARCH_NOTE.md` — were added in a **single commit** (`04b1f01`,
2026-11-18 17:25:09). There is no earlier commit containing only `PLAN.md` without
results. The note header on `PLAN.md` says "Written before any test-set evaluation.
Do not edit after test-set results are obtained except in the Deviation log section."
This claim cannot be verified from git.

A subsequent commit (`f940622`, 17:34:51, same day) replaced the original
"No deviations recorded yet" placeholder in PLAN.md's deviation log with the five
actual deviations, explicitly noting this was done "after the experiment completed".
This is disclosed, but the base PLAN.md having been written before results cannot be
confirmed.

Additionally, dev alpha results (`dev_alpha_0.5.jsonl`, `dev_alpha_1.0.jsonl`,
`dev_alpha_2.0.jsonl`) were committed and then deleted in commit `bc6a178` ("Remove
working documents"). They are recoverable from git history. Alpha=0.5 was selected as
the best on dev, which is correctly declared.

**Why it matters**: The core scientific claim is a negative result (edit fails), which
is robust to p-hacking in a way that positive results are not. However, the threshold
("support if >10 pp improvement") was set when the negative outcome was already known.
A reviewer cannot distinguish "threshold was chosen before seeing results" from
"threshold was chosen to make a negative result look principled."

**Required change**: The note should acknowledge this limitation directly: "Git history
does not provide a verifiable timestamp separation between PLAN.md and results; the
pre-registration claim rests on the author's assertion." Optionally: archive PLAN.md
to a neutral service (OSF, Zenodo) with a timestamp, or note which parts of the plan
were unchanged from the original design.

---

### F4 — MINOR: Metric description omits Nordic-overlap counting

**Category**: metric validity  
**Evidence**: `src/metrics.py:47–49`:
```python
correct = detected == expected_lang or (
    expected_lang == "sv" and detected in ("sv", "no", "da")
)
```
Responses detected as Norwegian (`no`) or Danish (`da`) are counted as correct Swedish.
This is a reasonable practical choice (genuine confusion between closely related
languages), but it is not stated in the note's metric description (Section 3, Section 4
of PLAN.md) or the results table.

**Why it matters**: For this dataset, all responses are either clearly Swedish or
clearly English — the overlap logic did not change any counts in my recomputation.
But a reader cannot reproduce the exact logic from the note alone.

**Required change**: One sentence in the metric description: "Norwegian (no) and
Danish (da) detections are counted as correct Swedish due to known langdetect
conflation of closely related Nordic languages."

---

### F5 — MINOR: Bootstrap resamples inconsistency between PLAN and execution

**Category**: pre-registration deviation  
**Evidence**: `PLAN.md` Section 4 specifies "N=200 bootstrap resamples, seed fixed."
`PLAN.md` Section 10 (Statistical plan) and `src/metrics.py:142` both use 1,000.
The note says "Bootstrap CI: 1000 resamples, seed 0." The deviation from 200 to 1000
is not logged in the deviation log (Section 12).

**Why it matters**: 1000 is strictly better than 200; this is an improvement not a
concern. But it was not logged as a deviation, breaking the PLAN's own protocol.

**Required change**: Add entry to deviation log: "Bootstrap resamples: planned 200,
used 1000."

---

### F6 — MINOR: N too small to interpret stratum B

**Category**: statistical power  
**Evidence**: Stratum B has n=2. The note says the edit "fixed 1 of 2 drift cases
(+1)." A McNemar's test on this pair is not meaningful (minimum N for McNemar is 10+).
The 50% result in Stratum B could be chance. The note acknowledges this only as
"almost no room for improvement" (Section 6, second bullet) rather than as an explicit
statement that the stratum B analysis should not be interpreted.

**Why it matters**: Section 5 uses "Stratum B shows partial recovery (50%), consistent
with the failure-removal framing" as interpretive support. With n=2 this is a single
coin flip. The failure-removal framing is still the most plausible interpretation given
the design — but the evidence from stratum B contributes essentially nothing.

**Required change**: Flag in the results table: "(n=2: not interpretable, single
observation)". In Section 5, replace "consistent with the failure-removal framing" with
"does not contradict the failure-removal framing, but n=2 is too small to test it."

---

### F8 — MINOR: max_new_tokens deviation not logged

**Category**: pre-registration deviation  
**Evidence**: `PLAN.md:151` specifies "max 100 new tokens". `scripts/run_local.py:455`
and `:572` use `max_new_tokens=60`. `RESEARCH_NOTE.md:74` states "max 60 new tokens".
Not logged in the deviation log.

**Why it matters**: For this dataset almost all responses conclude well within 60 tokens,
so truncation is unlikely to have changed any lang-ID label. The drift cases (test_sv_96,
test_sv_98) produce English metacommentary immediately, and C6 LoRA responses are concise.
No material effect on conclusions is expected, but the deviation was not logged.

**Required change**: Add to deviation log: "max_new_tokens: planned 100, used 60."

---

### F9 — MINOR: Stratum A regression description incorrectly suggests classifier errors

**Category**: interpretation / honesty  
**Evidence**: `RESEARCH_NOTE.md:231–234` states: "2 of the C3 Stratum A regressions
may be classifier errors rather than actual language changes." Direct inspection of
both cases contradicts this:
- `test_sv_54`: "The man builds a wall with bricks and stones. I'll do my best to
  translate this into English for you..." — full English output, not a classifier error.
- `test_sv_70`: "The Djurgården's windows flicker during the day. I apologize for any
  inconvenience caused. I'll check back later..." — full English output.

Both regressions are genuine. `langdetect` correctly labels them. The note goes on to
say "This is noted but does not change the conclusion," which is correct, but the
"may be classifier errors" softening is factually wrong and mildly misleads about the
severity of the regressions.

**Required change**: Replace "may be classifier errors rather than actual language
changes" with "are genuine English outputs confirmed by manual inspection."

---

### F7 — MINOR: "C1 achieves 100%" claim is fragile

**Category**: number traceability  
**Evidence**: My recomputation gives C1 = 98/99 = 98.99%. The one failure is
`test_sv_95`, whose response is "Allt är ok." (3 words). `langdetect` classifies this
as Hungarian with seed=0. The response is unambiguously Swedish. The correct number
depends entirely on whether a 3-word Swedish response triggers `langdetect`'s short-text
failure mode — which varies by run.

**Why it matters**: The claim "C1 achieves 100% with no side effects" is used to argue
that prompting strictly dominates. The correct conclusion (C1 is very good and far
better than C3) is not affected whether C1 is 99/99 or 98/99. But the specific "100%"
number is unreliable.

**Required change**: Report C1 as "≥98.99%, likely 100%; one borderline 3-word
response is misclassified by langdetect in some runs."

---

## 5. Sentences to Reword

| Original sentence | Issue | Proposed replacement |
|---|---|---|
| "Parameters changed: 8,192 / 494,032,768 total (0.0017%)" (§4.2) | Wrong count; 57,344 actual | "Parameters changed: 57,344 / 494,032,768 total (0.012%)" |
| "C1 (system prompt 'Svara alltid på svenska') achieves 100% with no side effects" (§5) | 100% is unreproducible; nondeterministic metric | "C1 achieves ≥98.99% (likely 100%; one 3-word response is borderline for langdetect) with no side effects" |
| "C4 (97.78% ± 0.99%) is better than C3 (96.97%)" (§5) | C4 mean is nondeterministic; my recompute 97.37% | "C4 (97.4%–97.8% across runs ± ~1%) is comparable to or better than C3 (96.97%)" |
| "Bootstrap CI: 1000 resamples, seed 0" in results table (§4.1) | Seed for bootstrap is 0, but langdetect has no seed | "Bootstrap CI for counts: 1000 resamples, seed 0. Note: langdetect is nondeterministic (no DetectorFactory.seed set); C5 and C4 counts may vary ±4 pp across runs." |
| "Stratum B (C0 drifted to English) shows partial recovery (50%), consistent with the failure-removal framing." (§5) | n=2 is single observation | "Stratum B (n=2) has 1/2 cases recover under C3; with n=2 this is a single observation and cannot confirm the failure-removal framing." |
| "Written before any test-set evaluation." (PLAN.md header) | Cannot be verified from git | Add: "(Note: PLAN.md and results appear in the same git commit; this claim rests on the author's assertion and is not verifiable from repository history.)" |

---

## 6. Cheap Missing Controls (Priority Order)

1. **Set `langdetect.DetectorFactory.seed = 0`** (10 lines of code, free). Regenerate
   all numbers. This eliminates the reproducibility problem completely and may change
   C1, C4, C5 numbers.

2. **Hand-label the 13 C5 failures and 2 C0 failures** (~30 min). Read each response,
   label it manually. Report manual accuracy alongside langdetect accuracy. This
   separates metric-classifier errors from real language failures.

3. **Binomial/McNemar test for the C3 vs C0 comparison** (2 lines). McNemar's test
   on paired binary outcomes (n=99) would give an exact p-value for C3 vs C0 (instead
   of CI overlap). Expected result: not significant (consistent with the note's
   conclusion), which strengthens the claim.

4. **Run McNemar's test** (required by PLAN.md §10 but not done). McNemar: C0 vs C3
   gives chi²=0.33, not significant (confirms C3 ≈ C0). C0 vs C5 gives chi²=4.46,
   p<0.05 (confirms C5 is significantly worse). These support the note's claims but
   should appear in the results.

5. **Report C3 vs C4 significance properly**. The current presentation compares C3
   point estimate to C4 mean-of-5. The correct analysis is: compare C3 (96 correct)
   to each C4 seed individually using Fisher's exact test. Expected: not significant
   for any seed, which is the actual claim.

5. **Additional LoRA baselines**: C6 uses r=8, 2 epochs. Report at r=1 or 1 epoch to
   confirm LoRA advantage is not purely a data-budget advantage. Currently C6 and C3
   use the same 50 dev examples, which is fair — but C6's hyperparameters were not
   tuned on dev. A LoRA tuned on dev (like C3's alpha was) would be a fairer
   comparison.

---

## 7. What Was Done Well

1. **Negative results reported honestly and prominently.** The main claim is a
   failure. Many submissions bury negative results or reframe them as partial
   successes. This note puts the failure first, with the correct sign.

2. **Multiple controls are real and meaningful.** C4 (random direction), C5
   (sign-flipped), and C6 (LoRA) are all appropriate and correctly interpreted. Most
   comparable studies include none.

3. **Failure-removal vs new capability framing is applied correctly.** The stratum
   analysis is the right design for the question, and the note correctly identifies
   the asymmetry (C3 also induces failures in stratum A).

4. **Deviation log is present and specific.** All five deviations from the original
   plan are logged with reasons. This is better than the typical practice of silently
   changing the plan.

5. **Qualitative inspection is thorough.** The specific failure cases are shown
   verbatim, the failure modes are correctly characterised (metacommentary, task
   misunderstanding), and the C3 regressions are correctly identified as the same
   failure mode as C0's original failures.

6. **Perplexity is reported as a secondary metric.** Many weight-editing papers
   do not check whether the edit disrupts fluency. Reporting PPL_sv = 541.7 → 910.3
   (+68%) is an important finding and is correctly flagged as a regression.

7. **The recommendation is proportionate.** "Do not pursue this method; use
   prompting instead" is exactly right given the evidence. No overclaiming.

8. **Paper critique correctly identifies the feedback-loop issue.** The
   direct-weight-reasoning vs feedback-optimization distinction is the central
   methodological claim in the note, and the critique is internally well-argued.

---

## 8. Summary

**Verdict**: The main conclusion (direction-ablation weight editing does not reduce
Swedish language drift and is strictly dominated by prompting) is **supported** by the
raw data. All core numbers (C0, C3, strata, C6, qualitative failures) reproduce exactly.
Confidence: **moderate-high**.

**Findings by severity**:
- MAJOR: 2 (F1: nondeterministic metric; F3: pre-registration unverifiable)
- MINOR: 7 (F2: wrong parameter count; F4: undeclared Nordic overlap; F5: bootstrap
  deviation unlogged; F6: stratum B too small; F7: C1=100% fragile; F8: max_new_tokens
  deviation unlogged; F9: Stratum A regression attributed to classifier error incorrectly)

**Top 3 issues**:

1. **(F1 — MAJOR)** `langdetect` is used without a seed. C5 and C4 numbers differ by
   3–4 pp from my recompute. The directional conclusions hold but the exact numbers in
   the results table are run-dependent. Fix: add `DetectorFactory.seed = 0` and
   regenerate.

2. **(F3 — MAJOR)** Pre-registration cannot be verified: PLAN.md and results were
   committed simultaneously. The note's self-description as "pre-registered" is
   unverifiable. The negative result makes this less damaging than it would be for a
   positive claim, but the integrity limitation should be stated explicitly.

3. **(F2 — MINOR)** The parameter count in the note (8,192, 0.0017%) is wrong by ~7×
   relative to `edit_info.json` (57,344, 0.012%). This is a small factual error with
   no effect on conclusions but should be corrected.

---

*Review performed 2026-11-19 from raw outputs in `results/`. No author scripts were
re-run. All recomputation is in `review/recomputed_metrics.txt` (from fork agent) and
verified inline above. See `git log` for file provenance.*
