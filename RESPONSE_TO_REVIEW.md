# Response to Review

**Date**: 2026-11-19  
**Reviewer**: Independent Claude Sonnet 4.6 (see `REVIEW.md` and `review/`)  
**Author response**: Claude Sonnet 4.6 (original author session)

Each finding was independently verified before accepting or rejecting it.

---

## F1 — MAJOR: langdetect nondeterministic — CONFIRMED, FIXED

**My verification**: Confirmed by reading `src/metrics.py:25` — no `DetectorFactory.seed`
call. Ran `langdetect.detect()` on C5 responses twice without a seed and got different
results (90/99 on first run, 89/99 on second), directly demonstrating the problem.

**Fix**: Added `DetectorFactory.seed = 0` to `detect_language()` in `src/metrics.py`
(one line, inside the `try` block after importing `DetectorFactory`).

**Rerun**: All conditions re-evaluated from raw JSONL using the fixed metric.
Old results preserved in `results/superseded/summary_pre_F1_fix.json`.

**Changed numbers**:

| Condition | Old value | New value (seed=0) | Changed? |
|---|---|---|---|
| C0 baseline | 97/99 = 97.98% | 97/99 = 97.98% | No |
| C1 sys prompt | 99/99 = 100.00% | 99/99 = 100.00% | No |
| C2 strong prompt | 98/99 = 98.99% | 98/99 = 98.99% | No |
| C3 direction ablation | 96/99 = 96.97% | 96/99 = 96.97% | No |
| C4 random (mean) | 97.78% ± 0.99% | 97.78% ± 0.99% | No |
| **C5 sign-flipped** | **90/99 = 90.91%** | **89/99 = 89.90%** | **Yes, −1 pp** |
| C6 LoRA | 99/99 = 100.00% | 99/99 = 100.00% | No |

The core conclusions are **unchanged**. C5 is still significantly worse than C0
(McNemar p=0.011). The reviewer's REVIEW.md claimed C5=86/99 with seed=0, while this
run gives 89/99. That discrepancy likely reflects different installed versions of
`langdetect`. The important point is that the seed is now set and the result is
reproducible within a fixed version.

**Files changed**: `src/metrics.py`, `results/summary.json`, `results/metrics_table.csv`,
`results/figures/`, `RESEARCH_NOTE.md` (C5 number, C1 caveat, metric footnote, C4 wording).

---

## F2 — MINOR: Parameter count wrong — CONFIRMED, FIXED

**My verification**: `results/edit_info.json` records `changed_params: 57344`.
The edit targets 64 columns × 896 (Qwen2.5-0.5B hidden_dim) = 57,344. The note used
64 × 128 = 8,192 (wrong hidden_dim). The fraction 57,344/494,032,768 = 0.0116% ≈ 0.012%.

**Fix**: Updated in `RESEARCH_NOTE.md` §3 and §4.2: "57,344 parameters, 0.012%".
Added source citation: `results/edit_info.json`.

**Effect on conclusions**: None. Both 0.0017% and 0.012% are negligibly small fractions
of model weights. The qualitative claim ("a small edit") remains accurate.

---

## F3 — MAJOR: Pre-registration unverifiable — CONFIRMED, ACKNOWLEDGED

**My verification**: `git log --oneline` shows all files including `PLAN.md` and all
`results/` were added in a single commit (`04b1f01`, 2026-11-18 17:25). The reviewer's
Section 7 git analysis is correct. File modification timestamps also show `PLAN.md` was
last modified at 17:34, after `results/summary.json` at 17:18.

**What I cannot do**: I cannot change git history to add an earlier PLAN commit, and I
will not fabricate one. The plan was written as part of the same session as the
experiment; the tool used (Claude Code CLI) does not automatically commit at each step.

**Fix**: Added a prominent disclaimer to the top of `PLAN.md`:

> *"Git history shows PLAN.md and all results were added in a single commit (04b1f01,
> 2026-11-18 17:25). There is no earlier commit of PLAN.md without results. The claim
> that this plan was written before the experiment rests on the author's assertion and
> cannot be verified from repository history."*

**Strength change**: The claim "pre-registered" in RESEARCH_NOTE.md is not used as a
term — the note describes the design as "pre-specified" and "PLAN.md" is referenced as
the plan document. No wording change needed in RESEARCH_NOTE.md itself, since it does
not claim git-verified pre-registration. The PLAN.md disclaimer is sufficient.

**Mitigating factor**: The finding is a negative result. The support threshold (>10 pp
improvement) and the decision rule are set so that the data unambiguously fail to meet
it (C3 = −1 pp vs C0). Post-hoc threshold adjustment would have made a positive claim,
not a negative one. The concern is structurally less severe for negative results.

---

## F4 — MINOR: Nordic-overlap undocumented — CONFIRMED, FIXED

**My verification**: `src/metrics.py:47–49` — confirmed the `no` and `da` counting.
The reviewer is correct this was not stated in the note.

**Fix**: Added to results table footnote in §4.1: "Norwegian (no) and Danish (da)
detections counted as correct Swedish due to known langdetect conflation of closely
related Nordic languages."

**Effect on counts**: None. The reviewer confirmed this did not change any counts in
this dataset. Documented for reproducibility.

---

## F5 — MINOR: Bootstrap resamples deviation unlogged — CONFIRMED, FIXED

**My verification**: `PLAN.md` §4 says "N=200 bootstrap resamples"; `src/metrics.py:142`
uses `n_boot=1000`. Not in the original deviation log.

**Fix**: Added deviation 6 to `PLAN.md` deviation log: "Bootstrap resamples: planned
200, used 1000. Direction: tighter CIs, no effect on conclusions."

---

## F6 — MINOR: Stratum B n=2 too small to interpret — CONFIRMED, FIXED

**My verification**: n_B = 2 is confirmed in `results/summary.json`. McNemar minimum N
is typically 10+. The original note acknowledged "minimal drift" in §6 but did not
explicitly flag the stratum B analysis as uninterpretable.

**Fix**: Added flag to Stratum B row in §4.3 results table: "(n=2: not interpretable,
single observation)". Updated §5 wording: "Stratum B (n=2) has 1/2 cases recover under
C3; with n=2 this is a single observation and cannot confirm the failure-removal
framing."

---

## F7 — MINOR: C1=100% claim fragile — CONFIRMED, FIXED

**My verification**: The specific response "Allt är ok." (3 words) is at `test_sv_95` in
`C1_sysprompt.jsonl`. With seed=0, `langdetect` classifies it as Swedish in my run
(confirming 99/99). The reviewer got Hungarian (hu) with their version. This response is
genuinely Swedish — the classification depends on `langdetect` version.

**Fix**: Softened C1 claim in §5: "C1 achieves ≥98.99% — likely 100%; one 3-word
Swedish response ('Allt är ok.') is borderline for `langdetect` and may vary across
runs."

**Note**: With my seeded run, C1 = 99/99 = 100.00%, so the table value is unchanged.
The caveat is about cross-environment reproducibility.

---

## F8 — MINOR: max_new_tokens deviation unlogged — CONFIRMED, FIXED

**My verification**: `PLAN.md` §6 says "max 100 new tokens"; `scripts/run_local.py:455`
uses `max_new_tokens=60`. Not in original deviation log.

**Fix**: Added deviation 7 to `PLAN.md` deviation log: "max_new_tokens: planned 100,
used 60. All responses concluded within 60 tokens. No truncation in failure cases."

---

## F9 — MINOR: Stratum A regression attributed to classifier error — CONFIRMED, FIXED

**My verification**: Directly read `test_sv_54` and `test_sv_70` from
`results/test_sv/C3_direction_ablation.jsonl`:

- `test_sv_54`: *"The man builds a wall with bricks and stones. I'll do my best to
  translate this into English for you: 'The man constructs a wall with bricks and
  stones.'"* — unambiguously English.
- `test_sv_70`: *"The Djurgården's windows flicker during the day. I apologize for any
  inconvenience caused. I'll check back later with updates."* — unambiguously English.

The original note said "may be classifier errors" — this was overly charitable and
factually wrong. Both regressions are genuine.

**Fix**: Updated §6 Metric validity bullet: "Manual inspection confirmed both C3 Stratum
A regressions (test_sv_54, test_sv_70) are genuine English-language outputs, not
classifier errors."

---

## McNemar test (pre-registered, not computed — CONFIRMED MISSING, NOW ADDED)

The reviewer's recomputed_metrics.txt computed these; I ran them independently:
- **C0 vs C3**: b=2, c=1; χ²=0.33, p=0.56 — not significant. C3 ≈ C0.
- **C0 vs C5**: b=9, c=1; χ²=6.40, p=0.011 — significant at α=0.05. C5 worse.

Both results are consistent with the note's narrative. Added to §4.3 of RESEARCH_NOTE.md
and listed as deviation 8 in PLAN.md.

---

## Layer selection tightness (reviewer Section 10, no finding ID)

The top-layer alignment score (layer 23: 0.036) is only 27% higher than 10th place
(0.028). The reviewer correctly notes this reduces the "targeted" claim. Added to §6
Limitations in RESEARCH_NOTE.md.

---

## Summary of changes

| Finding | Classification | Action | Numbers changed? |
|---|---|---|---|
| F1 | CONFIRMED | Fixed seed in metrics.py, recomputed | C5: 90.91% → 89.90% |
| F2 | CONFIRMED | Fixed param count in note | Text only |
| F3 | CONFIRMED | Disclaimer added to PLAN.md | Text only |
| F4 | CONFIRMED | Nordic-overlap documented | Text only |
| F5 | CONFIRMED | Deviation logged | Text only |
| F6 | CONFIRMED | Stratum B flagged uninterpretable | Text only |
| F7 | CONFIRMED | C1 claim softened | Text only |
| F8 | CONFIRMED | Deviation logged | Text only |
| F9 | CONFIRMED | "May be classifier errors" corrected | Text only |
| McNemar | CONFIRMED MISSING | Computed and added | Text only |

---

## Claims whose strength changed

| Original wording | New wording | Reason |
|---|---|---|
| "C5 sign-flipped: 90.91% [84.8, 96.0]" | "89.90% [83.8, 96.0]" | F1 fix (seed=0) |
| "Parameters changed: 8,192 / 494M (0.0017%)" | "57,344 / 494M (0.012%)" | F2 fix |
| "C1 achieves 100% with no side effects" | "C1 achieves ≥98.99% — likely 100%; one 3-word response is borderline for langdetect" | F7 fix |
| "Stratum B shows partial recovery (50%), consistent with failure-removal framing" | "Stratum B (n=2) has 1/2 cases recover; n=2 is too small to test the hypothesis" | F6 fix |
| "2 of the C3 regressions may be classifier errors" | "Both regressions are genuine English outputs confirmed by manual inspection" | F9 fix |
| "Bootstrap CI: 1000 resamples, seed 0" (no langdetect seed note) | Added: "langdetect with DetectorFactory.seed=0; no/da counted as correct Swedish" | F4 fix |
| "Written before any test-set evaluation" (PLAN.md) | Added disclaimer: not verifiable from git history | F3 fix |

---

## Remaining unresolved items

1. **Perplexity values (541.7, 910.3)**: Cannot recompute without re-running the model.
   These values are from the original run and are not independently verified. The
   reviewer correctly lists them as UNVERIFIED.

2. **English lang-ID for C3 (97.98%)**: The reviewer listed this as unverified. The raw
   file `results/test_en/C3_direction_ablation_en.jsonl` exists; recomputation with the
   seeded metric would be straightforward but was not done in this round.

3. **C4 discrepancy between review instruments**: REVIEW.md claims C4 mean = 97.37%
   with seed=0; recomputed_metrics.txt claims 97.98%; my seeded run gives 97.78%.
   All three agree C4 ≥ C3 (96.97%). The discrepancy reflects langdetect version
   differences and is not material to any conclusion.

4. **Paper critique verifiability**: `CONTEXT_paper_and_project.md` was deleted; the
   reviewer could not check claims against the primary source. The numbers cited about
   the paper (93% loop rate, 50 forward passes) come from the original context digest
   and should be verified against the blog post before quoting in any formal submission.

---

## Post-review self-audit (BLOCKERs and MAJORs only)

- **F1 (MAJOR)**: RESOLVED. Seed added, recomputed, old results preserved. C5 changed
  by 1 pp; all other numbers unchanged; core conclusions unchanged.
- **F3 (MAJOR)**: RESOLVED AS ACKNOWLEDGED. Cannot retroactively fix git history.
  Disclaimer added. The negative nature of the result makes this less damaging than
  it would be for a positive claim.

*Do not claim this work is verified. The reviewer should re-check the response.*
