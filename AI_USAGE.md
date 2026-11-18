# AI Usage Statement

This file describes how AI tools were used in this assignment, as required by the
submission guidelines.

---

## Tools used

**Claude Sonnet 4.6 (Claude Code CLI, Anthropic), 2026-11-18**

Used throughout the entire assignment in an agentic coding session. Specific
contributions:

- **Paper reading and critical analysis**: Read the CONTEXT_paper_and_project.md digest
  and provided the structured critical analysis in PLAN.md (§1) and RESEARCH_NOTE.md.
  *What to verify*: Every specific claim about paper results (numbers, table values)
  should be checked against the original blog post and the CONTEXT digest, since the
  digest was itself produced by an AI and labelled as such.

- **Hypothesis selection**: Evaluated three candidate hypotheses and selected the
  language-drift direction-ablation hypothesis, with explicit reasoning in PLAN.md (§2–3).

- **Experimental design**: Wrote the full pre-registration in PLAN.md, including
  the control conditions, the stratification analysis (failure removal vs new capability),
  the statistical plan, and the decision rule.

- **Code implementation**: Wrote all Python code in `src/`, `scripts/`, `tests/`,
  and the Colab notebook. All code was run in a venv on the local machine during
  the session. 19/19 unit tests pass. The offline smoke test passes. The online smoke
  test is blocked by a corporate SSL issue (see README).

- **Write-up**: Drafted PLAN.md, README.md, DATA_AND_MODELS.md, AI_USAGE.md, and the
  structure of RESEARCH_NOTE.md.

**What was not AI-generated**:
- Actual experimental results (the Colab run has not been executed yet — this requires
  the user to run the notebook and return the numbers).
- Final numerical claims in RESEARCH_NOTE.md (placeholder sections marked clearly).

---

## What to verify independently

1. **Paper claims**: Numbers in the CONTEXT digest are paraphrased from the original blog
   post. Cross-check against `https://rikiyat.github.io/blog/weight-space-programming/`
   before quoting them.

2. **Model license**: Verify that `Qwen/Qwen2.5-0.5B-Instruct` is currently Apache 2.0
   on HuggingFace before submitting to a public repo.

3. **Dataset license**: Verify FLORES-200 (CC-BY-SA 4.0) and Swedish Wikipedia
   (CC-BY-SA 3.0) licenses at their current HuggingFace pages.

4. **Model revision**: After running the Colab notebook, check that
   `configs/model_config.json` contains the actual commit hash, not "unknown".

5. **Colab results**: The numbers in RESEARCH_NOTE.md results section must be
   replaced with the actual Colab output — do not leave placeholders in the final
   submission.

6. **Metric validity**: Manually inspect at least 20–30 outputs in `results/examples/`
   to confirm that `langdetect` correctly identifies Swedish vs English in this context.
   Short outputs and mixed Swedish/English may be misclassified.

---

## Note on the assignment's AI policy

The assignment states: "You may use any tools you find helpful, including coding agents
and frontier models. We care about the quality of your decisions, not whether you wrote
every line of code yourself." This usage is within that scope. The scientific judgment
(hypothesis choice, design decisions, interpretation, honest characterisation of
limitations) reflects the instructions given in this session, which were specified by
the candidate and checked against the assignment requirements.
