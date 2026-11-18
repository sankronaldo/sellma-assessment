# AI Usage Statement

## Tools used

**Claude Sonnet 4.6 (Claude Code CLI, Anthropic), 2026-11-18**

Used throughout the assignment in an agentic coding session. Specific contributions:

- **Paper reading and critical analysis**: Provided the structured critical analysis in
  `PLAN.md` §1 and `RESEARCH_NOTE.md` §1, based on the paper blog post and a background
  digest provided at the start of the session.

- **Hypothesis selection**: Evaluated three candidate hypotheses and selected the
  language-drift direction-ablation hypothesis, with explicit reasoning in `PLAN.md` §2–3.

- **Experimental design**: Wrote the full pre-registration in `PLAN.md`, including
  control conditions, the stratification analysis (failure removal vs new capability),
  the statistical plan, and the decision rule.

- **Code implementation**: Wrote all Python code in `src/`, `scripts/`, and `tests/`.
  All code was run locally on an RTX 2050 GPU. 19/19 unit tests pass.

- **Experiment execution**: Downloaded the model via ModelScope, ran all 7 conditions,
  generated figures and the metrics CSV.

- **Write-up**: Drafted `PLAN.md`, `RESEARCH_NOTE.md`, `README.md`,
  `DATA_AND_MODELS.md`, and this file.

## What was not AI-generated

The experimental results are real: all numbers in `RESEARCH_NOTE.md` come from code
that ran on actual model weights and are traceable to `results/summary.json`.
The scientific interpretation (what the results mean and why) was written by the AI
but reflects genuine outputs, not fabricated ones.

## What to verify independently

1. **Paper claims**: Numbers cited about the original WSP paper should be checked
   against the blog post at `https://rikiyat.github.io/blog/weight-space-programming/`.

2. **Model license**: `Qwen/Qwen2.5-0.5B-Instruct` is Apache 2.0. Verify at
   the ModelScope or HuggingFace model page before redistribution.

3. **Metric validity**: `langdetect` can conflate Swedish with Norwegian/Danish on
   short outputs. Manual inspection of `results/examples/` confirmed reliable
   classification on outputs of >10 tokens in this experiment.

4. **Synthetic data**: The Swedish and English sentences used for dev and test are
   hardcoded in `scripts/run_local.py`. They are factual, varied, and cover multiple
   topics, but they are not drawn from a standard benchmark. Results on a naturalistically
   sampled dataset may differ.

## Note on AI policy

The assignment states: "You may use any tools you find helpful, including coding agents
and frontier models. We care about the quality of your decisions, not whether you wrote
every line of code yourself." This usage is within that scope.
