## Candidate assignment: Weight-Space Programming for SE.LLMA

Read Sparks of Weight-Space Programming (https://rikiyat.github.io/blog/weight-space-programming/) and investigate whether the ideas presented could be useful for SE.LLMA's goal of building strong Swedish language models. [URL 🔗](https://wasp-sweden.org/research/projects/sellma/)

We're interested in researchers and engineers who can critically evaluate new ideas, design useful experiments, and draw sound conclusions from limited evidence.

Your task is to investigate one specific, testable claim.

For example:

- Can a targeted weight edit improve a model's Swedish language behavior?

- Can we identify a model's learned behavior by examining its weights?

- Can a targeted edit correct undesirable behavior without degrading other capabilities?

- Does an apparent improvement reflect a genuinely new capability, or the removal of a failure that previously prevented the model from using an existing one?

These are suggestions. You're encouraged to pursue your own ideas.

We value honest investigations, not positive results. A well-designed experiment that demonstrates why an approach doesn't work can be just as valuable as one that succeeds.

Identify what you consider the article's strongest evidence, its most important limitations, and which conclusions are or aren't justified.

Pay particular attention to the distinction between reasoning directly about weights and optimizing edits through repeated model feedback.

Choose one narrow question relevant to Swedish language models.

- Why the question matters.

- What result would support your hypothesis.

- What evidence would challenge it.

Build the smallest experiment you think can meaningfully investigate your question.

You may use open-weight models, public or synthetic Swedish datasets, existing research code, and frontier AI models.

You don't need to train a large model or reproduce the paper's headline results.

Use appropriate controls and compare your approach to at least one reasonable alternative.

Where relevant, consider:

- An unchanged baseline model.

- Random or otherwise untargeted edits.

- Changes to inference settings or prompting.

## The task

## 1. Critically examine the research

## 2. Form a hypothesis

## Explain:

## 3. Design and run an experiment

## 4. Evaluate your findings


- A conventional fine-tuning or steering approach.

Keep evaluation examples separate from those used to develop or select your method. Measure your target behavior and check for at least one possible regression. Include examples, quantitative results where feasible, and the limitations of your conclusions.

## 5. Make a recommendation

Based on your findings, should SE.LLMA investigate this approach further? What would be the most informative next experiment if you had more time or compute?

## Submission

Submit a GitHub repository (public) containing:

- Code: The implementation and a straightforward way to reproduce your main experiment.

- Results: Evaluation examples, outcomes, and relevant configuration details, including model revisions and random seeds where applicable.

- Research note: A concise Markdown report (approximately 1–2 pages) explaining your hypothesis, experiment, results, interpretation, and recommendation.

- Report: Make a PDF based on the research note, that can be longer than two pages if you want.

- AI usage: A short description of the AI tools you used and their contributions.

You may use any tools you find helpful, including coding agents and frontier models. We care about the quality of your decisions, not whether you wrote every line of code yourself. Use only data and model weights that you have the right to use and share.

## Time and resources

Please scope your work accordingly. We don't expect a comprehensive study or a large training run.

A small, insightful experiment is preferable to an ambitious but unfinished one. CPU-only experiments, tiny models, or analysis of existing model checkpoints are acceptable. You are also welcome to submit a partially completed experiment if you clearly explain what you accomplished, what remains unresolved, and what you learned. We do not expect you to pay for compute or API access. If resource constraints prevent your

preferred experiment, explain how you adapted the investigation.

We primarily value:

- 1. Scientific judgment: Can you distinguish strong evidence from weak evidence and avoid unsupported conclusions?

- 2. Experimental design: Is the hypothesis meaningful, and do the controls actually test it?

- 3. Technical execution: Can you implement and evaluate an experiment that works?

## Target effort: 4 hours.

## How we'll assess your work


- 4. Independent thinking: Do you make thoughtful decisions, challenge assumptions, and identify promising directions?

- 5. Communication and reproducibility: Can someone else understand, inspect, and build on your work?

We value creativity, thoughtful negative results, and useful scientific insights more than impressive headline scores.

The central question isn't whether you can improve a model in four hours. It's whether

your investigation makes us better informed about how to improve one.
