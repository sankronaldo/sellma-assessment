# Context digest: "Sparks of Weight-Space Programming" and SE.LLMA

**Purpose of this file.** Background material for the take-home assignment (see `ASSIGNMENT.md` / the assignment PDF). It has three parts: a digest of the paper, a digest of the SE.LLMA project, and a seed list of critical observations and experiment ideas.

**Provenance and trust levels.**
- Sections 1 and 2 are a paraphrased digest of the two web pages, retrieved on 2026-11-18. Numbers are copied from the pages. Wording is not.
- Section 3 onward is analysis written by an AI assistant (Claude). It is a starting point, not ground truth. Verify it against the sources, and do not copy it into the research note without checking.
- Anything marked **[VERIFY]** was not checked against a primary source.

**Sources**
- Paper (blog post): https://rikiyat.github.io/blog/weight-space-programming/
  - Authors: Takehi, Teoh, Zhang, Khattab (MIT CSAIL). Published 2026-11-02. A "Response to Comments" section was added 2026-11-12.
  - Logs of every experiment: https://github.com/RikiyaT/lm_understand-logs
  - Edited Qwen3.5-0.8B (Experiment 8): https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
  - Third-party replications are listed in the post (Hugging Face demo space, plus GitHub repos by Khajavi and statsleelab). **[VERIFY]** Their quality has not been checked.
- Project: https://wasp-sweden.org/research/projects/sellma/

---

## 1. The paper, digested

### 1.1 What it claims
- It is a blog post, not a peer-reviewed paper. There is no formal methods section.
- Central claim: a frontier LLM, used as-is, can **read** neural network weights to infer what the network does, and can **write or edit** weights to change behavior, with no gradient-based training by the LLM.
- The authors call this **Weight-Space Programming (WSP)**: treat weights as a program that is read and edited.
- The "programmer" is a closed frontier model, called **"GPT-6 Astra"** in the post. You cannot access it.
- In the Nov 12 addendum, the authors say WSP is **a concept and a statement about an emerging model capability, not a method**. They say the experiments are evidence of "sparks" of the capability, not a proposed editing technique, and that baselines are chosen to test each experiment's narrow claim rather than to rank methods.

### 1.2 What the LLM is allowed to use, by experiment
This matters for the assignment's required distinction between "reasoning directly about weights" and "optimizing through repeated model feedback".

| # | Task | Model / network | What the LLM gets | Feedback loop? |
|---|------|-----------------|-------------------|----------------|
| 1 | Name the game a network was trained on | MLP 500→32→32→5, 17,253 weights, PPO-trained on 4 grid games | All weights as text, no tools | No |
| 2 | Name the shape a diffusion model generates | MLP, 23,824 weights, 3 DDPMs (circle/square/triangle) | Weights as text, no tools, no forward pass | No |
| 3 | Name the language a char-level LM was trained on, and which heads are induction (copying) heads | 3 char-level transformers, 7.2M params, 4 layers × 6 heads. Embedding and output layers removed | Middle blocks only, Python allowed for analysis, up to 100 runs, not allowed to run the network | No forward passes |
| 4 | Fix a PPO policy stuck in a local optimum (Kinetix car-on-ramp) | 171,664 weights | Weights, scene description, one recorded episode, single reply, no tools | No feedback. One shot |
| 5 | Write weights from scratch for 3 MinAtar games | 600→32→32→6, all weights start at zero | Game rules and source code, no tools, no rollouts | No feedback. One shot |
| 6 | Write weights from scratch for 9×9 Go | 243→512×6→82, about 1.48M weights | Python sandbox, 0 to 50 forward passes (128 rollouts each) | **Yes, when forward passes > 0** |
| 7 | Identify which of two Gemma 3 270M fine-tunes is sycophantic | Same recipe, embeddings and output layer frozen. 3 pairs | Weights alone, then relaxed to attention patterns from LLM-written texts | Weights only: failed. Relaxed version uses model runs |
| 8 | Make Qwen3.5-0.8B better at math | 0.8B params, thinking on | Python kernel over the weights, training-set problems, **10 / 25 / 50 forward passes** | **Yes** |

### 1.3 Key results (copied from the post)

**Experiments 1–3 (reading).** Astra named the right game for all 4 networks, the right shape for the diffusion models, and the right language for all 3 char-LMs in 2 attempts each. In Exp 3 its confidence was only about 0.5–0.6. It also listed exactly the induction heads that were measured by running the networks: Python 2.1, 3.1, 3.5; romaji 3.3; English 3.1.

**Experiment 4 (fixing).** One edit of 3.5% of the weights took the car from 0% to 100% success (1,024 episodes).

**Experiment 5 (writing, MinAtar).**

| Game | PPO 100M steps | PPO 500M steps | Astra, one-shot |
|------|---------------|----------------|-----------------|
| Breakout | 15.8 | 25.3 | 517.4 |
| Seaquest | 5.6 | 88.9 | 53.1 |
| Asterix | 25.8 | 55.2 | 53.2 |

**Experiment 6 (Go).** With 0 forward passes, the written network wins more than half its games against two small PPO networks. With 50 forward passes it beats an equal-sized PPO network trained on 20M games.

**Experiment 7 (sycophancy).** From weights alone, Astra failed in all 4 attempts. With attention patterns from texts it wrote itself, it picked the sycophant in 3 of 3 pairs. Head 10.2 attends to the user's suggested answer: 85.7% in the sycophant vs 3.5% in the honest model.

**Experiment 8 (Qwen3.5-0.8B, the most relevant one).**
- Setup:
  - Thinking mode on, up to 32,768 tokens.
  - Sampling: temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5. The same sampling is used for base and edited models.
  - The model card warns that this model is prone to thinking loops.
  - Practice problems come from the GSM8K and MATH training sets. Evaluation is on MATH-500, GSM8K, IFEval and others.
- Diagnosis by Astra: looping was traced to one direction written by the MLPs of layers 15 and 19. The edit subtracts that direction from 64 output columns in each of those two layers, which is 131,072 weights, or 0.016% of the model.
- Results:

| Condition | MATH-500 % | GSM8K % | IFEval % | Hit 32k token limit % |
|-----------|-----------|---------|----------|-----------------------|
| Base | 7.9 | 13.8 | 21.0 | 93.0 |
| Edited, 10 fwd passes | 12.3 | 25.7 | 49.0 | 87.2 |
| Edited, 25 fwd passes | 14.1 | 25.1 | 39.9 | 86.4 |
| Edited, 50 fwd passes | 48.1 | 67.1 | 62.3 | 33.1 |

- Number of samples per problem: 16 on AMC23, 4 on MATH-500, Minerva Math, OlympiadBench and GPQA Diamond, 2 on GSM8K and IFEval, 1 on MMLU-Pro (fixed 1,000 questions).

**Additional experiment (Gemma 3 270M, a model that rarely loops).** Only 8.5% of answers hit the limit. Astra could not improve MATH-500: scores stayed between 6.8 and 7.7 across 0 to 50 forward passes, and Astra itself reported the gains as not established. Authors' own conclusion: editing can remove an obstacle that hid an ability, but did not add an ability the model lacked.

### 1.4 What the authors say they did not do
- No gradients and no training loops by the LLM in the main experiments.
- In the authors' rebuttal about turning thinking off, they argue that turning thinking off would avoid the loop by not using the mode that has it, and so it "does not fix the weights".
- They have not compared Astra to human interpretability researchers.
- They say a benchmark and harness are being built.

---

## 2. SE.LLMA, digested

- WASP (Wallenberg AI, Autonomous Systems and Software Program) project, funded by the Knut and Alice Wallenberg Foundation. Runs for two years.
- Led by researchers at Linköping, Umeå and Uppsala universities, together with NAISS and representatives from media, authors, publishers and other rights holders.
- **Goal:** train Swedish language models with high-quality Swedish performance, evaluated on Swedish culture, history, societal principles and norms.
- **Approach:** not training from scratch. Build on existing open models that are transparent about data and weights and comply with legal requirements. Train iteratively, from smaller to larger models. Several domain-specialized models.
- **Data:** public-sector documents plus editorial and literary material, used with rights-holder agreements. Data is used only inside the project.
- **Stated goals relevant to evaluation:**
  - Build evaluation and benchmarking workflows specific to Swedish. A key question is how to capture the finer nuances of Swedish and adapt to Swedish cultural context.
  - Measure how much high-quality Swedish data improves small and large models, with results published continuously.
  - Build competence for Swedish adaptations of state-of-the-art models, including reasoning and multimodal ones.
  - Establish licensing and governance frameworks.

**Why this matters for choosing a hypothesis.** SE.LLMA adapts existing open models, so cheap, targeted ways to diagnose and repair a base model's Swedish behavior are potentially relevant. The project also values transparency, traceability and evaluation, which favors work that is careful about measurement. The project values copyright-clean data, so use only data and models you have the right to use and share.

---

## 3. Critical-reading seeds (analysis by the assistant, verify before use)

### 3.1 Strongest evidence
- **Exp 3 (induction heads).** The ground truth is measured independently, by running the network on a repeated random string. The LLM never sees it and was not allowed to run the model. The match is exact in 3 of 3 models. The method it rediscovered (composing weight matrices) is a known interpretability technique. Caveat: only 3 models, and the same architecture and training setup for all.
- **Exp 7.** Honest negative: weights alone failed, and success appeared only after relaxing to behavior-derived attention patterns. This is the clearest statement of where direct weight reading stopped working.
- **Gemma control.** An honest null that supports the framing that edits remove failures and do not add capabilities.
- **Public logs.** Everything is released, which makes independent checking possible.

### 3.2 Most important limitations and open questions
1. **Feedback optimization vs weight reasoning.** Experiment 8 (and 6 with forward passes) gives the LLM repeated model feedback through forward passes on practice problems. The result is a search-and-evaluate loop over edits, run by an LLM. "No gradients" does not mean "no feedback optimization". The post itself notes that Astra "keeps a change only if it wins more" in Exp 6. Experiments 1–5 are closer to pure weight reasoning, but they concern small purpose-built networks.
2. **Non-monotonic budget curve in Exp 8.** MATH-500 goes 7.9 → 12.3 → 14.1 → 48.1 across 0/10/25/50 forward passes. The jump from 25 to 50 is large and unexplained. The number of independent edit runs per budget is not stated in the post **[VERIFY in the logs]**. If each budget is one trajectory, this is one draw per condition, with no seed variance.
3. **Pathological baseline.** The base model hits the 32k token limit on 93% of MATH-500 answers, with a presence penalty of 1.5. A large gain from removing a looping failure is therefore expected to look dramatic. The absolute numbers say little about "math ability". What matters is whether the edit is better than cheap alternatives that also remove looping.
4. **Missing controls in Exp 8.** The post does not report:
   - a random-direction edit of matched norm and matched layers and columns;
   - a standard activation-steering or direction-ablation baseline using the same layers;
   - changed inference settings (a lower presence penalty or repetition penalty, a shorter max length with a forced answer, or non-thinking mode, which the authors discuss but argue does not count);
   - a conventional fine-tune or LoRA on the same practice problems;
   - regression measurements beyond the benchmarks reported (e.g. perplexity, or non-reasoning tasks).
5. **Failure-removal vs capability.** The authors state this distinction themselves. IFEval tripling is plausibly the same loop removal and not new instruction-following ability. The assignment asks exactly this question.
6. **Selection and contamination risks.** Practice problems come from train sets, which is good. But the model editor is a closed frontier model whose training data may include the Qwen model card, public loop-direction literature, and possibly the evaluation benchmarks. The experiment cannot be re-run by outsiders with the same editor.
7. **Small-N and cherry-picking risks in Exps 1–5.** Best-of-3 PPO seeds in Exp 1, two attempts in Exp 3, 3 pairs in Exp 7, one network and one edit in Exp 4. Baselines in Exp 5 are PPO at a small budget, and there is no hand-coded-policy baseline. A game solved by a simple tracking rule should be easy for a programmer who reads the source code.
8. **Conflation risk.** The post's headline sentence is that weights can be "read and written natively". Several of the best results give the LLM the source code, the game rules, or the scene description. That is closer to "LLM compiles a spec into a network" than "LLM reads an unknown network".
9. **Scale and generality.** Real-LM evidence is one model (Qwen3.5-0.8B) and one success mode (loops). The one other real LM (Gemma 270M) was a null.
10. **Replication claims.** The post lists many replications and a Hugging Face demo. **[VERIFY]** how independent they are and whether they re-ran the editing, or only re-evaluated the released edited weights.

### 3.3 Conclusions that look justified vs not (suggested wording to test)
- *Justified:* a frontier LLM can, in small and structured settings, recover facts about a network from its weights, and can produce a useful edit to a stuck network.
- *Justified:* in one case (Qwen 0.8B), a very small edit can sharply reduce a looping failure under the stated sampling settings.
- *Not established:* that the Exp 8 edit is better than simpler alternatives; that it generalizes to other models, languages or failure types; that the process is "weight reasoning" and not feedback search; that gains reflect new capability.
- *Not established at all:* that anything here transfers to Swedish, or to larger production models.

---

## 4. Practical facts for planning the experiment

- **You cannot use "Astra".** Realistic stand-ins for the "weight programmer":
  1. a human or scripted procedure (deterministic, directly derived from weights, e.g. a weight-diff or SVD analysis between a base and a Swedish-adapted checkpoint);
  2. Claude Code or another agent in a Python kernel with a hard, logged budget of forward passes (a miniature of Exp 8; report the budget and every edit attempted);
  3. a conventional method as the comparison (LoRA, steering vector, direction ablation).
  If you use an agent as the editor, record its full transcript and the number of forward passes it spent. That makes the "direct weights reasoning vs feedback optimization" distinction auditable.
- **Compute:** target Google Colab Free (single T4, about 15 GB VRAM, about 12 GB RAM, sessions may disconnect). T4 has no bf16. Prefer small models (≤ about 1.5B parameters), short generations, and resumable runs. Smoke-test on CPU first.
- **Models mentioned in the paper:** Qwen3.5-0.8B and Gemma 3 270M. **[VERIFY]** each model's license, its exact revision, and whether the weights fit Colab with your generation lengths. Thinking-mode generations of up to 32k tokens are far too long for a free T4, so any loop-related experiment must use a much shorter cap and say so.
- **Data:** public Swedish datasets or synthetic ones. **[VERIFY]** licenses. Candidates to check (not confirmed here): EuroEval/ScandEval Swedish tasks, SuperLim, FLORES Swedish, Swedish Wikipedia. Template-generated Swedish grammar items (en/ett agreement, definite forms, V2 order) have exact ground truth.

---

## 5. Candidate narrow hypotheses (seeds, not instructions)

Choose one, or replace with your own. Each should be pre-registered in `PLAN.md` before running test data.

1. **Does a loop-removal edit transfer across languages?** For a small reasoning model that loops, does a looping-direction edit found with English practice problems also reduce looping on Swedish prompts, and is the Swedish accuracy gain explained by finishing answers (failure removal) and not by new Swedish ability? Controls: random-direction edit, steering-vector ablation, a changed repetition penalty, and a no-edit baseline.
2. **Wrong-language drift.** Small multilingual models sometimes answer Swedish prompts in English. Can a targeted edit reduce this without hurting English and Swedish perplexity? Controls: Swedish system prompt, forced-language decoding, LoRA on a small dev set.
3. **Locating Swedish behavior in weights.** Using a base model and a Swedish-adapted checkpoint (if available and licensed), can weight-difference analysis identify layers or components whose ablation or transplant moves Swedish behavior more than matched random components do? This is a directly weight-based analysis with no model feedback needed to find the candidates, and feedback is only used to evaluate.
4. **Failure removal vs new capability in Swedish.** Design an evaluation that separates "model could answer but failed (language drift, formatting, early stop)" from "model cannot answer", then test whether any edit shifts only the former.
5. **Swedish grammar probes.** Synthetic en/ett and definite-suffix tests, checking whether the paper's weight-reading style analysis (e.g. spotting heads or neurons) can predict which components matter, validated by ablation.

For whichever you pick, state in advance: the primary metric, the support and refute thresholds, at least one regression metric (English ability or Swedish perplexity on held-out text), the number of configurations you tried, and which steps were weight-reasoning and which were feedback-driven.

---

## 6. Checklist of what a strong submission shows (mirrors the grading criteria)

- Scientific judgment: separates strong from weak evidence in the paper (see 3.1 and 3.2) and avoids over-claiming from a small-model result.
- Experimental design: pre-registered hypothesis, held-out test split touched once, baseline plus prompt/inference control plus random-edit control plus a conventional alternative.
- Technical execution: runs end to end, within Colab Free limits, with logged seeds and pinned revisions.
- Independent thinking: a defensible choice of question, and a recommendation proportionate to the evidence, including "do not pursue" if that is what the data say.
- Communication: a 1–2 page note, a PDF report, a reproducible repo, and an honest AI-usage statement.
