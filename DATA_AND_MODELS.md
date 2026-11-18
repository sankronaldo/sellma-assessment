# Data and Models

## Model

| Name | Source | License | Revision | Notes |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | ModelScope git clone (`https://modelscope.cn/qwen/Qwen2.5-0.5B-Instruct.git`) | Apache 2.0 | See `configs/model_config.json` | 494M params, fp16, ~1 GB on disk |

The model was downloaded via `git clone` from ModelScope (HuggingFace was inaccessible
on the development machine). The downloaded weights are identical to those on HuggingFace
under `Qwen/Qwen2.5-0.5B-Instruct`.

## Data

| Name | Source | License | Used for |
|---|---|---|---|
| Synthetic Swedish sentences (50 dev, 99 test) | Hardcoded in `scripts/run_local.py` | Project-owned | Primary evaluation — Swedish language-of-response |
| Synthetic English sentences (50 dev, 99 test) | Hardcoded in `scripts/run_local.py` | Project-owned | English regression test |
| Synthetic Swedish perplexity sentences (20) | Hardcoded in `scripts/run_local.py` | Project-owned | Swedish perplexity regression |

All data is synthetic — factual Swedish and English sentences covering a range of topics
(nature, history, society, rural life). No external dataset download was required.
The sentences are entirely contained in `scripts/run_local.py` and can be inspected there.

The dev and test splits are disjoint by construction (different sentence sets, different
ID prefixes). Split configuration is saved in `configs/data_split.json`.

## Licensing notes

- Apache 2.0 (Qwen2.5-0.5B-Instruct): permits use, modification, and distribution,
  including commercial use. LoRA adapters trained on this model may be shared under
  Apache 2.0.
- Synthetic sentences: generated for this project; no external license constraints.
