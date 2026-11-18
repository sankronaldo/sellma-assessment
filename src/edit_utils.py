"""
Weight-edit utilities.

Implements:
  1. Activation-based language direction finding (feedback-based — declared as such)
  2. Direction ablation applied to MLP output projection weights
  3. Random-direction control edits
  4. Sign-flipped control edit
  5. All edits are applied in-place to a copy of the model's state_dict.
"""

import copy
import json
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_layer_activations(
    model,
    tokenizer,
    prompts: list[str],
    layer_indices: Optional[list[int]] = None,
    device: str = "cpu",
    max_length: int = 128,
) -> dict[int, torch.Tensor]:
    """
    Extract residual-stream activations (after each transformer layer) for a list
    of prompts. Returns a dict mapping layer index -> Tensor[n_prompts, hidden_dim].

    Uses the last token position as the "response direction" probe.

    This function RUNS THE MODEL (feedback-based) and must be declared as such.
    """
    from transformers.modeling_outputs import BaseModelOutputWithPast

    if layer_indices is None:
        # Default: all layers
        try:
            n_layers = model.config.num_hidden_layers
        except AttributeError:
            n_layers = len(model.model.layers)
        layer_indices = list(range(n_layers))

    # Register forward hooks
    activations: dict[int, list] = {l: [] for l in layer_indices}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output may be a tuple; take the first element (hidden states)
            hidden = output[0] if isinstance(output, tuple) else output
            # Take last-token activation, shape [batch, hidden_dim]
            activations[layer_idx].append(hidden[:, -1, :].detach().cpu())
        return hook

    layers = model.model.layers
    for li in layer_indices:
        h = layers[li].register_forward_hook(make_hook(li))
        hooks.append(h)

    model.eval()
    for i in range(0, len(prompts), 4):
        batch = prompts[i : i + 4]
        if hasattr(tokenizer, "apply_chat_template"):
            formatted = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch
            ]
        else:
            formatted = batch
        enc = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            model(**enc)

    for h in hooks:
        h.remove()

    return {l: torch.cat(activations[l], dim=0) for l in layer_indices}


# ---------------------------------------------------------------------------
# Language direction computation
# ---------------------------------------------------------------------------

def compute_language_direction(
    model,
    tokenizer,
    sv_prompts: list[str],
    en_prompts: list[str],
    device: str = "cpu",
) -> dict[int, torch.Tensor]:
    """
    Compute per-layer language direction as mean(Swedish) - mean(English) activations.

    Returns dict mapping layer index -> direction vector [hidden_dim].

    NOTE: This is a FEEDBACK-BASED step (runs the model). Declared as such.
    """
    n_layers = model.config.num_hidden_layers
    layer_indices = list(range(n_layers))

    sv_acts = extract_layer_activations(
        model, tokenizer, sv_prompts, layer_indices, device
    )
    en_acts = extract_layer_activations(
        model, tokenizer, en_prompts, layer_indices, device
    )

    directions = {}
    for l in layer_indices:
        d = sv_acts[l].mean(0) - en_acts[l].mean(0)
        d = d / (d.norm() + 1e-8)
        directions[l] = d.to(dtype=torch.float32)

    return directions


def layer_direction_scores(
    directions: dict[int, torch.Tensor],
    model,
) -> list[tuple[int, float]]:
    """
    Rank layers by how much the language direction aligns with the MLP down-projection
    weight columns (a heuristic for where the direction is "written" into the residual
    stream by MLP output projections).

    Returns list of (layer_idx, score) sorted descending.
    """
    scores = []
    for li, d in directions.items():
        try:
            mlp = model.model.layers[li].mlp
            # down_proj is the final linear in the MLP block
            W = mlp.down_proj.weight.data.float().cpu()  # [hidden_dim, intermediate]
            d_cpu = d.float().cpu()
            # Score: mean absolute cosine similarity between direction and each column
            d_norm = d_cpu / (d_cpu.norm() + 1e-8)
            # W columns: each column is a direction added to the residual stream
            W_col_norms = W.norm(dim=0, keepdim=True) + 1e-8
            W_normed = W / W_col_norms
            similarities = (W_normed.T @ d_norm).abs()  # [intermediate]
            scores.append((li, float(similarities.mean())))
        except AttributeError:
            scores.append((li, 0.0))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ---------------------------------------------------------------------------
# Direction ablation edit
# ---------------------------------------------------------------------------

def apply_direction_ablation(
    model,
    directions: dict[int, torch.Tensor],
    target_layers: list[int],
    alpha: float = 1.0,
    n_columns: int = 64,
    inplace: bool = False,
) -> object:
    """
    For each target layer, subtract alpha * d from the top-n_columns output columns
    of the MLP down_proj weight matrix that have the highest alignment with d.

    This is DIRECT WEIGHT SURGERY — applied once, no feedback.

    Returns the (optionally modified in-place) model.
    """
    if not inplace:
        model = copy.deepcopy(model)

    with torch.no_grad():
        for li in target_layers:
            d = directions[li].to(device=model.device, dtype=model.dtype)
            d = d / (d.norm() + 1e-8)
            mlp = model.model.layers[li].mlp
            W = mlp.down_proj.weight  # [hidden_dim, intermediate]

            # Rank columns by alignment with direction
            W_col_norms = W.data.float().norm(dim=0) + 1e-8
            W_normed = W.data.float() / W_col_norms
            sims = (W_normed.T @ d.float()).abs()  # [intermediate]
            top_cols = sims.topk(min(n_columns, sims.shape[0])).indices

            # Subtract direction from selected columns
            # Each column j of W (shape [hidden_dim]) → W[:,j] -= alpha * d
            d_edit = (alpha * d.to(dtype=W.dtype)).unsqueeze(1)  # [hidden_dim, 1]
            W.data[:, top_cols] -= d_edit.expand(-1, top_cols.shape[0])

    return model


# ---------------------------------------------------------------------------
# Random-direction control
# ---------------------------------------------------------------------------

def apply_random_direction_edit(
    model,
    reference_direction: torch.Tensor,
    target_layers: list[int],
    alpha: float = 1.0,
    n_columns: int = 64,
    seed: int = 0,
) -> object:
    """
    Apply an edit with a random direction of matched norm to target_layers.
    Returns a deepcopy of the model.
    """
    torch.manual_seed(seed)
    hidden_dim = reference_direction.shape[0]
    rand_d = torch.randn(hidden_dim)
    rand_d = rand_d / (rand_d.norm() + 1e-8)
    rand_directions = {l: rand_d for l in target_layers}
    return apply_direction_ablation(
        model, rand_directions, target_layers, alpha=alpha, n_columns=n_columns
    )


def apply_sign_flipped_edit(
    model,
    directions: dict[int, torch.Tensor],
    target_layers: list[int],
    alpha: float = 1.0,
    n_columns: int = 64,
) -> object:
    """Apply the negative (sign-flipped) direction as a control."""
    flipped = {l: -d for l, d in directions.items()}
    return apply_direction_ablation(
        model, flipped, target_layers, alpha=alpha, n_columns=n_columns
    )


# ---------------------------------------------------------------------------
# Edit accounting
# ---------------------------------------------------------------------------

def count_edited_params(
    base_model, edited_model, threshold: float = 1e-9
) -> dict:
    """Count parameters that differ between base and edited model."""
    total = 0
    n_changed = 0
    for (name, p_base), (_, p_edit) in zip(
        base_model.named_parameters(), edited_model.named_parameters()
    ):
        total += p_base.numel()
        n_changed += (p_base.data - p_edit.data).abs().gt(threshold).sum().item()
    return {
        "total_params": total,
        "changed_params": n_changed,
        "fraction_changed": n_changed / total if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Save / load edit config
# ---------------------------------------------------------------------------

def save_edit_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_edit_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
