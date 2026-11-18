"""
Model loading utilities.

Loads Qwen2.5-0.5B-Instruct in fp16 (or fp32 on CPU) and pins the revision.
"""

import json
from pathlib import Path
from typing import Optional

import torch

CONFIG_DIR = Path(__file__).parent.parent / "configs"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(
    model_id: str = MODEL_ID,
    revision: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
):
    """
    Load model and tokenizer, pin revision, and record to configs/model_config.json.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or get_device()
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    # Pin and record revision
    actual_revision = getattr(model.config, "_commit_hash", None) or revision or "unknown"
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_file = CONFIG_DIR / "model_config.json"
    cfg = {
        "model_id": model_id,
        "revision": actual_revision,
        "dtype": str(dtype),
        "device": device,
    }
    with open(cfg_file, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Model loaded: {model_id}@{actual_revision} on {device} as {dtype}")
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    device: Optional[str] = None,
    batch_size: int = 4,
) -> list[str]:
    """
    Generate responses for a list of prompts.

    Uses greedy decoding (temperature=0) by default for reproducibility.
    """
    device = device or get_device()
    all_responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        # Apply chat template if available
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
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            if temperature == 0.0:
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )

        for seq in out:
            new_tokens = seq[input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            all_responses.append(response)

    return all_responses
