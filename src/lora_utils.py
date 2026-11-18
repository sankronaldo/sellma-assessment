"""
LoRA fine-tuning baseline (Condition C6).

Trains a lightweight LoRA adapter on the same dev prompts used to identify
the language direction. Matches the data budget with the direction-ablation method.
"""

import json
from pathlib import Path
from typing import Optional

import torch

RESULTS_DIR = Path(__file__).parent.parent / "results"
CONFIG_DIR = Path(__file__).parent.parent / "configs"


def train_lora_baseline(
    model,
    tokenizer,
    dev_prompts: list[dict],
    out_dir: Optional[Path] = None,
    num_epochs: int = 3,
    lr: float = 2e-4,
    r: int = 8,
    lora_alpha: int = 16,
    seed: int = 42,
    device: Optional[str] = None,
) -> object:
    """
    Fine-tune a LoRA adapter on dev_prompts using the expected-language responses
    as supervision signals. The training signal is: produce the source text in
    the expected language (a form of supervised language-output training).

    Returns the adapted model.
    """
    from peft import get_peft_model, LoraConfig, TaskType
    import torch.nn.functional as F

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = out_dir or RESULTS_DIR / "lora_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build training examples: prompt -> expected Swedish output (use source_text as target)
    examples = []
    for p in dev_prompts:
        expected_lang = p.get("expected_lang", "sv")
        if expected_lang == "sv":
            if hasattr(tokenizer, "apply_chat_template"):
                input_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                input_text = p["prompt"]
            # Target: source text (the Swedish sentence itself)
            target_text = p.get("source_text", "")
            examples.append({"input": input_text, "target": target_text})

    if not examples:
        print("No training examples for LoRA baseline; skipping.")
        return model

    # Configure LoRA
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    peft_model.to(device)

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr)

    peft_model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for ex in examples:
            full_text = ex["input"] + ex["target"] + tokenizer.eos_token
            enc = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            input_ids = enc["input_ids"].to(device)
            # Build labels: mask input tokens with -100
            input_only_len = len(tokenizer(ex["input"], return_tensors="pt")["input_ids"][0])
            labels = input_ids.clone()
            labels[:, :input_only_len] = -100

            loss = peft_model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(examples))
        print(f"  LoRA epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.4f}")

    peft_model.eval()

    # Save adapter
    peft_model.save_pretrained(str(out_dir))
    cfg = {
        "r": r,
        "lora_alpha": lora_alpha,
        "num_epochs": num_epochs,
        "lr": lr,
        "seed": seed,
        "n_examples": len(examples),
    }
    with open(CONFIG_DIR / "lora_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"LoRA adapter saved to {out_dir}")

    return peft_model
