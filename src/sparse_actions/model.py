"""Model / tokenizer / LoRA loading and decision-token id resolution."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Left padding => the decision position is index -1 for every row in a batch,
    # which is exactly where we read/train the gate logit.
    tok.padding_side = "left"
    return tok


def pick_token_id(tok, word: str) -> int:
    """Resolve `word` to a single vocab id, preferring the leading-space variant
    (that is how it appears mid-sequence after the 'Decision:' cue). Warn if the
    word is not single-token -- the analytic readout assumes a one-token gate."""
    for cand in (" " + word, word):
        ids = tok(cand, add_special_tokens=False).input_ids
        if len(ids) == 1:
            return ids[0]
    ids = tok(" " + word, add_special_tokens=False).input_ids
    print(
        f"[model] WARNING: gate word {word!r} is not single-token "
        f"({ids}); using first id {ids[0]}. Pick a single-token word for exactness."
    )
    return ids[0]


def decision_ids(tok, cfg) -> tuple[int, int]:
    """(safe_id, action_id) for the gate tokens."""
    return pick_token_id(tok, cfg.tokens.safe_token), pick_token_id(tok, cfg.tokens.action_token)


def load_model(cfg, adapter_dir: str | None = None, train: bool = False, device: str = "cuda"):
    dtype = getattr(torch, cfg.model.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
    )
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=train)
    elif train:
        from peft import LoraConfig, get_peft_model

        lc = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=list(cfg.lora.target_modules),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lc)
        model.print_trainable_parameters()
    model.to(device)
    if train:
        model.config.use_cache = False
    return model
